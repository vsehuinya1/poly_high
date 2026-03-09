"""
Strategy B — Inflection Sniping for Tennis.

Two structural triggers that exploit market microstructure inefficiencies
during high-leverage tennis moments:
    Trigger 1: Panic Discount — break point with favorite serving
    Trigger 2: Set Mean Reversion — favorite down 0-1 in sets

No ML. Pure structural signals based on Markov fair value vs market.

v2.0 — added pre-evaluation guards:
    - Hard market price floor (dead market filter)
    - State dedup via composite key (prevents signal spam)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from tennis.state import TennisState, TennisModelOutput, compute_momentum_delta
from tennis.model import get_win_prob

log = logging.getLogger("tennis.strategy")


# ═══════════════════════════════════════════════════════════════════════
#  Signal Object
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TennisSignal:
    """Detected edge signal from a Strategy B trigger."""
    timestamp: float
    match_id: str
    trigger_type: str              # "PANIC_DISCOUNT" | "SET_MEAN_REVERSION"
    edge: float                    # fair_price - market_price (positive = underpriced)
    fair_price: float
    market_price: float
    state_snapshot: TennisState
    model_output: TennisModelOutput
    momentum_delta: float = 0.0

    def __str__(self) -> str:
        return (
            f"TENNIS_SIGNAL {self.trigger_type} | "
            f"edge={self.edge:+.4f} fair={self.fair_price:.4f} "
            f"mkt={self.market_price:.4f} | "
            f"{self.state_snapshot}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Strategy B Engine
# ═══════════════════════════════════════════════════════════════════════

class InflectionStrategy:
    """Scans TennisState for Strategy B inflection triggers.

    Guards (pre-evaluation):
        - Dead market filter: market_price < price_floor → suppress.
        - State dedup: identical state key → suppress.

    Triggers:
        - Panic Discount: break point with favorite serving, market underprice.
        - Set Mean Reversion: favorite down 0-1, market over-adjusts.

    Only produces signals; execution gating is handled separately.

    Args:
        panic_edge_threshold:     Minimum edge for Trigger 1.
        reversion_edge_threshold: Minimum edge for Trigger 2.
        price_floor:              Minimum market price to evaluate (default 0.05).
    """

    def __init__(self, panic_edge_threshold: float = 0.06,
                 reversion_edge_threshold: float = 0.05,
                 price_floor: float = 0.05):
        self.panic_edge = panic_edge_threshold
        self.reversion_edge = reversion_edge_threshold
        self.price_floor = price_floor

        # ── State dedup tracking ──────────────────────────────────
        # key: (match_id, selection_key) → last state_key that produced a signal
        self._last_signal_state: dict[str, str] = {}

        # ── Dead market logging (log-once-per-match) ──────────────
        self._dead_market_logged: set[str] = set()

        # ── Edge persistence tracker (Steps 2-4) ─────────────────
        # key: (match_id, selection_id) → (first_ts, edge_value, tick_count)
        # Prevents early entries by requiring edge to persist for N ticks.
        self._edge_persistence: dict[tuple[str, str], tuple[float, float, int]] = {}

    # ── State Key ─────────────────────────────────────────────────

    @staticmethod
    def _state_key(state: TennisState, selection_id: str = "") -> str:
        """Composite key for state dedup. Includes selection_id to prevent
        cross-runner contamination."""
        return (
            f"{state.match_id}:{selection_id}:"
            f"{state.sets_a}-{state.sets_b}:"
            f"{state.games_a}-{state.games_b}:"
            f"{state.point_a.value}-{state.point_b.value}:"
            f"{state.server_id}"
        )

    # ── Main Evaluation ───────────────────────────────────────────

    def evaluate(self, state: TennisState,
                 market_price: float,
                 selection_id: str = "") -> Optional[TennisSignal]:
        """Evaluate both triggers against current state.

        Pre-evaluation guards are applied BEFORE the Markov model call:
            1. Market price floor (dead book filter).
            2. State dedup (identical state suppression).

        Args:
            state:         Current match state snapshot.
            market_price:  Current Polymarket price for the pre-game favorite.
            selection_id:  Optional runner/selection ID for cross-runner safety.

        Returns:
            TennisSignal if a trigger fires, else None.
        """
        # ── Guard 0: Invalid price ────────────────────────────────
        if market_price <= 0 or market_price >= 1:
            return None

        # ── Guard 1: Dead market / penny book (BEFORE edge calc) ──
        if market_price < self.price_floor:
            if state.match_id not in self._dead_market_logged:
                log.info("DEAD_MARKET | match=%s mkt=%.4f < floor=%.2f — suppressed",
                         state.match_id, market_price, self.price_floor)
                self._dead_market_logged.add(state.match_id)
            return None

        # ── Guard 2: State dedup ──────────────────────────────────
        key = self._state_key(state, selection_id)
        dedup_key = f"{state.match_id}:{selection_id}"
        if self._last_signal_state.get(dedup_key) == key:
            # Identical state — suppress (no model call, no log spam)
            return None

        # ── Markov model call ─────────────────────────────────────
        model = get_win_prob(state)

        # Determine fair price for the pre-game favorite
        if state.pregame_favorite_id == state.player_a_id:
            fair_fav = model.p_a
        else:
            fair_fav = model.p_b

        momentum = compute_momentum_delta(state)

        # ── Trigger 1: Panic Discount ─────────────────────────────
        sig = self._check_panic_discount(state, fair_fav, market_price,
                                          model, momentum)
        if sig is not None:
            self._last_signal_state[dedup_key] = key
            return sig

        # ── Trigger 2: Set Mean Reversion ─────────────────────────
        sig = self._check_set_mean_reversion(state, fair_fav, market_price,
                                              model, momentum)
        if sig is not None:
            self._last_signal_state[dedup_key] = key
            return sig

        # No trigger but state was new — update dedup to avoid
        # re-running the model on the same state next tick
        self._last_signal_state[dedup_key] = key
        return None

    def _check_panic_discount(self, state: TennisState,
                               fair_fav: float, market_price: float,
                               model: TennisModelOutput,
                               momentum: float) -> Optional[TennisSignal]:
        """Trigger 1: Panic Discount.

        Fires when:
            1. It's a break point (returner can win the game).
            2. The server IS the pre-game favorite.
            3. The market underprices the favorite beyond threshold.

        Reasoning:
            Markets over-react to break points against favorites.
            The structural probability of holding serve from break point
            is still substantial (~36% even at 30-40), and the match-win
            impact is less than the market typically prices.
        """
        if not state.is_break_point:
            return None

        if not state.favorite_is_serving:
            return None

        edge = fair_fav - market_price
        if edge < self.panic_edge:
            return None

        log.info("PANIC_DISCOUNT triggered | edge=%.4f fair=%.4f mkt=%.4f | %s",
                 edge, fair_fav, market_price, state)

        return TennisSignal(
            timestamp=time.time(),
            match_id=state.match_id,
            trigger_type="PANIC_DISCOUNT",
            edge=edge,
            fair_price=fair_fav,
            market_price=market_price,
            state_snapshot=state,
            model_output=model,
            momentum_delta=momentum,
        )

    def _check_set_mean_reversion(self, state: TennisState,
                                   fair_fav: float, market_price: float,
                                   model: TennisModelOutput,
                                   momentum: float) -> Optional[TennisSignal]:
        """Trigger 2: Set Mean Reversion.

        Fires when:
            1. The pre-game favorite is down 0-1 in sets.
            2. Not currently in a tiebreak.
            3. Fair value > market price by threshold.
            4. Edge has persisted for required ticks (2 normal, 3 extreme).

        Reasoning:
            Losing the first set often causes >15% market drop for the
            favorite, but structural match-win probability for a strong
            server is typically only ~8-12% lower after losing set 1.
            The market over-adjusts.
        """
        if state.is_tiebreak:
            return None

        # Favorite must be down exactly 0-1
        if state.favorite_sets != 0 or state.underdog_sets != 1:
            return None

        edge = fair_fav - market_price

        # ── Persistence tracker key ──────────────────────────────
        persist_key = (state.match_id, state.pregame_favorite_id or "")

        # ── Safe reset: edge dropped below threshold ─────────────
        if edge < self.reversion_edge:
            if persist_key in self._edge_persistence:
                del self._edge_persistence[persist_key]
            return None

        # ── Update or create persistence entry ───────────────────
        now = time.time()
        if persist_key in self._edge_persistence:
            prev_ts, prev_edge, prev_count = self._edge_persistence[persist_key]
            self._edge_persistence[persist_key] = (prev_ts, edge, prev_count + 1)
        else:
            self._edge_persistence[persist_key] = (now, edge, 1)

        _, _, tick_count = self._edge_persistence[persist_key]

        # ── Required tick count (extreme edge safety) ────────────
        required_ticks = 3 if edge > 0.30 else 2

        if tick_count < required_ticks:
            if tick_count == 1:
                log.info("SET_MR_PENDING | edge=%.4f | ticks=%d/%d | %s",
                         edge, tick_count, required_ticks, state.match_id)
            return None

        # ── Edge has persisted — fire signal ──────────────────────
        # Clear persistence tracker (one signal per persistence window)
        del self._edge_persistence[persist_key]

        log.info("SET_MEAN_REVERSION triggered | edge=%.4f fair=%.4f mkt=%.4f | ticks=%d | %s",
                 edge, fair_fav, market_price, tick_count, state)

        return TennisSignal(
            timestamp=time.time(),
            match_id=state.match_id,
            trigger_type="SET_MEAN_REVERSION",
            edge=edge,
            fair_price=fair_fav,
            market_price=market_price,
            state_snapshot=state,
            model_output=model,
            momentum_delta=momentum,
        )
