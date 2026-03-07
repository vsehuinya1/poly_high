"""
Cricket Strategy — Three Signal Types.

Signal A: Momentum Edge — 3-over rolling RR > required RR + threshold
Signal B: Wicket Overreaction — wicket event with run rate still strong
Signal C: Latency Snipe — boundary with slow market update (log-only)

Also includes Gemini hypotheses (log-only):
    - Pitch Bias: spinner economy tracking
    - Anchor Trap: overweighted wicket events
    - Boundary Latency: event-to-price lag measurement
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from cricket.state import CricketState, BoundaryEvent, CricketModelOutput
from cricket.model import get_win_prob

log = logging.getLogger("cricket.strategy")


# ═══════════════════════════════════════════════════════════════════════
#  Signal Object
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CricketSignal:
    """Detected edge signal from a cricket strategy trigger."""
    timestamp: float
    match_id: str
    signal_type: str         # "MOMENTUM_EDGE", "WICKET_OVERREACTION", "LATENCY_SNIPE"
    edge: float              # fair_price - market_price
    fair_price: float        # model probability
    market_price: float      # current market probability
    state_snapshot: CricketState
    model_output: CricketModelOutput
    latency_ms: float = 0.0  # for LATENCY_SNIPE
    is_tradeable: bool = True  # False for log-only signals (LATENCY_SNIPE)

    # Gemini hypothesis flags
    pitch_bias_active: bool = False
    anchor_trap_active: bool = False

    @property
    def direction(self) -> str:
        """BUY batting team or SELL batting team."""
        return "BUY" if self.edge > 0 else "SELL"

    def __str__(self) -> str:
        return (
            f"CRICKET_{self.signal_type} | {self.state_snapshot} | "
            f"edge={self.edge:+.3f} fair={self.fair_price:.3f} "
            f"mkt={self.market_price:.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Strategy Engine
# ═══════════════════════════════════════════════════════════════════════

class CricketStrategy:
    """Scans CricketState for the three signal types.

    Signals:
        A — Momentum Edge:
            rolling_rr > required_rr + threshold
            AND overs_remaining > 6

        B — Wicket Overreaction:
            recent wicket
            AND run_rate >= required_run_rate
            AND wickets <= 2

        C — Latency Snipe (log-only):
            boundary event
            AND price_update_latency > 2s

    All signals are paper-only. No live execution.
    """

    def __init__(
        self,
        momentum_rr_threshold: float = 2.0,
        momentum_edge_threshold: float = 0.08,
        wicket_edge_threshold: float = 0.10,
        latency_threshold_ms: float = 2000.0,
        min_overs_remaining: float = 6.0,
        max_wickets_for_wicket_signal: int = 2,
    ):
        self.momentum_rr_threshold = momentum_rr_threshold
        self.momentum_edge_threshold = momentum_edge_threshold
        self.wicket_edge_threshold = wicket_edge_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.min_overs_remaining = min_overs_remaining
        self.max_wickets_for_wicket_signal = max_wickets_for_wicket_signal

        # State dedup: {match_id: last_state_key}
        self._last_state: dict[str, str] = {}

        # Gemini hypothesis logging
        self._pitch_bias_logged: set[str] = set()
        self._anchor_trap_logged: set[str] = set()

    def _state_key(self, state: CricketState) -> str:
        """Composite key for state dedup."""
        return (
            f"{state.match_id}:{state.innings}:{state.runs}:"
            f"{state.wickets}:{state.overs}:{state.balls}"
        )

    def evaluate(
        self,
        state: CricketState,
        market_price: float,
    ) -> list[CricketSignal]:
        """Evaluate all three signal types against current state.

        Args:
            state: Current match state.
            market_price: Current Polymarket probability for batting team.

        Returns:
            List of signals (may be empty, or multiple if both A+B fire).
        """
        # State dedup — skip if nothing changed
        key = self._state_key(state)
        if self._last_state.get(state.match_id) == key:
            return []
        self._last_state[state.match_id] = key

        # Must be 2nd innings (chase) for meaningful signals
        if state.innings != 2:
            return []

        # Must have enough overs data
        if state.overs < 3.0:
            return []

        # Get model probability
        model = get_win_prob(state)
        fair_price = model.p_batting  # probability batting (chasing) team wins

        signals: list[CricketSignal] = []
        now = time.time()

        # ── Signal A: Momentum Edge ───────────────────────────────
        sig_a = self._check_momentum_edge(state, fair_price, market_price, model, now)
        if sig_a:
            signals.append(sig_a)

        # ── Signal B: Wicket Overreaction ─────────────────────────
        sig_b = self._check_wicket_overreaction(state, fair_price, market_price, model, now)
        if sig_b:
            signals.append(sig_b)

        # ── Signal C: Latency Snipe (log-only) ───────────────────
        sig_c = self._check_latency_snipe(state, fair_price, market_price, model, now)
        if sig_c:
            signals.append(sig_c)

        # ── Gemini Hypotheses (log-only) ─────────────────────────
        self._log_gemini_hypotheses(state, model)

        return signals

    def _check_momentum_edge(
        self,
        state: CricketState,
        fair_price: float,
        market_price: float,
        model: CricketModelOutput,
        now: float,
    ) -> CricketSignal | None:
        """Signal A: Momentum Edge.

        Fires when:
            1. 3-over rolling RR > required RR + threshold
            2. Overs remaining > min_overs_remaining
            3. Model edge > momentum_edge_threshold
        """
        if state.overs_remaining < self.min_overs_remaining:
            return None

        if state.required_run_rate <= 0:
            return None

        rolling_rr = state.rolling_run_rate_3
        rr_delta = rolling_rr - state.required_run_rate

        if rr_delta < self.momentum_rr_threshold:
            return None

        edge = fair_price - market_price
        if abs(edge) < self.momentum_edge_threshold:
            return None

        log.info(
            "MOMENTUM_EDGE | %s | rolling_rr=%.1f req_rr=%.1f delta=%.1f | "
            "fair=%.3f mkt=%.3f edge=%+.3f",
            state, rolling_rr, state.required_run_rate, rr_delta,
            fair_price, market_price, edge,
        )

        return CricketSignal(
            timestamp=now,
            match_id=state.match_id,
            signal_type="MOMENTUM_EDGE",
            edge=edge,
            fair_price=fair_price,
            market_price=market_price,
            state_snapshot=state,
            model_output=model,
            is_tradeable=True,
        )

    def _check_wicket_overreaction(
        self,
        state: CricketState,
        fair_price: float,
        market_price: float,
        model: CricketModelOutput,
        now: float,
    ) -> CricketSignal | None:
        """Signal B: Wicket Overreaction.

        Fires when:
            1. A wicket fell recently (last 6 balls)
            2. Run rate >= required run rate (team still on track)
            3. Wickets lost <= max_wickets_for_wicket_signal
            4. Model edge > wicket_edge_threshold
        """
        if not state.had_recent_wicket:
            return None

        if state.wickets > self.max_wickets_for_wicket_signal:
            return None

        if state.required_run_rate > 0 and state.run_rate < state.required_run_rate:
            return None

        edge = fair_price - market_price
        if abs(edge) < self.wicket_edge_threshold:
            return None

        # Anchor Trap hypothesis check
        anchor_trap = (
            state.wickets <= 2 and
            state.run_rate >= 8.0 and
            state.overs_remaining >= 10.0
        )

        log.info(
            "WICKET_OVERREACTION | %s | wickets=%d rr=%.1f rrr=%.1f | "
            "fair=%.3f mkt=%.3f edge=%+.3f anchor_trap=%s",
            state, state.wickets, state.run_rate, state.required_run_rate,
            fair_price, market_price, edge, anchor_trap,
        )

        return CricketSignal(
            timestamp=now,
            match_id=state.match_id,
            signal_type="WICKET_OVERREACTION",
            edge=edge,
            fair_price=fair_price,
            market_price=market_price,
            state_snapshot=state,
            model_output=model,
            is_tradeable=True,
            anchor_trap_active=anchor_trap,
        )

    def _check_latency_snipe(
        self,
        state: CricketState,
        fair_price: float,
        market_price: float,
        model: CricketModelOutput,
        now: float,
    ) -> CricketSignal | None:
        """Signal C: Latency Snipe (log-only).

        Fires when:
            1. A boundary event occurred recently
            2. Market price update lag > threshold
        """
        if not state.recent_boundaries:
            return None

        latest = state.recent_boundaries[-1]

        # Only fire on fresh boundaries (within 10 seconds)
        if now - latest.timestamp > 10.0:
            return None

        if latest.latency_ms < self.latency_threshold_ms:
            return None

        edge = fair_price - market_price

        log.info(
            "LATENCY_SNIPE | %s | boundary=%d runs | latency=%.0fms | "
            "fair=%.3f mkt=%.3f edge=%+.3f | LOG_ONLY",
            state, latest.runs, latest.latency_ms,
            fair_price, market_price, edge,
        )

        return CricketSignal(
            timestamp=now,
            match_id=state.match_id,
            signal_type="LATENCY_SNIPE",
            edge=edge,
            fair_price=fair_price,
            market_price=market_price,
            state_snapshot=state,
            model_output=model,
            latency_ms=latest.latency_ms,
            is_tradeable=False,  # LOG ONLY — do not trade
        )

    def _log_gemini_hypotheses(
        self,
        state: CricketState,
        model: CricketModelOutput,
    ) -> None:
        """Log Gemini experimental hypotheses (no trade action)."""
        mid = state.match_id

        # Pitch Bias hypothesis
        if model.pitch_bias != 0 and mid not in self._pitch_bias_logged:
            log.info(
                "GEMINI_PITCH_BIAS | %s | venue=%s spinner_eco=%.1f bias=%.3f",
                state, state.venue, state.spinner_economy, model.pitch_bias,
            )
            self._pitch_bias_logged.add(mid)

        # Anchor Trap hypothesis
        if (state.wickets <= 2 and state.run_rate >= 8.0 and
                state.overs_remaining >= 10.0 and
                mid not in self._anchor_trap_logged):
            log.info(
                "GEMINI_ANCHOR_TRAP | %s | wickets=%d rr=%.1f overs_rem=%.1f | "
                "DO NOT fade batting team",
                state, state.wickets, state.run_rate, state.overs_remaining,
            )
            self._anchor_trap_logged.add(mid)
