"""
Cricket Scoreboard Strategy Engine.

Consumes EventResult from sm_state and produces trade decisions:
    CHAOS → MOMENTUM (WICKET→SHORT, BOUNDARY→LONG, SURGE→LONG)
    STRUCTURED → FADE (WICKET→LONG, BOUNDARY→SHORT, delayed)

Safety:
    - Position lock (no overlapping trades per fixture)
    - 30s cooldown per fixture
    - Minimum signal quality filter
    - Ghost liquidity filter (spread > 0.04 → skip)
    - Low-band filter (price < 0.20, spread/price > 0.25 → skip)

v1.0.1 — 2026-04-25
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

from cricket.sm_state import EventResult, CricketEvent, CricketRegime

log = logging.getLogger("cricket.sm_strategy")


# ═════════════════════════════════════════════════════════════════════
#  Config
# ═════════════════════════════════════════════════════════════════════

COOLDOWN_S = 30.0           # minimum seconds between trades per fixture
MIN_RUNS_DELTA = 2          # minimum runs_delta for non-WICKET signals
GHOST_SPREAD_MAX = 0.04     # skip if spread > this
GHOST_BOOK_AGE_S = 60.0     # skip if book age > this
LOW_BAND_PRICE = 0.20       # low-band threshold
LOW_BAND_MTS_RATIO = 0.25   # skip low-band if spread/price > this

# STRUCTURED fade delay — wait N polls before entry
FADE_DELAY_POLLS = 2        # wait 2 polls (~10s at 5s interval)


# ═════════════════════════════════════════════════════════════════════
#  Trade Decision
# ═════════════════════════════════════════════════════════════════════

@dataclass
class TradeDecision:
    """Output of strategy evaluation."""
    should_trade: bool
    direction: str = ""        # "LONG" or "SHORT"
    reason: str = ""           # signal description
    skip_reason: str = ""      # if should_trade=False
    fixture_id: int = 0
    event: str = ""
    regime: str = ""
    runs_delta: int = 0
    wickets: int = 0
    overs: float = 0.0
    pressure: float = 0.0
    is_fade: bool = False      # True = STRUCTURED fade (delayed)


# ═════════════════════════════════════════════════════════════════════
#  Strategy Engine
# ═════════════════════════════════════════════════════════════════════

class SmCricketStrategy:
    """Scoreboard-based cricket strategy.

    Maintains per-fixture state for:
        - Position locks
        - Trade cooldowns
        - Fade delay tracking
    """

    def __init__(self):
        self._positions: dict[int, bool] = {}       # fixture_id → has position
        self._last_trade_ts: dict[int, float] = {}   # fixture_id → last trade ts
        self._fade_queue: dict[int, dict] = {}       # fixture_id → pending fade info
        self._fade_polls: dict[int, int] = {}        # fixture_id → polls since fade queued

    def evaluate(
        self,
        event_result: EventResult,
        market_price: float,
        spread: float,
        book_age_s: float = 0.0,
    ) -> TradeDecision:
        """Evaluate an event and produce a trade decision.

        Args:
            event_result: From SmCricketState.update()
            market_price: Current Polymarket mid price
            spread: Current bid-ask spread
            book_age_s: Age of latest book data in seconds

        Returns:
            TradeDecision
        """
        fid = event_result.fixture_id
        now = event_result.timestamp

        # ── 1. Position lock ──────────────────────────────────────
        if self._positions.get(fid, False):
            return self._skip(fid, event_result, "POSITION_LOCK")

        # ── 2. Cooldown ───────────────────────────────────────────
        last_trade = self._last_trade_ts.get(fid, 0)
        if now - last_trade < COOLDOWN_S:
            remaining = COOLDOWN_S - (now - last_trade)
            return self._skip(
                fid, event_result,
                f"COOLDOWN ({remaining:.0f}s remaining)",
            )

        # ── 3. Ghost liquidity filter ─────────────────────────────
        if spread > GHOST_SPREAD_MAX:
            return self._skip(
                fid, event_result,
                f"GHOST_SPREAD ({spread:.4f} > {GHOST_SPREAD_MAX})",
            )
        if book_age_s > GHOST_BOOK_AGE_S:
            return self._skip(
                fid, event_result,
                f"STALE_BOOK ({book_age_s:.0f}s > {GHOST_BOOK_AGE_S:.0f}s)",
            )

        # ── 4. Minimum signal quality ─────────────────────────────
        if (event_result.event != CricketEvent.WICKET and
                event_result.runs_delta < MIN_RUNS_DELTA):
            return self._skip(
                fid, event_result,
                f"LOW_SIGNAL (runs_delta={event_result.runs_delta} < {MIN_RUNS_DELTA})",
            )

        # ── 5. Low-band filter ────────────────────────────────────
        if market_price < LOW_BAND_PRICE:
            if spread > 0 and (spread / market_price) > LOW_BAND_MTS_RATIO:
                return self._skip(
                    fid, event_result,
                    f"LOW_BAND_MTS (spread/price={spread/market_price:.2f} > {LOW_BAND_MTS_RATIO})",
                )
            # Low-band: FADE ONLY (no momentum)
            if event_result.regime == CricketRegime.CHAOS:
                return self._skip(
                    fid, event_result,
                    "LOW_BAND_NO_MOMENTUM (price < 0.20, CHAOS blocked)",
                )

        # ── 6. Tail collapse override ─────────────────────────────
        if event_result.tail_collapse:
            direction = event_result.forced_direction  # "SHORT"
            log.info(
                "CRICKET_DECISION | fixture=%d | TAIL_COLLAPSE → %s | "
                "wickets=%d | pressure=%.2f",
                fid, direction, event_result.wickets, event_result.pressure,
            )
            return TradeDecision(
                should_trade=True,
                direction=direction,
                reason=f"TAIL_COLLAPSE (wickets={event_result.wickets})",
                fixture_id=fid,
                event=event_result.event.value,
                regime=event_result.regime.value,
                runs_delta=event_result.runs_delta,
                wickets=event_result.wickets,
                overs=event_result.overs,
                pressure=event_result.pressure,
            )

        # ── 7. Strategy by regime ─────────────────────────────────
        if event_result.regime == CricketRegime.CHAOS:
            return self._chaos_momentum(event_result)
        else:
            return self._structured_fade(event_result, market_price, spread)

    def _chaos_momentum(self, er: EventResult) -> TradeDecision:
        """CHAOS regime → immediate momentum trades."""
        fid = er.fixture_id
        event = er.event

        direction_map = {
            CricketEvent.WICKET: "SHORT",
            CricketEvent.BOUNDARY: "LONG",
            CricketEvent.SURGE: "LONG",
        }

        direction = direction_map.get(event)
        if not direction:
            return self._skip(fid, er, f"CHAOS_NO_DIRECTION (event={event.value})")

        reason = f"CHAOS_MOMENTUM ({event.value} → {direction})"

        log.info(
            "CRICKET_DECISION | fixture=%d | %s | event=%s | "
            "overs=%.1f | wickets=%d | pressure=%.2f",
            fid, reason, event.value,
            er.overs, er.wickets, er.pressure,
        )

        return TradeDecision(
            should_trade=True,
            direction=direction,
            reason=reason,
            fixture_id=fid,
            event=event.value,
            regime="CHAOS",
            runs_delta=er.runs_delta,
            wickets=er.wickets,
            overs=er.overs,
            pressure=er.pressure,
        )

    def _structured_fade(
        self, er: EventResult, market_price: float, spread: float,
    ) -> TradeDecision:
        """STRUCTURED regime → fade trades (delayed 1-2 polls).

        For now, execute immediately but flag as fade for logging.
        The delay mechanism can be implemented via the fade queue
        in future iterations.
        """
        fid = er.fixture_id
        event = er.event

        direction_map = {
            CricketEvent.WICKET: "LONG",     # fade panic
            CricketEvent.BOUNDARY: "SHORT",  # fade hype
            CricketEvent.SURGE: "SHORT",     # fade hype
        }

        direction = direction_map.get(event)
        if not direction:
            return self._skip(fid, er, f"STRUCTURED_NO_DIRECTION (event={event.value})")

        reason = f"STRUCTURED_FADE ({event.value} → {direction})"

        log.info(
            "CRICKET_DECISION | fixture=%d | %s | event=%s | "
            "overs=%.1f | wickets=%d | pressure=%.2f",
            fid, reason, event.value,
            er.overs, er.wickets, er.pressure,
        )

        return TradeDecision(
            should_trade=True,
            direction=direction,
            reason=reason,
            fixture_id=fid,
            event=event.value,
            regime="STRUCTURED",
            runs_delta=er.runs_delta,
            wickets=er.wickets,
            overs=er.overs,
            pressure=er.pressure,
            is_fade=True,
        )

    def _skip(
        self, fixture_id: int, er: EventResult, reason: str,
    ) -> TradeDecision:
        """Create a skip decision with logging."""
        log.info(
            "CRICKET_SKIP_REASON | fixture=%d | reason=%s | "
            "event=%s | regime=%s",
            fixture_id, reason,
            er.event.value, er.regime.value,
        )
        return TradeDecision(
            should_trade=False,
            skip_reason=reason,
            fixture_id=fixture_id,
            event=er.event.value,
            regime=er.regime.value,
            runs_delta=er.runs_delta,
            wickets=er.wickets,
            overs=er.overs,
            pressure=er.pressure,
        )

    # ── Position management ───────────────────────────────────────

    def record_entry(self, fixture_id: int) -> None:
        """Mark fixture as having an open position."""
        self._positions[fixture_id] = True
        self._last_trade_ts[fixture_id] = time.time()
        log.info("CRICKET_POSITION_OPEN | fixture=%d", fixture_id)

    def record_exit(self, fixture_id: int) -> None:
        """Mark fixture position as closed."""
        self._positions[fixture_id] = False
        log.info("CRICKET_POSITION_CLOSED | fixture=%d", fixture_id)

    def has_position(self, fixture_id: int) -> bool:
        """Check if fixture has an open position."""
        return self._positions.get(fixture_id, False)
