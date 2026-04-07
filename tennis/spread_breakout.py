"""
Spread Breakout Detector — tick-level microstructure signal.

Detects spread compression → breakout patterns:
    1. Spread narrows below NARROW_THRESHOLD for NARROW_WINDOW+ ticks
    2. Spread suddenly widens past WIDEN_THRESHOLD
    3. First impulse: price moves ≥ IMPULSE_THRESHOLD from pre_widen_mid
    4. Stability: at least 1 tick without reversal
    5. Continuation: 2 more ticks in same direction
    6. Entry within price band [0.20, 0.80]

Filters:
    - Exhaustion: skip if initial move > EXHAUSTION_THRESHOLD (already done)
    - Snap-back: cancel if price reverses within 2 ticks of impulse

Independent from SET_MEAN_REVERSION. Runs in parallel.

Exit logic is self-contained:
    - STOP_LOSS: -8% from entry
    - TICK_STOP: 3 adverse ticks
    - TIMEOUT: 600s (10 min)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("tennis.spread_breakout")


# ═══════════════════════════════════════════════════════════════════════
#  Detection Parameters
# ═══════════════════════════════════════════════════════════════════════

NARROW_THRESHOLD = 0.02     # spread must be below this
NARROW_WINDOW = 5           # for this many consecutive ticks
WIDEN_THRESHOLD = 0.015     # spread must widen past this to trigger (was 0.03)
PRICE_BAND = (0.20, 0.80)   # entry price range

# Phase 2: Confirmation parameters
CONFIRM_WINDOW_S = 10.0     # max 10s to complete confirmation (was 5s)
IMPULSE_THRESHOLD = 0.015   # minimum |delta| for first impulse
EXHAUSTION_THRESHOLD = 0.04 # skip if instant move > this (already exhausted)
CONTINUATION_TICKS = 2      # ticks needed in same direction after stability
FALLBACK_NARROW_S = 15.0    # fallback signal after narrow persists this long

# Exit Parameters (local to SPREAD_BREAKOUT)
SB_STOP_LOSS = 0.08         # -8% hard stop
SB_TICK_STOP = 3            # 3 adverse ticks
SB_TIMEOUT_S = 600.0        # 10 min max hold


# ═══════════════════════════════════════════════════════════════════════
#  Per-Market State
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _MarketState:
    """Tracks spread compression + breakout confirmation state."""
    # Phase 1: compression
    narrow_count: int = 0
    pre_widen_mid: float = 0.0
    _narrow_start_logged: bool = False
    _narrow_start_ts: float = 0.0     # timestamp when narrow started

    # Phase 2: breakout confirmation
    triggered: bool = False
    trigger_ts: float = 0.0
    direction: str = ""             # "UP" or "DOWN"

    # Sub-phases within Phase 2
    # 0 = awaiting first impulse
    # 1 = impulse seen, awaiting stability tick
    # 2 = stable, awaiting continuation ticks
    phase: int = 0
    impulse_price: float = 0.0      # price at first impulse
    ticks_since_impulse: int = 0    # ticks since impulse was detected
    continuation_count: int = 0     # consecutive continuation ticks
    last_mid: float = 0.0           # previous tick mid (for reversal check)


# ═══════════════════════════════════════════════════════════════════════
#  Open Trade (for exit tracking)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SpreadBreakoutTrade:
    """Tracks an open SPREAD_BREAKOUT trade for exit management."""
    match_id: str
    token_id: str
    player: str
    entry_price: float
    direction: str              # "UP" = bought expecting rise
    entry_ts: float
    peak_price: float = 0.0    # best price seen
    adverse_ticks: int = 0     # consecutive ticks against
    last_price: float = 0.0
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_ts: Optional[float] = None
    is_open: bool = True

    @property
    def r_multiple(self) -> float:
        p = self.exit_price if self.exit_price else self.last_price
        if self.direction == "UP":
            return (p - self.entry_price) / self.entry_price if self.entry_price else 0
        else:
            return (self.entry_price - p) / self.entry_price if self.entry_price else 0

    @property
    def duration_s(self) -> float:
        end = self.exit_ts or time.time()
        return end - self.entry_ts


# ═══════════════════════════════════════════════════════════════════════
#  Detector
# ═══════════════════════════════════════════════════════════════════════

class SpreadBreakoutDetector:
    """Tick-level spread compression → breakout detector.

    Entry flow (after compression detected):
        1. Wait for first impulse (≥0.015 from pre_widen_mid)
        2. Skip if exhaustion (>0.04 instant move)
        3. Require 1 stability tick (no reversal)
        4. Require 2 continuation ticks in same direction
        5. Enter on confirmed second-phase continuation

    Call `tick()` on every market tick. Returns a signal dict if entry
    conditions are met, else None.

    Call `check_exits()` on every loop iteration to manage open trades.
    """

    def __init__(self):
        self._states: dict[str, _MarketState] = {}
        self._trades: dict[str, SpreadBreakoutTrade] = {}
        self._cooldown: dict[str, float] = {}
        self.COOLDOWN_S = 300.0

    @property
    def open_trades(self) -> dict[str, SpreadBreakoutTrade]:
        return {k: v for k, v in self._trades.items() if v.is_open}

    def tick(
        self,
        token_id: str,
        match_id: str,
        mid: float,
        spread: float,
        match_title: str = "",
    ) -> Optional[dict]:
        """Process one tick. Returns signal dict if entry conditions met."""
        if token_id in self._trades and self._trades[token_id].is_open:
            return None

        last_entry = self._cooldown.get(token_id, 0)
        if time.time() - last_entry < self.COOLDOWN_S:
            return None

        st = self._states.setdefault(token_id, _MarketState())

        # ── Phase 1: Track spread compression ──────────────────────
        if not st.triggered:
            if spread < NARROW_THRESHOLD and spread > 0:
                st.narrow_count += 1
                if st.narrow_count == NARROW_WINDOW and not st._narrow_start_logged:
                    log.info("SPREAD_NARROW_START | %s | spread=%.4f | ticks=%d",
                             match_title[:40], spread, st.narrow_count)
                    st._narrow_start_logged = True
                    st._narrow_start_ts = time.time()
                st.pre_widen_mid = mid

                # ── Fallback: emit weak signal if narrow persists ──
                if (st._narrow_start_ts > 0 and
                        time.time() - st._narrow_start_ts > FALLBACK_NARROW_S and
                        PRICE_BAND[0] <= mid <= PRICE_BAND[1]):
                    log.info("SPREAD_FALLBACK_TIMEOUT | %s | "
                             "mid=%.4f | narrow_dur=%.0fs",
                             match_title[:40], mid,
                             time.time() - st._narrow_start_ts)
                    signal = {
                        "trigger": "SPREAD_BREAKOUT",
                        "direction": "UP" if mid < 0.50 else "DOWN",
                        "entry_price": mid,
                        "match_id": match_id,
                        "token_id": token_id,
                        "pre_widen_mid": st.pre_widen_mid,
                        "match_title": match_title,
                        "strength": "WEAK",
                        "reason": "fallback_timeout",
                    }
                    self._cooldown[token_id] = time.time()
                    self._reset_state(token_id)
                    return signal

            elif st.narrow_count >= NARROW_WINDOW and spread > WIDEN_THRESHOLD * 0.7:
                # BREAKOUT detected (relaxed: 70% of threshold)
                st.triggered = True
                st.trigger_ts = time.time()
                st.phase = 0
                st.direction = ""
                st.last_mid = mid
                log.info("SPREAD_PRE_WIDEN_CAPTURE | %s | spread=%.4f | "
                         "pre_mid=%.4f | cur_mid=%.4f",
                         match_title[:40], spread, st.pre_widen_mid, mid)
            else:
                if st.narrow_count > 0:
                    st.narrow_count = 0
                    st._narrow_start_logged = False
                    st._narrow_start_ts = 0.0
            return None

        # ── Phase 2: Three-stage confirmation ──────────────────────

        # Timeout: 10s to complete all stages
        if time.time() - st.trigger_ts > CONFIRM_WINDOW_S:
            log.info("SPREAD_CONFIRM_FAIL | %s | spread_before=%.4f "
                     "spread_after=%.4f | delta=%.4f | elapsed=%.1fs | "
                     "phase=%d",
                     match_title[:40], st.pre_widen_mid, mid,
                     mid - st.pre_widen_mid,
                     time.time() - st.trigger_ts, st.phase)
            self._reset_state(token_id)
            return None

        # Price band check
        if mid < PRICE_BAND[0] or mid > PRICE_BAND[1]:
            log.info("SPREAD_BREAKOUT_SKIP_BAND | %s | price=%.4f",
                     match_title[:40], mid)
            self._reset_state(token_id)
            return None

        price_delta = mid - st.pre_widen_mid

        # ── Stage 0: Awaiting first impulse ────────────────────────
        if st.phase == 0:
            if abs(price_delta) >= EXHAUSTION_THRESHOLD:
                log.info("SPREAD_SKIP_EXHAUSTION | %s | delta=%.4f (>%.2f)",
                         match_title[:40], price_delta, EXHAUSTION_THRESHOLD)
                self._reset_state(token_id)
                return None

            if abs(price_delta) >= IMPULSE_THRESHOLD:
                st.direction = "UP" if price_delta > 0 else "DOWN"
                st.impulse_price = mid
                st.ticks_since_impulse = 0
                st.phase = 1
                log.info("SPREAD_FIRST_IMPULSE | %s | dir=%s | delta=%.4f | "
                         "impulse_price=%.4f",
                         match_title[:40], st.direction, price_delta, mid)

            st.last_mid = mid
            return None

        # ── Stage 1: Stability check (1 tick, no reversal) ─────────
        if st.phase == 1:
            st.ticks_since_impulse += 1

            # Check for snap-back reversal
            if st.direction == "UP" and mid < st.impulse_price - 0.005:
                log.info("SPREAD_CANCEL_REVERSAL | %s | snap_back dir=%s "
                         "impulse=%.4f now=%.4f",
                         match_title[:40], st.direction, st.impulse_price, mid)
                self._reset_state(token_id)
                return None
            if st.direction == "DOWN" and mid > st.impulse_price + 0.005:
                log.info("SPREAD_CANCEL_REVERSAL | %s | snap_back dir=%s "
                         "impulse=%.4f now=%.4f",
                         match_title[:40], st.direction, st.impulse_price, mid)
                self._reset_state(token_id)
                return None

            # Stability confirmed — move to continuation phase
            st.phase = 2
            st.continuation_count = 0
            st.last_mid = mid
            return None

        # ── Stage 2: Continuation (2 ticks in same direction) ──────
        if st.phase == 2:
            # Check direction relative to impulse price
            if st.direction == "UP":
                continuing = mid >= st.last_mid - 0.001
                reversed_ = mid < st.impulse_price - 0.005
            else:
                continuing = mid <= st.last_mid + 0.001
                reversed_ = mid > st.impulse_price + 0.005

            if reversed_:
                log.info("SPREAD_CANCEL_REVERSAL | %s | continuation_fail "
                         "dir=%s impulse=%.4f now=%.4f",
                         match_title[:40], st.direction, st.impulse_price, mid)
                self._reset_state(token_id)
                return None

            if continuing:
                st.continuation_count += 1
            else:
                st.continuation_count = 0

            st.last_mid = mid

            if st.continuation_count < CONTINUATION_TICKS:
                log.debug("SPREAD_CONTINUATION | %s | dir=%s | cont=%d/%d",
                          match_title[:40], st.direction,
                          st.continuation_count, CONTINUATION_TICKS)
                return None

            # ── All stages passed: emit entry signal ───────────────
            log.info("SPREAD_CONTINUATION_CONFIRMED | %s | dir=%s | "
                     "price=%.4f | pre_mid=%.4f | impulse=%.4f",
                     match_title[:40], st.direction, mid,
                     st.pre_widen_mid, st.impulse_price)

            signal = {
                "trigger": "SPREAD_BREAKOUT",
                "direction": st.direction,
                "entry_price": mid,
                "match_id": match_id,
                "token_id": token_id,
                "pre_widen_mid": st.pre_widen_mid,
                "match_title": match_title,
            }

            self._cooldown[token_id] = time.time()
            self._reset_state(token_id)
            return signal

        return None

    def register_trade(
        self,
        token_id: str,
        match_id: str,
        player: str,
        entry_price: float,
        direction: str,
    ):
        """Register an executed SPREAD_BREAKOUT trade for exit tracking."""
        # ═══ GLOBAL EDGE GUARD (v6.1 — inside function, non-bypassable) ═══
        from sports.guards import validate_trade_execution
        can_exec, block_reason = validate_trade_execution(
            edge=0.01,  # Spread breakout has no explicit edge — use minimum
            price=entry_price, sport="tennis_sb",
            context=f"SPREAD_BREAKOUT {direction} | {match_id}",
        )
        if not can_exec:
            return

        self._trades[token_id] = SpreadBreakoutTrade(
            match_id=match_id,
            token_id=token_id,
            player=player,
            entry_price=entry_price,
            direction=direction,
            entry_ts=time.time(),
            peak_price=entry_price,
            last_price=entry_price,
        )

    def check_exits(
        self,
        get_market_price,
    ) -> list[SpreadBreakoutTrade]:
        """Check all open trades for exit conditions.

        Args:
            get_market_price: callable(token_id) → float or None

        Returns:
            List of trades that were closed this tick.
        """
        closed = []
        for token_id, trade in list(self._trades.items()):
            if not trade.is_open:
                continue

            price = get_market_price(token_id)
            if price is None:
                continue

            trade.last_price = price

            # Update peak
            if trade.direction == "UP":
                if price > trade.peak_price:
                    trade.peak_price = price
            else:
                if price < trade.peak_price:
                    trade.peak_price = price

            # ── Exit checks (ordered by priority) ──────────────────
            reason = None
            now = time.time()

            # 1. Stop-loss
            r = trade.r_multiple
            if r <= -SB_STOP_LOSS:
                reason = "EXIT_STOP_LOSS"

            # 2. Tick-stop (3 adverse ticks)
            if reason is None:
                if trade.direction == "UP":
                    adverse = price < trade.last_price - 0.001
                else:
                    adverse = price > trade.last_price + 0.001
                # Use a simpler check: price vs entry
                if trade.direction == "UP":
                    adverse = price < (trade.peak_price - 0.01)
                else:
                    adverse = price > (trade.peak_price + 0.01)

                if adverse:
                    trade.adverse_ticks += 1
                else:
                    trade.adverse_ticks = 0

                if trade.adverse_ticks >= SB_TICK_STOP:
                    reason = "EXIT_TICK_STOP"

            # 3. Timeout
            if reason is None:
                if trade.duration_s >= SB_TIMEOUT_S:
                    reason = "EXIT_TIMEOUT"

            if reason:
                trade.exit_price = price
                trade.exit_reason = reason
                trade.exit_ts = now
                trade.is_open = False

                r_final = trade.r_multiple
                log.info(
                    "SPREAD_EXIT | %s | reason=%s | entry=%.4f → exit=%.4f | "
                    "R=%+.4f | dur=%.0fs | dir=%s",
                    trade.match_id, reason,
                    trade.entry_price, price,
                    r_final, trade.duration_s, trade.direction,
                )
                closed.append(trade)

        return closed

    def _reset_state(self, token_id: str):
        """Reset detection state for a token."""
        self._states[token_id] = _MarketState()
