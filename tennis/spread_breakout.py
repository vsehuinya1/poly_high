"""
Spread Breakout Detector — tick-level microstructure signal.

Detects spread compression → breakout patterns:
    1. Spread narrows below NARROW_THRESHOLD for NARROW_WINDOW+ ticks
    2. Spread suddenly widens past WIDEN_THRESHOLD
    3. Direction confirmed by 2 consecutive ticks in same direction
    4. Entry within price band [0.20, 0.80]

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
WIDEN_THRESHOLD = 0.03      # spread must widen past this to trigger
CONFIRM_TICKS = 2           # consecutive ticks in same direction
MIN_PRICE_DELTA = 0.01      # minimum |delta| to establish direction
PRICE_BAND = (0.20, 0.80)   # entry price range

# Exit Parameters (local to SPREAD_BREAKOUT)
SB_STOP_LOSS = 0.08         # -8% hard stop
SB_TICK_STOP = 3            # 3 adverse ticks
SB_TIMEOUT_S = 600.0        # 10 min max hold


# ═══════════════════════════════════════════════════════════════════════
#  Per-Market State
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _MarketState:
    """Tracks spread compression state for a single market."""
    narrow_count: int = 0           # consecutive ticks with spread < NARROW
    triggered: bool = False         # breakout detected, awaiting confirmation
    pre_widen_mid: float = 0.0      # mid price just before the widen
    confirm_count: int = 0          # consecutive ticks in same direction
    direction: str = ""             # "UP" or "DOWN"
    trigger_ts: float = 0.0         # when breakout was triggered
    _narrow_start_logged: bool = False


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

    Call `tick()` on every market tick. Returns a signal dict if entry
    conditions are met, else None.

    Call `check_exits()` on every loop iteration to manage open trades.
    """

    def __init__(self):
        # Per-market detection state: token_id → _MarketState
        self._states: dict[str, _MarketState] = {}
        # Open trades: token_id → SpreadBreakoutTrade
        self._trades: dict[str, SpreadBreakoutTrade] = {}
        # Cooldown: token_id → last_entry_ts (prevent duplicate entries)
        self._cooldown: dict[str, float] = {}
        self.COOLDOWN_S = 300.0  # 5 min between entries per market

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
        """Process one tick. Returns signal dict if entry conditions met.

        Returns:
            dict with keys: trigger, direction, entry_price, match_id,
            token_id, pre_widen_mid, match_title
            — or None
        """
        # Skip if we already have an open trade for this token
        if token_id in self._trades and self._trades[token_id].is_open:
            return None

        # Cooldown check
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
                # Store mid continuously during narrow phase
                st.pre_widen_mid = mid
            elif st.narrow_count >= NARROW_WINDOW and spread > WIDEN_THRESHOLD:
                # BREAKOUT: spread was narrow, now it widened
                st.triggered = True
                st.trigger_ts = time.time()
                st.confirm_count = 0
                st.direction = ""
                log.info("SPREAD_BREAKOUT_TRIGGER | %s | spread=%.4f | "
                         "pre_mid=%.4f | cur_mid=%.4f",
                         match_title[:40], spread, st.pre_widen_mid, mid)
            else:
                # Reset — spread didn't stay narrow or didn't widen enough
                if st.narrow_count > 0:
                    st.narrow_count = 0
                    st._narrow_start_logged = False
            return None

        # ── Phase 2: Confirm direction after breakout ──────────────
        # Timeout: 30s to confirm, else reset
        if time.time() - st.trigger_ts > 30.0:
            log.info("SPREAD_BREAKOUT_TIMEOUT | %s | no confirm in 30s",
                     match_title[:40])
            self._reset_state(token_id)
            return None

        # Price band check
        if mid < PRICE_BAND[0] or mid > PRICE_BAND[1]:
            log.info("SPREAD_BREAKOUT_SKIP_BAND | %s | price=%.4f",
                     match_title[:40], mid)
            self._reset_state(token_id)
            return None

        price_delta = mid - st.pre_widen_mid
        if abs(price_delta) < MIN_PRICE_DELTA:
            return None  # no clear direction yet

        tick_dir = "UP" if price_delta > 0 else "DOWN"

        if st.direction == "":
            st.direction = tick_dir
            st.confirm_count = 1
        elif tick_dir == st.direction:
            st.confirm_count += 1
        else:
            # Direction flipped — reset confirmation
            st.direction = tick_dir
            st.confirm_count = 1

        if st.confirm_count < CONFIRM_TICKS:
            log.info("SPREAD_CONFIRM | %s | dir=%s | confirm=%d/%d | "
                     "delta=%.4f",
                     match_title[:40], st.direction, st.confirm_count,
                     CONFIRM_TICKS, price_delta)
            return None

        # ── Confirmed: emit signal ─────────────────────────────────
        log.info("SPREAD_ENTRY | %s | dir=%s | price=%.4f | "
                 "pre_mid=%.4f | delta=%.4f",
                 match_title[:40], st.direction, mid,
                 st.pre_widen_mid, price_delta)

        signal = {
            "trigger": "SPREAD_BREAKOUT",
            "direction": st.direction,
            "entry_price": mid,
            "match_id": match_id,
            "token_id": token_id,
            "pre_widen_mid": st.pre_widen_mid,
            "match_title": match_title,
        }

        # Record entry for cooldown + exit tracking
        self._cooldown[token_id] = time.time()
        self._reset_state(token_id)

        return signal

    def register_trade(
        self,
        token_id: str,
        match_id: str,
        player: str,
        entry_price: float,
        direction: str,
    ):
        """Register an executed SPREAD_BREAKOUT trade for exit tracking."""
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
