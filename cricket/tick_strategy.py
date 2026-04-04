"""
Cricket Tick-Based Reversion Strategy — Pure tick data only.

Two independent signal types:
  1. DRIFT_REVERSION  — fade slow moves (≥0.05 over 10min)
  2. SPIKE_REVERSION  — fade fast spikes (≥0.03 in ≤60s)

No ESPN, no score data, no match events.
Only uses: price (mid), spread, timestamp.

v1.0 — 2026-04-04
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("cricket.tick_strategy")


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

# Ring buffer size (max ticks to keep per market)
MAX_TICK_HISTORY = 600  # ~10 min at 1 tick/s

# Guards
PRICE_FLOOR = 0.20
PRICE_CEIL = 0.80
MAX_SPREAD = 0.05
STALE_TICK_S = 30.0     # skip if no tick movement in 30s

# Drift Reversion
DRIFT_MIN_MOVE = 0.05   # min net price move
DRIFT_WINDOW_S = 600.0  # 10 minutes lookback
DRIFT_SMOOTH_RATIO = 0.60  # net/range > 0.60 means "smooth drift" (not choppy)

# Spike Reversion
SPIKE_MIN_MOVE = 0.03   # min price jump
SPIKE_WINDOW_S = 60.0   # must occur within 60s
SPIKE_MAX_SPREAD = 0.04 # only trade if spread ≤ this at spike time

# Exits
STOP_LOSS_R = 0.08      # -8% of entry price
DRIFT_TARGET = 0.035    # midpoint of 0.03–0.05 range
SPIKE_TARGET = 0.04
DRIFT_TIMEOUT_S = 900.0  # 15 minutes
SPIKE_TIMEOUT_S = 300.0  # 5 minutes

# Cooldown per market
COOLDOWN_S = 120.0       # 2min between entries on same market


# ═══════════════════════════════════════════════════════════════════════
#  Tick Data Point
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TickPoint:
    timestamp: float
    mid: float
    spread: float


# ═══════════════════════════════════════════════════════════════════════
#  Active Trade
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TickTrade:
    """Tracks an active tick-strategy trade for exit management."""
    match_id: str
    signal_type: str        # DRIFT_REVERSION or SPIKE_REVERSION
    entry_price: float
    entry_timestamp: float
    direction: str          # "LONG" or "SHORT" — direction WE entered
    target: float           # absolute target price
    stop: float             # absolute stop price
    timeout_s: float        # trade-specific timeout


# ═══════════════════════════════════════════════════════════════════════
#  Signal Output
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CricketTickSignal:
    """Output from the tick-based detector."""
    signal_type: str        # DRIFT_REVERSION or SPIKE_REVERSION
    match_id: str
    direction: str          # LONG or SHORT — what WE do (fade the move)
    move: float             # the observed move magnitude
    entry_price: float      # current mid at signal time
    fair_price: float       # our estimated fair value
    edge: float             # fair_price - entry_price (signed)
    spread: float           # current spread


# ═══════════════════════════════════════════════════════════════════════
#  Detector — per-market instance
# ═══════════════════════════════════════════════════════════════════════

class CricketTickDetector:
    """Self-contained tick-based signal detector for cricket markets.

    Maintains a ring buffer of recent ticks per market and detects
    drift and spike reversion patterns. Also manages active trade
    exits independently.

    Usage:
        detector = CricketTickDetector()
        signal = detector.on_tick(match_id, mid, spread, timestamp)
        if signal:
            # emit signal and register trade
            detector.register_trade(signal)
        exits = detector.check_exits(get_price)
    """

    def __init__(self):
        # Per-market tick history: match_id → deque of TickPoint
        self._ticks: dict[str, deque[TickPoint]] = {}
        # Active trades managed by this detector
        self._trades: dict[str, TickTrade] = {}
        # Cooldown tracking
        self._last_signal_ts: dict[str, float] = {}
        # Skip-reason counters for periodic logging
        self._skip_counts: dict[str, int] = {
            "no_move": 0, "spread": 0, "band": 0, "cooldown": 0,
        }
        self._last_skip_log = 0.0

    # ── Tick Processing ─────────────────────────────────────────────

    def on_tick(
        self,
        match_id: str,
        mid: float,
        spread: float,
        timestamp: float,
    ) -> Optional[CricketTickSignal]:
        """Process a new tick. Returns a signal if one is detected.

        Called once per loop iteration per cricket market.
        """
        # Store tick
        if match_id not in self._ticks:
            self._ticks[match_id] = deque(maxlen=MAX_TICK_HISTORY)
        buf = self._ticks[match_id]
        buf.append(TickPoint(timestamp=timestamp, mid=mid, spread=spread))

        # ── Guards ────────────────────────────────────────────────
        # Already have an active trade for this market
        if match_id in self._trades:
            return None

        # Price band
        if mid < PRICE_FLOOR or mid > PRICE_CEIL:
            self._skip("band")
            return None

        # Spread too wide
        if spread > MAX_SPREAD:
            self._skip("spread")
            return None

        # Staleness — last tick must have changed recently
        if len(buf) >= 2:
            last_change_ts = timestamp
            for i in range(len(buf) - 2, -1, -1):
                if abs(buf[i].mid - mid) > 0.001:
                    last_change_ts = buf[i].timestamp
                    break
            if timestamp - last_change_ts > STALE_TICK_S:
                self._skip("no_move")
                return None

        # Cooldown
        last_sig = self._last_signal_ts.get(match_id, 0)
        if timestamp - last_sig < COOLDOWN_S:
            self._skip("cooldown")
            return None

        # Need enough data
        if len(buf) < 10:
            return None

        # ── Check patterns (priority: spike first) ────────────────
        signal = self._check_spike(match_id, buf, mid, spread, timestamp)
        if signal:
            return signal

        signal = self._check_drift(match_id, buf, mid, spread, timestamp)
        if signal:
            return signal

        # Periodic skip logging (every 60s)
        if timestamp - self._last_skip_log > 60.0:
            total_skips = sum(self._skip_counts.values())
            if total_skips > 0:
                log.info(
                    "CRICKET_TICK_SKIPS | no_move=%d | spread=%d | "
                    "band=%d | cooldown=%d",
                    self._skip_counts["no_move"],
                    self._skip_counts["spread"],
                    self._skip_counts["band"],
                    self._skip_counts["cooldown"],
                )
                self._skip_counts = {k: 0 for k in self._skip_counts}
            self._last_skip_log = timestamp

        return None

    # ── Pattern: Spike Reversion ──────────────────────────────────

    def _check_spike(
        self, match_id: str, buf: deque, mid: float,
        spread: float, now: float,
    ) -> Optional[CricketTickSignal]:
        """Detect sharp move ≥0.03 within ≤60s and fade it."""
        if spread > SPIKE_MAX_SPREAD:
            return None

        # Look back through buffer for a price 60s ago
        for i in range(len(buf) - 2, -1, -1):
            dt = now - buf[i].timestamp
            if dt > SPIKE_WINDOW_S:
                break
            move = mid - buf[i].mid
            if abs(move) >= SPIKE_MIN_MOVE:
                # Fade the spike: if price spiked UP, we go SHORT (sell)
                # but in Poly terms: if price spiked up, fair value is lower
                direction = "SHORT" if move > 0 else "LONG"
                # Fair value = price before spike + small offset
                fair = buf[i].mid + (0.01 if move > 0 else -0.01)
                edge = fair - mid if direction == "LONG" else mid - fair

                self._last_signal_ts[match_id] = now
                log.info(
                    "CRICKET_SIGNAL | type=SPIKE_REVERSION | "
                    "move=%.4f | dir=%s | mid=%.4f | spread=%.4f | "
                    "fair=%.4f | edge=%.4f | match=%s",
                    move, direction, mid, spread, fair, edge, match_id,
                )
                return CricketTickSignal(
                    signal_type="SPIKE_REVERSION",
                    match_id=match_id,
                    direction=direction,
                    move=abs(move),
                    entry_price=mid,
                    fair_price=fair,
                    edge=edge,
                    spread=spread,
                )
        return None

    # ── Pattern: Drift Reversion ──────────────────────────────────

    def _check_drift(
        self, match_id: str, buf: deque, mid: float,
        spread: float, now: float,
    ) -> Optional[CricketTickSignal]:
        """Detect smooth drift ≥0.05 over 10min and fade it."""
        # Find the price from ~10 minutes ago
        old_tick = None
        for i in range(len(buf)):
            dt = now - buf[i].timestamp
            if dt >= DRIFT_WINDOW_S:
                old_tick = buf[i]
            else:
                break  # buf is chronological; stop once we're within window

        if old_tick is None:
            return None

        net_move = mid - old_tick.mid
        if abs(net_move) < DRIFT_MIN_MOVE:
            return None

        # Check smoothness: net_move / total_range > threshold
        prices_in_window = [
            t.mid for t in buf if now - t.timestamp <= DRIFT_WINDOW_S
        ]
        if not prices_in_window:
            return None
        total_range = max(prices_in_window) - min(prices_in_window)
        if total_range <= 0:
            return None
        smoothness = abs(net_move) / total_range
        if smoothness < DRIFT_SMOOTH_RATIO:
            return None  # too choppy, not a clean drift

        # Fade the drift
        direction = "SHORT" if net_move > 0 else "LONG"
        # Fair value = midpoint between current and origin
        fair = (mid + old_tick.mid) / 2.0
        edge = fair - mid if direction == "LONG" else mid - fair

        self._last_signal_ts[match_id] = now
        log.info(
            "CRICKET_SIGNAL | type=DRIFT_REVERSION | "
            "move=%.4f | dir=%s | mid=%.4f | spread=%.4f | "
            "fair=%.4f | edge=%.4f | smooth=%.2f | match=%s",
            net_move, direction, mid, spread, fair, edge,
            smoothness, match_id,
        )
        return CricketTickSignal(
            signal_type="DRIFT_REVERSION",
            match_id=match_id,
            direction=direction,
            move=abs(net_move),
            entry_price=mid,
            fair_price=fair,
            edge=edge,
            spread=spread,
        )

    # ── Trade Registration ────────────────────────────────────────

    def register_trade(self, signal: CricketTickSignal) -> None:
        """Register an active trade for exit tracking."""
        if signal.direction == "LONG":
            target = signal.entry_price + (
                DRIFT_TARGET if "DRIFT" in signal.signal_type else SPIKE_TARGET
            )
            stop = signal.entry_price * (1.0 - STOP_LOSS_R)
        else:
            target = signal.entry_price - (
                DRIFT_TARGET if "DRIFT" in signal.signal_type else SPIKE_TARGET
            )
            stop = signal.entry_price * (1.0 + STOP_LOSS_R)

        timeout = (
            DRIFT_TIMEOUT_S if "DRIFT" in signal.signal_type
            else SPIKE_TIMEOUT_S
        )

        self._trades[signal.match_id] = TickTrade(
            match_id=signal.match_id,
            signal_type=signal.signal_type,
            entry_price=signal.entry_price,
            entry_timestamp=time.time(),
            direction=signal.direction,
            target=target,
            stop=stop,
            timeout_s=timeout,
        )
        log.info(
            "CRICKET_TICK_TRADE | %s | %s | entry=%.4f | "
            "target=%.4f | stop=%.4f | timeout=%ds | %s",
            signal.signal_type, signal.direction,
            signal.entry_price, target, stop,
            int(timeout), signal.match_id,
        )

    # ── Exit Checks ───────────────────────────────────────────────

    def check_exits(
        self, get_price: callable,
    ) -> list[tuple[str, str, float, float, str]]:
        """Check all active tick-trades for exits.

        Args:
            get_price: fn(match_id) → (mid, spread) or None

        Returns:
            List of (match_id, signal_type, entry_price, exit_price, reason)
        """
        exits = []
        now = time.time()

        for match_id in list(self._trades.keys()):
            trade = self._trades[match_id]
            result = get_price(match_id)
            if result is None:
                continue
            mid, spread = result
            if mid <= 0:
                continue

            elapsed = now - trade.entry_timestamp
            reason = None

            if trade.direction == "LONG":
                # Stop: price fell below stop
                if mid <= trade.stop:
                    reason = "STOP_LOSS"
                # Target: price rose to target
                elif mid >= trade.target:
                    reason = "TARGET"
            else:  # SHORT
                # Stop: price rose above stop
                if mid >= trade.stop:
                    reason = "STOP_LOSS"
                # Target: price fell to target
                elif mid <= trade.target:
                    reason = "TARGET"

            # Timeout
            if not reason and elapsed >= trade.timeout_s:
                reason = "TIMEOUT"

            if reason:
                pnl = (mid - trade.entry_price) if trade.direction == "LONG" \
                    else (trade.entry_price - mid)
                log.info(
                    "CRICKET_TICK_EXIT | %s | %s | entry=%.4f "
                    "exit=%.4f | pnl=%+.4f | hold=%.0fs | %s",
                    trade.signal_type, reason,
                    trade.entry_price, mid, pnl,
                    elapsed, match_id,
                )
                exits.append((
                    match_id, trade.signal_type,
                    trade.entry_price, mid, reason,
                ))
                del self._trades[match_id]

        return exits

    # ── Internals ─────────────────────────────────────────────────

    def _skip(self, reason: str) -> None:
        self._skip_counts[reason] = self._skip_counts.get(reason, 0) + 1

    @property
    def active_trade_count(self) -> int:
        return len(self._trades)

    @property
    def has_trades(self) -> bool:
        return bool(self._trades)
