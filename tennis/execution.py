"""
Tennis Execution Regime v2.0 — Hardened.

Hard guards that gate signal execution. Returns (can_execute, reason)
for each signal. No trading logic — pure guard layer.

Guards (in evaluation order):
    0. Disabled match: auto-disabled due to consecutive stale events.
    1. Price cap: no entries above 0.85.
    2. Tiebreak block: no trading during tiebreaks.
    3. Staleness: score must be < 3s old (with auto-disable after 5 stales).
    4. Position limit: max 1 open position per match.
    5. Cooldown: 120s after last exit.
    6. Rate limit: max 10 signals per match per rolling hour.

Health metrics tracked via TennisHealthStats.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

from tennis.state import TennisState
from tennis.strategy import TennisSignal

log = logging.getLogger("tennis.execution")


# ═══════════════════════════════════════════════════════════════════════
#  Health Metrics
# ═══════════════════════════════════════════════════════════════════════

class TennisHealthStats:
    """Runtime health counters for the tennis execution layer.

    Tracks all guard activations, trade events, and feed metrics.
    Thread-safe for single-threaded async — no lock needed.
    """

    def __init__(self):
        self.signals_fired: int = 0
        self.dead_market: int = 0
        self.dedup_suppressed: int = 0
        self.position_block: int = 0
        self.stale_block: int = 0
        self.rate_limited: int = 0
        self.price_cap_block: int = 0
        self.tiebreak_block: int = 0
        self.cooldown_block: int = 0
        self.disabled_block: int = 0
        self.trades_executed: int = 0
        self.ws_reconnects: int = 0
        self.max_staleness_ms: float = 0.0
        self._start_time: float = time.time()

    def log_summary(self) -> None:
        """Print daily health snapshot."""
        uptime_h = (time.time() - self._start_time) / 3600
        log.info("=" * 60)
        log.info("TENNIS HEALTH SNAPSHOT")
        log.info("=" * 60)
        log.info("  Uptime:                  %.1f hours", uptime_h)
        log.info("  Signals fired:           %d", self.signals_fired)
        log.info("  Suppressed (dead mkt):   %d", self.dead_market)
        log.info("  Suppressed (dedup):      %d", self.dedup_suppressed)
        log.info("  Blocked (position):      %d", self.position_block)
        log.info("  Blocked (stale):         %d", self.stale_block)
        log.info("  Blocked (rate limit):    %d", self.rate_limited)
        log.info("  Blocked (price cap):     %d", self.price_cap_block)
        log.info("  Blocked (tiebreak):      %d", self.tiebreak_block)
        log.info("  Blocked (cooldown):      %d", self.cooldown_block)
        log.info("  Blocked (disabled):      %d", self.disabled_block)
        log.info("  Trades executed:         %d", self.trades_executed)
        log.info("  WS reconnects:           %d", self.ws_reconnects)
        log.info("  Max staleness (ms):      %.0f", self.max_staleness_ms)
        log.info("=" * 60)

    def as_dict(self) -> dict:
        """Return snapshot as dict for JSON serialization."""
        return {
            "signals_fired": self.signals_fired,
            "dead_market": self.dead_market,
            "dedup_suppressed": self.dedup_suppressed,
            "position_block": self.position_block,
            "stale_block": self.stale_block,
            "rate_limited": self.rate_limited,
            "price_cap_block": self.price_cap_block,
            "tiebreak_block": self.tiebreak_block,
            "cooldown_block": self.cooldown_block,
            "disabled_block": self.disabled_block,
            "trades_executed": self.trades_executed,
            "ws_reconnects": self.ws_reconnects,
            "max_staleness_ms": self.max_staleness_ms,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Execution Result
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExecutionDecision:
    """Result of execution guard evaluation."""
    can_execute: bool
    reason: str = ""        # empty if can_execute is True

    def __str__(self) -> str:
        if self.can_execute:
            return "PASS"
        return f"BLOCKED: {self.reason}"


# ═══════════════════════════════════════════════════════════════════════
#  Match Execution State — per-match tracking
# ═══════════════════════════════════════════════════════════════════════

class MatchExecutionState:
    """Per-match execution hygiene tracker.

    Tracks:
        - Whether a position is currently open.
        - When the last exit occurred (for cooldown).
        - State key at entry (for position loop breaker).
    """

    def __init__(self, match_id: str):
        self.match_id = match_id
        self.has_open_position: bool = False
        self.last_exit_time: float = 0.0
        self.trade_count: int = 0
        self.pnl: float = 0.0
        self.entry_state_key: str = ""      # state key when position was opened
        self.entry_edge_sign: float = 0.0   # sign of edge at entry (+1 or -1)

    def record_entry(self, state_key: str = "", edge: float = 0.0) -> None:
        self.has_open_position = True
        self.trade_count += 1
        self.entry_state_key = state_key
        self.entry_edge_sign = 1.0 if edge >= 0 else -1.0

    def record_exit(self, pnl: float) -> None:
        self.has_open_position = False
        self.last_exit_time = time.time()
        self.pnl += pnl
        self.entry_state_key = ""
        self.entry_edge_sign = 0.0

    def cooldown_remaining(self, cooldown_s: float) -> float:
        if self.last_exit_time == 0:
            return 0.0
        elapsed = time.time() - self.last_exit_time
        return max(0.0, cooldown_s - elapsed)


# ═══════════════════════════════════════════════════════════════════════
#  Execution Guard
# ═══════════════════════════════════════════════════════════════════════

class TennisExecutionGuard:
    """Evaluates whether a signal passes all execution guards.

    v2.0 additions:
        - Position loop breaker: skip eval when position open + state unchanged.
        - Stale auto-disable: 5 consecutive stales → 5min blackout.
        - Rolling window rate limiter: max signals per match per hour.
        - TennisHealthStats integration.

    Usage:
        guard = TennisExecutionGuard()
        decision = guard.can_execute(signal, state)
        if decision.can_execute:
            # proceed with paper trade
        else:
            log.info("Blocked: %s", decision.reason)
    """

    def __init__(self, price_cap: float = 0.85,
                 staleness_s: float = 3.0,
                 cooldown_s: float = 120.0,
                 max_signals_per_hour: int = 10,
                 stale_disable_count: int = 5,
                 stale_disable_s: float = 300.0):
        self.price_cap = price_cap
        self.staleness_s = staleness_s
        self.cooldown_s = cooldown_s
        self.max_signals_per_hour = max_signals_per_hour
        self.stale_disable_count = stale_disable_count
        self.stale_disable_s = stale_disable_s

        self._match_states: dict[str, MatchExecutionState] = {}
        self.stats = TennisHealthStats()

        # ── Stale auto-disable ────────────────────────────────────
        self._stale_counts: dict[str, int] = {}           # consecutive stale events
        self._disabled_until: dict[str, float] = {}       # match_id → re-enable time
        self._stale_disable_logged: set[str] = set()      # log-once per disable

        # ── Rolling rate limiter ──────────────────────────────────
        self._signal_times: dict[str, deque] = {}         # match_id → deque of timestamps
        self._rate_limit_logged: set[str] = set()         # log-once per match per window

        # ── Position loop breaker ─────────────────────────────────
        self._position_logged: set[str] = set()           # log-once per open position

    def _get_match_state(self, match_id: str) -> MatchExecutionState:
        if match_id not in self._match_states:
            self._match_states[match_id] = MatchExecutionState(match_id)
        return self._match_states[match_id]

    # ── Pre-check: should we even evaluate this match? ────────────

    def should_evaluate(self, match_id: str, state_key: str = "",
                        edge: float = 0.0) -> bool:
        """Check if a match should be evaluated at all.

        Called BEFORE strategy.evaluate() to prevent:
            - Position loop spam (open position + same state).
            - Disabled matches (consecutive stale auto-disable).

        Returns True if evaluation should proceed.
        """
        now = time.time()

        # ── Disabled match (stale auto-disable) ───────────────────
        disabled_until = self._disabled_until.get(match_id, 0)
        if now < disabled_until:
            self.stats.disabled_block += 1
            return False

        # ── Position loop breaker ─────────────────────────────────
        ms = self._get_match_state(match_id)
        if ms.has_open_position:
            # Allow re-evaluation ONLY if state key changed AND edge flipped
            if state_key and ms.entry_state_key:
                state_changed = (state_key != ms.entry_state_key)
                edge_flipped = (
                    (edge >= 0 and ms.entry_edge_sign < 0) or
                    (edge < 0 and ms.entry_edge_sign > 0)
                )
                if state_changed and edge_flipped:
                    return True
            # Default: skip evaluation while position open
            if match_id not in self._position_logged:
                log.info("SKIP_EVAL | match=%s position open — suppressing until state change",
                         match_id)
                self._position_logged.add(match_id)
            self.stats.position_block += 1
            return False

        return True

    def can_execute(self, signal: TennisSignal,
                    state: TennisState) -> ExecutionDecision:
        """Run all guards. Returns first failure or PASS.

        Guards are checked in order of cheapest-to-evaluate first.
        """
        now = time.time()

        # 0. Disabled match (redundant safety — also checked in should_evaluate)
        disabled_until = self._disabled_until.get(signal.match_id, 0)
        if now < disabled_until:
            remaining = disabled_until - now
            self.stats.disabled_block += 1
            return ExecutionDecision(False, "BLOCK_DISABLED")

        # 1. Price cap — no entries when market is above 0.85
        if signal.market_price > self.price_cap:
            log.info("BLOCK_PRICE_CAP | match=%s mkt=%.3f > %.2f",
                     signal.match_id, signal.market_price, self.price_cap)
            self.stats.price_cap_block += 1
            return ExecutionDecision(False, "BLOCK_PRICE_CAP")

        # 2. Tiebreak — too volatile, skip
        if state.is_tiebreak:
            log.info("BLOCK_TIEBREAK | match=%s", signal.match_id)
            self.stats.tiebreak_block += 1
            return ExecutionDecision(False, "BLOCK_TIEBREAK")

        # 3. Staleness — score data must be fresh
        age = now - state.timestamp
        if age > self.staleness_s:
            age_ms = age * 1000
            self.stats.stale_block += 1
            self.stats.max_staleness_ms = max(self.stats.max_staleness_ms, age_ms)

            # Track consecutive stale events
            self._stale_counts[signal.match_id] = \
                self._stale_counts.get(signal.match_id, 0) + 1
            count = self._stale_counts[signal.match_id]

            if count >= self.stale_disable_count:
                self._disabled_until[signal.match_id] = now + self.stale_disable_s
                if signal.match_id not in self._stale_disable_logged:
                    log.warning(
                        "STALE_DISABLE | match=%s disabled for %ds "
                        "(consecutive_stales=%d, last_age=%.1fs, delta_ms=%.0f)",
                        signal.match_id, int(self.stale_disable_s),
                        count, age, age_ms)
                    self._stale_disable_logged.add(signal.match_id)

            log.info("BLOCK_STALE | match=%s age=%.1fs delta_ms=%.0f consecutive=%d",
                     signal.match_id, age, age_ms, count)
            return ExecutionDecision(False, "BLOCK_STALE")

        # Staleness passed → reset stale counter
        if signal.match_id in self._stale_counts:
            self._stale_counts[signal.match_id] = 0
            # Clear disable state on recovery
            self._disabled_until.pop(signal.match_id, None)
            self._stale_disable_logged.discard(signal.match_id)

        # 4. Position limit — max 1 per match
        ms = self._get_match_state(signal.match_id)
        if ms.has_open_position:
            self.stats.position_block += 1
            log.info("BLOCK_POSITION | match=%s already has open position",
                     signal.match_id)
            return ExecutionDecision(False, "BLOCK_POSITION")

        # 5. Cooldown after exit
        cd = ms.cooldown_remaining(self.cooldown_s)
        if cd > 0:
            self.stats.cooldown_block += 1
            log.info("BLOCK_COOLDOWN | match=%s cooldown=%.0fs remaining",
                     signal.match_id, cd)
            return ExecutionDecision(False, "BLOCK_COOLDOWN")

        # 6. Rolling rate limiter — max signals per match per hour
        if signal.match_id not in self._signal_times:
            self._signal_times[signal.match_id] = deque()
        times = self._signal_times[signal.match_id]
        # Prune timestamps older than 1 hour
        cutoff = now - 3600.0
        while times and times[0] < cutoff:
            times.popleft()
        if len(times) >= self.max_signals_per_hour:
            self.stats.rate_limited += 1
            if signal.match_id not in self._rate_limit_logged:
                log.warning("BLOCK_RATE_LIMIT | match=%s signals=%d >= %d/hr",
                            signal.match_id, len(times), self.max_signals_per_hour)
                self._rate_limit_logged.add(signal.match_id)
            return ExecutionDecision(False, "BLOCK_RATE_LIMIT")
        times.append(now)

        # All guards passed
        self.stats.signals_fired += 1
        return ExecutionDecision(True)

    def record_entry(self, match_id: str,
                     state_key: str = "", edge: float = 0.0) -> None:
        """Mark that a position has been opened for this match."""
        ms = self._get_match_state(match_id)
        ms.record_entry(state_key=state_key, edge=edge)
        self.stats.trades_executed += 1
        # Clear position-logged flag so we log again if a new position opens later
        self._position_logged.discard(match_id)

    def record_exit(self, match_id: str, pnl: float) -> None:
        """Mark that a position has been closed for this match."""
        self._get_match_state(match_id).record_exit(pnl)
        # Clear position-logged flag
        self._position_logged.discard(match_id)

    def get_match_state(self, match_id: str) -> MatchExecutionState:
        return self._get_match_state(match_id)
