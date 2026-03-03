"""
Tennis Execution Regime v1.4.

Hard guards that gate signal execution. Returns (can_execute, reason)
for each signal. No trading logic — pure guard layer.

Guards:
    1. Price cap: no entries above 0.85.
    2. Tiebreak block: no trading during tiebreaks.
    3. Staleness: score must be < 3s old.
    4. Position limit: max 1 open position per match.
    5. Cooldown: 120s after last exit.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from tennis.state import TennisState
from tennis.strategy import TennisSignal

log = logging.getLogger("tennis.execution")


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
    """

    def __init__(self, match_id: str):
        self.match_id = match_id
        self.has_open_position: bool = False
        self.last_exit_time: float = 0.0
        self.trade_count: int = 0
        self.pnl: float = 0.0

    def record_entry(self) -> None:
        self.has_open_position = True
        self.trade_count += 1

    def record_exit(self, pnl: float) -> None:
        self.has_open_position = False
        self.last_exit_time = time.time()
        self.pnl += pnl

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
                 cooldown_s: float = 120.0):
        self.price_cap = price_cap
        self.staleness_s = staleness_s
        self.cooldown_s = cooldown_s
        self._match_states: dict[str, MatchExecutionState] = {}

    def _get_match_state(self, match_id: str) -> MatchExecutionState:
        if match_id not in self._match_states:
            self._match_states[match_id] = MatchExecutionState(match_id)
        return self._match_states[match_id]

    def can_execute(self, signal: TennisSignal,
                    state: TennisState) -> ExecutionDecision:
        """Run all guards. Returns first failure or PASS.

        Guards are checked in order of cheapest-to-evaluate first.
        """
        now = time.time()

        # 1. Price cap — no entries when market is above 0.85
        if signal.market_price > self.price_cap:
            log.info("BLOCK_PRICE_CAP | match=%s mkt=%.3f > %.2f",
                     signal.match_id, signal.market_price, self.price_cap)
            return ExecutionDecision(False, "BLOCK_PRICE_CAP")

        # 2. Tiebreak — too volatile, skip
        if state.is_tiebreak:
            log.info("BLOCK_TIEBREAK | match=%s", signal.match_id)
            return ExecutionDecision(False, "BLOCK_TIEBREAK")

        # 3. Staleness — score data must be fresh
        age = now - state.timestamp
        if age > self.staleness_s:
            log.info("BLOCK_STALE | match=%s age=%.1fs > %.1fs",
                     signal.match_id, age, self.staleness_s)
            return ExecutionDecision(False, "BLOCK_STALE")

        # 4. Position limit — max 1 per match
        ms = self._get_match_state(signal.match_id)
        if ms.has_open_position:
            log.info("BLOCK_POSITION | match=%s already has open position",
                     signal.match_id)
            return ExecutionDecision(False, "BLOCK_POSITION")

        # 5. Cooldown after exit
        cd = ms.cooldown_remaining(self.cooldown_s)
        if cd > 0:
            log.info("BLOCK_COOLDOWN | match=%s cooldown=%.0fs remaining",
                     signal.match_id, cd)
            return ExecutionDecision(False, "BLOCK_COOLDOWN")

        return ExecutionDecision(True)

    def record_entry(self, match_id: str) -> None:
        """Mark that a position has been opened for this match."""
        self._get_match_state(match_id).record_entry()

    def record_exit(self, match_id: str, pnl: float) -> None:
        """Mark that a position has been closed for this match."""
        self._get_match_state(match_id).record_exit(pnl)

    def get_match_state(self, match_id: str) -> MatchExecutionState:
        return self._get_match_state(match_id)
