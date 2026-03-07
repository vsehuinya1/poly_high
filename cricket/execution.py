"""
Cricket Execution Guard — Paper-Only.

Hard guards that gate signal execution. Returns (can_execute, reason)
for each signal. Paper-only mode: no live trades are ever executed.

Guards (in evaluation order):
    1. PAPER_ONLY — all trades are paper (always enforced)
    2. BLOCK_SPREAD — spread > max threshold
    3. BLOCK_STALE — market data > staleness threshold
    4. BLOCK_POSITION — already holding position in this match
    5. BLOCK_COOLDOWN — too soon since last exit
    6. BLOCK_LOG_ONLY — signal marked as log-only (LATENCY_SNIPE)
    7. PASS — signal cleared for paper execution
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from cricket.state import CricketState
from cricket.strategy import CricketSignal

log = logging.getLogger("cricket.execution")


# ═══════════════════════════════════════════════════════════════════════
#  Execution Decision
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CricketExecutionDecision:
    """Result of execution guard evaluation."""
    can_execute: bool
    reason: str = ""

    def __str__(self) -> str:
        return f"{'PASS' if self.can_execute else self.reason}"


# ═══════════════════════════════════════════════════════════════════════
#  Match Execution State
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CricketMatchState:
    """Per-match execution hygiene tracker."""
    match_id: str
    has_open_position: bool = False
    last_exit_ts: float = 0.0
    trade_count: int = 0
    total_pnl: float = 0.0

    def record_entry(self):
        self.has_open_position = True
        self.trade_count += 1

    def record_exit(self, pnl: float):
        self.has_open_position = False
        self.last_exit_ts = time.time()
        self.total_pnl += pnl

    def cooldown_remaining(self, cooldown_s: float) -> float:
        if self.last_exit_ts <= 0:
            return 0.0
        elapsed = time.time() - self.last_exit_ts
        return max(0.0, cooldown_s - elapsed)


# ═══════════════════════════════════════════════════════════════════════
#  Health Stats
# ═══════════════════════════════════════════════════════════════════════

class CricketHealthStats:
    """Runtime health counters for the cricket execution layer."""

    def __init__(self):
        self.signals_fired = 0
        self.signals_momentum = 0
        self.signals_wicket = 0
        self.signals_latency = 0
        self.blocked_spread = 0
        self.blocked_stale = 0
        self.blocked_position = 0
        self.blocked_cooldown = 0
        self.blocked_log_only = 0
        self.trades_executed = 0
        self.gemini_pitch_bias = 0
        self.gemini_anchor_trap = 0

    def log_summary(self):
        log.info(
            "CRICKET HEALTH SNAPSHOT\n"
            "  Signals fired:          %d\n"
            "    Momentum Edge:        %d\n"
            "    Wicket Overreaction:   %d\n"
            "    Latency Snipe:        %d\n"
            "  Blocked (spread):       %d\n"
            "  Blocked (stale):        %d\n"
            "  Blocked (position):     %d\n"
            "  Blocked (cooldown):     %d\n"
            "  Blocked (log_only):     %d\n"
            "  Paper trades executed:  %d\n"
            "  Gemini pitch bias:      %d\n"
            "  Gemini anchor trap:     %d",
            self.signals_fired,
            self.signals_momentum,
            self.signals_wicket,
            self.signals_latency,
            self.blocked_spread,
            self.blocked_stale,
            self.blocked_position,
            self.blocked_cooldown,
            self.blocked_log_only,
            self.trades_executed,
            self.gemini_pitch_bias,
            self.gemini_anchor_trap,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Execution Guard
# ═══════════════════════════════════════════════════════════════════════

class CricketExecutionGuard:
    """Paper-only execution guard for cricket signals.

    CRITICAL: This guard NEVER allows live execution.
    All trades are paper-simulated only.

    Guards:
        1. BLOCK_LOG_ONLY: signal is log-only (LATENCY_SNIPE)
        2. BLOCK_SPREAD: spread too wide
        3. BLOCK_STALE: data too old
        4. BLOCK_POSITION: already holding position
        5. BLOCK_COOLDOWN: too soon since last exit
    """

    def __init__(
        self,
        max_spread: float = 0.02,
        staleness_s: float = 5.0,
        cooldown_s: float = 120.0,
        trade_size_usd: float = 200.0,
    ):
        self.max_spread = max_spread
        self.staleness_s = staleness_s
        self.cooldown_s = cooldown_s
        self.trade_size_usd = trade_size_usd

        self._match_states: dict[str, CricketMatchState] = {}
        self.stats = CricketHealthStats()

    def _get_match_state(self, match_id: str) -> CricketMatchState:
        if match_id not in self._match_states:
            self._match_states[match_id] = CricketMatchState(match_id=match_id)
        return self._match_states[match_id]

    def can_execute(
        self,
        signal: CricketSignal,
        spread: float = 0.0,
        data_age_s: float = 0.0,
    ) -> CricketExecutionDecision:
        """Run all guards. Returns first failure or PASS.

        Args:
            signal: The cricket signal to evaluate.
            spread: Current market spread.
            data_age_s: Age of the latest market data in seconds.

        Returns:
            CricketExecutionDecision with can_execute and reason.
        """
        self.stats.signals_fired += 1

        # Track signal type
        if signal.signal_type == "MOMENTUM_EDGE":
            self.stats.signals_momentum += 1
        elif signal.signal_type == "WICKET_OVERREACTION":
            self.stats.signals_wicket += 1
        elif signal.signal_type == "LATENCY_SNIPE":
            self.stats.signals_latency += 1

        # 1. BLOCK_LOG_ONLY — signal not tradeable
        if not signal.is_tradeable:
            self.stats.blocked_log_only += 1
            return CricketExecutionDecision(False, "BLOCK_LOG_ONLY")

        # 2. BLOCK_SPREAD
        if spread > self.max_spread:
            self.stats.blocked_spread += 1
            log.info("BLOCK_SPREAD | %s | spread=%.4f > %.4f",
                     signal.match_id, spread, self.max_spread)
            return CricketExecutionDecision(False, "BLOCK_SPREAD")

        # 3. BLOCK_STALE
        if data_age_s > self.staleness_s:
            self.stats.blocked_stale += 1
            log.info("BLOCK_STALE | %s | age=%.1fs > %.1fs",
                     signal.match_id, data_age_s, self.staleness_s)
            return CricketExecutionDecision(False, "BLOCK_STALE")

        # 4. BLOCK_POSITION
        ms = self._get_match_state(signal.match_id)
        if ms.has_open_position:
            self.stats.blocked_position += 1
            return CricketExecutionDecision(False, "BLOCK_POSITION")

        # 5. BLOCK_COOLDOWN
        cd = ms.cooldown_remaining(self.cooldown_s)
        if cd > 0:
            self.stats.blocked_cooldown += 1
            log.info("BLOCK_COOLDOWN | %s | %.0fs remaining",
                     signal.match_id, cd)
            return CricketExecutionDecision(False, "BLOCK_COOLDOWN")

        # ── PASS ─────────────────────────────────────────────────
        self.stats.trades_executed += 1
        return CricketExecutionDecision(True, "PASS")

    def record_entry(self, match_id: str):
        """Mark that a paper position has been opened."""
        ms = self._get_match_state(match_id)
        ms.record_entry()

    def record_exit(self, match_id: str, pnl: float):
        """Mark that a paper position has been closed."""
        ms = self._get_match_state(match_id)
        ms.record_exit(pnl)

    def get_match_state(self, match_id: str) -> CricketMatchState:
        return self._get_match_state(match_id)
