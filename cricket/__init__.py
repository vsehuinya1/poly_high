"""
Cricket in-play trading module.

v1.0.1 — Sportmonks v2 Scoreboard Mode (2026-04-25)

New pipeline:
    sm_feeds    — Sportmonks v2 fixture poller (scoreboards)
    sm_state    — Scoreboard-delta event derivation + regime
    sm_strategy — Momentum/Fade strategy engine
    sm_mapping  — Fixture → Polymarket token hard mapping

Legacy modules (still used):
    execution   — Execution guards (spread, stale, cooldown)
    live_executor — Limit-offset order placement
    exit_manager  — Trade exit management
"""
from cricket.sm_feeds import SmCricketFeed, ScoreboardSnapshot
from cricket.sm_state import SmCricketState, EventResult, CricketEvent, CricketRegime
from cricket.sm_strategy import SmCricketStrategy, TradeDecision
from cricket.sm_mapping import (
    get_mapping, get_all_fixture_ids, add_mapping,
    FixtureMapping, IPL_FIXTURE_MAP,
)

# Legacy — still used for execution
from cricket.execution import (
    CricketExecutionGuard,
    CricketExecutionDecision,
    CricketHealthStats,
    CricketMatchState,
)
from cricket.live_executor import CricketLiveExecutor, CricketOrderResult
from cricket.exit_manager import CricketExitManager

# Legacy — kept for backwards compatibility but NOT used in v1.0.1
from cricket.state import CricketState, CricketModelOutput, BoundaryEvent, InningsPhase
from cricket.model import CricketWinModel, get_win_prob
from cricket.strategy import CricketStrategy, CricketSignal
from cricket.feeds import CricketFeed, CricketCSVLogger
from cricket.health import (
    CricketBookHealthMonitor,
    check_cricket_readiness,
    ReadinessResult,
    ReadinessStatus,
    FailureReason,
    MarketHealth,
    SpreadPhase,
    get_spread_phase,
    get_liquidity_threshold,
    spread_ok,
)

__all__ = [
    # v1.0.1 — Sportmonks pipeline
    "SmCricketFeed", "ScoreboardSnapshot",
    "SmCricketState", "EventResult", "CricketEvent", "CricketRegime",
    "SmCricketStrategy", "TradeDecision",
    "get_mapping", "get_all_fixture_ids", "add_mapping",
    "FixtureMapping", "IPL_FIXTURE_MAP",
    # Execution (shared)
    "CricketExecutionGuard", "CricketExecutionDecision",
    "CricketHealthStats", "CricketMatchState",
    "CricketLiveExecutor", "CricketOrderResult",
    "CricketExitManager",
    # Legacy
    "CricketState", "CricketModelOutput", "BoundaryEvent", "InningsPhase",
    "CricketWinModel", "get_win_prob",
    "CricketStrategy", "CricketSignal",
    "CricketFeed", "CricketCSVLogger",
    "CricketBookHealthMonitor", "check_cricket_readiness",
    "ReadinessResult", "ReadinessStatus", "FailureReason",
    "MarketHealth", "SpreadPhase",
    "get_spread_phase", "get_liquidity_threshold", "spread_ok",
]
