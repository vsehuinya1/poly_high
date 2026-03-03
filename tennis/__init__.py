"""
Tennis in-play probability engine + Strategy B (Inflection Sniping).

Markov-chain model that computes exact match-win probabilities from
current score state. No ML. No external data. Pure structural model.

Modules:
    state     — TennisState, TennisPointEvent, TennisModelOutput
    model     — TennisMarkovModel (point → game → set → match)
    strategy  — InflectionStrategy (Panic Discount + Set Mean Reversion)
    execution — TennisExecutionGuard (v1.4 regime)
    logger    — TennisCSVLogger (edge validation CSV)
    feeds     — TennisDataFeed (abstract) + FlashscoreFeed (concrete)
"""
from tennis.state import (
    TennisState,
    TennisPointEvent,
    TennisModelOutput,
    PointScore,
    update_from_point,
    compute_break_point_flag,
    compute_momentum_delta,
)
from tennis.model import TennisMarkovModel, get_win_prob
from tennis.strategy import InflectionStrategy, TennisSignal
from tennis.execution import TennisExecutionGuard, ExecutionDecision
from tennis.logger import TennisCSVLogger
from tennis.feeds import TennisDataFeed, FlashscoreFeed

__all__ = [
    "TennisState",
    "TennisPointEvent",
    "TennisModelOutput",
    "PointScore",
    "update_from_point",
    "compute_break_point_flag",
    "compute_momentum_delta",
    "TennisMarkovModel",
    "get_win_prob",
    "InflectionStrategy",
    "TennisSignal",
    "TennisExecutionGuard",
    "ExecutionDecision",
    "TennisCSVLogger",
    "TennisDataFeed",
    "FlashscoreFeed",
]
