"""
Cricket in-play trading module — Paper Only.

DLS-inspired win probability model with three signal types
designed to exploit prediction market microstructure inefficiencies:
    Signal A: Momentum Edge (run rate momentum lag)
    Signal B: Wicket Overreaction (fade panic drops)
    Signal C: Latency Snipe (boundary-to-price lag, log-only)

Modules:
    state     — CricketState, BoundaryEvent, CricketModelOutput
    model     — CricketWinModel (DLS resource table)
    strategy  — CricketStrategy (three signal types)
    execution — CricketExecutionGuard (paper-only guards)
"""
from cricket.state import (
    CricketState,
    CricketModelOutput,
    BoundaryEvent,
    InningsPhase,
)
from cricket.model import CricketWinModel, get_win_prob
from cricket.strategy import CricketStrategy, CricketSignal
from cricket.execution import (
    CricketExecutionGuard,
    CricketExecutionDecision,
    CricketHealthStats,
    CricketMatchState,
)
from cricket.feeds import CricketFeed, CricketCSVLogger

__all__ = [
    "CricketState",
    "CricketModelOutput",
    "BoundaryEvent",
    "InningsPhase",
    "CricketWinModel",
    "get_win_prob",
    "CricketStrategy",
    "CricketSignal",
    "CricketExecutionGuard",
    "CricketExecutionDecision",
    "CricketHealthStats",
    "CricketMatchState",
    "CricketFeed",
    "CricketCSVLogger",
]
