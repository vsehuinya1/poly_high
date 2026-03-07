"""
Cricket in-play state dataclass.

Immutable snapshot of a cricket match at a single point in time.
Designed as a pure data container — no logic, no side effects.
Follows the same pattern as tennis/state.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
#  Innings Phase Enum
# ═══════════════════════════════════════════════════════════════════════

class InningsPhase(str, Enum):
    """T20 innings phase based on over number."""
    POWERPLAY = "powerplay"   # Overs 1-6
    MIDDLE    = "middle"      # Overs 7-15
    DEATH     = "death"       # Overs 16-20
    COMPLETED = "completed"   # Innings over

    @classmethod
    def from_overs(cls, overs: float) -> "InningsPhase":
        if overs >= 20.0:
            return cls.COMPLETED
        elif overs >= 15.0:
            return cls.DEATH
        elif overs >= 6.0:
            return cls.MIDDLE
        else:
            return cls.POWERPLAY


# ═══════════════════════════════════════════════════════════════════════
#  Boundary Event — for latency tracking
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BoundaryEvent:
    """A boundary (4 or 6) observed from the data feed."""
    timestamp: float         # when the event was detected
    runs: int                # 4 or 6
    over: float              # e.g. 12.3
    batting_team: str
    price_before: float = 0.0   # market price before event
    price_after: float = 0.0    # market price after event
    price_update_ts: float = 0.0  # when market price updated
    latency_ms: float = 0.0    # event_ts → price_update_ts

    @property
    def is_snipeable(self) -> bool:
        """True if latency > 2 seconds — potential snipe window."""
        return self.latency_ms > 2000.0


# ═══════════════════════════════════════════════════════════════════════
#  Cricket Match State
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CricketState:
    """Immutable snapshot of a cricket match at a single point in time.

    All score fields represent the *current* state after the most
    recent ball has been delivered.

    Designed for T20 format but extensible to ODI/Test.
    """
    # ── Identity ──────────────────────────────────────────────────
    match_id: str
    format: str = "t20"        # "t20", "odi", "test"

    # ── Teams ─────────────────────────────────────────────────────
    team_a: str = ""           # Team A name
    team_b: str = ""           # Team B name
    batting_team: str = ""     # Currently batting
    bowling_team: str = ""     # Currently bowling
    toss_winner: str = ""
    toss_decision: str = ""    # "bat" or "field"

    # ── Innings State ────────────────────────────────────────────
    innings: int = 1           # 1 or 2
    runs: int = 0
    wickets: int = 0
    overs: float = 0.0        # e.g. 12.3 = 12 overs, 3 balls
    balls: int = 0             # total balls bowled this innings
    extras: int = 0

    # ── Target & Required (2nd innings only) ─────────────────────
    target_score: int = 0      # runs needed to win (2nd innings)
    first_innings_total: int = 0

    # ── Derived Rates ────────────────────────────────────────────
    run_rate: float = 0.0
    required_run_rate: float = 0.0

    # ── Recent History ───────────────────────────────────────────
    recent_over_runs: tuple[int, ...] = ()     # runs per over (last 6)
    recent_wickets: tuple[float, ...] = ()     # over numbers of recent wickets
    recent_boundaries: tuple[BoundaryEvent, ...] = ()  # last 5 boundaries

    # ── Pitch / Conditions ───────────────────────────────────────
    venue: str = ""
    pitch_type: str = ""       # "flat", "spin", "pace", "unknown"
    spinner_economy: float = 0.0  # for Gemini hypothesis

    # ── Pregame ──────────────────────────────────────────────────
    pregame_favorite: str = ""
    pregame_prob_a: float = 0.5
    pregame_prob_b: float = 0.5

    # ── Timestamps ───────────────────────────────────────────────
    timestamp: float = 0.0
    last_ball_ts: float = 0.0
    last_wicket_ts: float = 0.0
    last_boundary_ts: float = 0.0

    # ── Computed Properties ──────────────────────────────────────

    @property
    def overs_remaining(self) -> float:
        """Overs remaining in this innings (T20 = 20 max)."""
        total = 20.0 if self.format == "t20" else (50.0 if self.format == "odi" else 450.0)
        return max(0.0, total - self.overs)

    @property
    def balls_remaining(self) -> int:
        """Balls remaining in this innings."""
        total = 120 if self.format == "t20" else 300  # 20*6 or 50*6
        return max(0, total - self.balls)

    @property
    def phase(self) -> InningsPhase:
        return InningsPhase.from_overs(self.overs)

    @property
    def runs_remaining(self) -> int:
        """Runs still needed to win (2nd innings)."""
        if self.innings == 1:
            return 0
        return max(0, self.target_score - self.runs)

    @property
    def rolling_run_rate_3(self) -> float:
        """3-over rolling run rate from recent_over_runs."""
        recent = self.recent_over_runs[-3:] if len(self.recent_over_runs) >= 3 else self.recent_over_runs
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    @property
    def is_chasing(self) -> bool:
        return self.innings == 2

    @property
    def is_powerplay(self) -> bool:
        return self.phase == InningsPhase.POWERPLAY

    @property
    def wickets_in_hand(self) -> int:
        return 10 - self.wickets

    @property
    def had_recent_wicket(self) -> bool:
        """True if a wicket fell in the last 6 balls."""
        if not self.recent_wickets:
            return False
        latest_wicket_over = self.recent_wickets[-1]
        return (self.overs - latest_wicket_over) <= 1.0

    def as_dict(self) -> dict:
        """Flat dict for CSV logging / JSON serialization."""
        return {
            "match_id": self.match_id,
            "format": self.format,
            "team_a": self.team_a,
            "team_b": self.team_b,
            "batting_team": self.batting_team,
            "bowling_team": self.bowling_team,
            "innings": self.innings,
            "runs": self.runs,
            "wickets": self.wickets,
            "overs": self.overs,
            "balls": self.balls,
            "run_rate": self.run_rate,
            "required_run_rate": self.required_run_rate,
            "target_score": self.target_score,
            "overs_remaining": self.overs_remaining,
            "phase": self.phase.value,
            "rolling_rr_3": self.rolling_run_rate_3,
            "venue": self.venue,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        wick = f"{self.runs}/{self.wickets}"
        ov = f"({self.overs} ov)"
        rr = f"RR={self.run_rate:.1f}"
        if self.is_chasing:
            rrr = f"RRR={self.required_run_rate:.1f}"
            need = f"need {self.runs_remaining}"
            return f"{self.batting_team} {wick} {ov} {rr} {rrr} {need}"
        return f"{self.batting_team} {wick} {ov} {rr}"


# ═══════════════════════════════════════════════════════════════════════
#  Cricket Model Output
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CricketModelOutput:
    """Output from the cricket win probability model."""
    p_batting: float           # P(batting team wins)
    p_bowling: float           # P(bowling team wins)
    resource_pct: float        # DLS resource percentage remaining
    par_score: float           # DLS par score at this point
    method: str = "dls_approx"
    momentum_factor: float = 0.0
    pitch_bias: float = 0.0   # Gemini hypothesis
