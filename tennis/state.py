"""
Tennis in-play state dataclass.

Immutable snapshot of a tennis match at a single point in time.
Designed as a pure data container — no logic, no side effects.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
#  Point Score Enum
# ═══════════════════════════════════════════════════════════════════════

class PointScore(IntEnum):
    """Tennis point scoring mapped to ordinal for Markov transitions."""
    LOVE = 0
    P15  = 1
    P30  = 2
    P40  = 3
    AD   = 4    # advantage (only valid during deuce)

    def __str__(self) -> str:
        _MAP = {0: "0", 1: "15", 2: "30", 3: "40", 4: "AD"}
        return _MAP[self.value]

    @classmethod
    def from_str(cls, s: str) -> "PointScore":
        _REV = {"0": cls.LOVE, "15": cls.P15, "30": cls.P30,
                "40": cls.P40, "AD": cls.AD}
        return _REV[str(s).upper().strip()]


# ═══════════════════════════════════════════════════════════════════════
#  Tennis Match State
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TennisState:
    """Immutable snapshot of a tennis match at a single point in time.

    All score fields represent the *current* state after the most recent
    point has been resolved. The convention is Player A vs Player B
    (A is always the first-listed / home player).

    Attributes:
        match_id:             Unique identifier for this match.
        sets_a / sets_b:      Sets won by each player.
        games_a / games_b:    Games won in the *current* set.
        point_a / point_b:    Current point score for each player.
        server_id:            ID of the player currently serving.
        receiver_id:          ID of the player currently receiving.
        pregame_favorite_id:  ID of the pre-match favorite.
        is_tiebreak:          Whether a tiebreak is in progress.
        best_of:              3 or 5 sets (default 3 for ATP/WTA).
        player_p_anchor:      Optional serve-win probability override.
        surface:              Court surface ("hard", "clay", "grass").
        timestamp:            Unix timestamp of this state snapshot.
        last_market_price:    Last observed Polymarket price for the
                              pre-match favorite.
        recent_points:        Last N point outcomes for momentum calc.
                              Each entry: server_id of the point winner.
    """
    match_id: str

    # Set score
    sets_a: int = 0
    sets_b: int = 0

    # Game score in current set
    games_a: int = 0
    games_b: int = 0

    # Point score
    point_a: PointScore = PointScore.LOVE
    point_b: PointScore = PointScore.LOVE

    # Server / receiver
    server_id: str = ""
    receiver_id: str = ""

    # Pre-game
    pregame_favorite_id: str = ""
    player_a_id: str = ""
    player_b_id: str = ""

    # Match format
    best_of: int = 3
    is_tiebreak: bool = False

    # Anchors
    player_p_anchor: Optional[float] = None   # serve-win override
    surface: str = "hard"                      # hard / clay / grass

    # Timestamps and market
    timestamp: float = 0.0
    last_market_price: float = 0.0

    # Momentum tracking
    recent_points: tuple[str, ...] = ()        # last N point-winner IDs

    @property
    def set_score(self) -> tuple[int, int]:
        return (self.sets_a, self.sets_b)

    @property
    def game_score(self) -> tuple[int, int]:
        return (self.games_a, self.games_b)

    @property
    def point_score(self) -> tuple[PointScore, PointScore]:
        return (self.point_a, self.point_b)

    # ── Derived properties ────────────────────────────────────────

    @property
    def is_break_point(self) -> bool:
        """True if the returner can break serve on this point."""
        return compute_break_point_flag(self)

    @property
    def is_deuce(self) -> bool:
        return self.point_a == PointScore.P40 and self.point_b == PointScore.P40

    @property
    def favorite_is_serving(self) -> bool:
        return self.server_id == self.pregame_favorite_id

    @property
    def favorite_sets(self) -> int:
        if self.pregame_favorite_id == self.player_a_id:
            return self.sets_a
        return self.sets_b

    @property
    def underdog_sets(self) -> int:
        if self.pregame_favorite_id == self.player_a_id:
            return self.sets_b
        return self.sets_a

    def as_dict(self) -> dict:
        """Flat dict for CSV logging / JSON serialization."""
        return {
            "match_id": self.match_id,
            "sets_a": self.sets_a,
            "sets_b": self.sets_b,
            "games_a": self.games_a,
            "games_b": self.games_b,
            "point_a": str(self.point_a),
            "point_b": str(self.point_b),
            "server_id": self.server_id,
            "receiver_id": self.receiver_id,
            "is_break_point": self.is_break_point,
            "is_tiebreak": self.is_tiebreak,
            "pregame_favorite_id": self.pregame_favorite_id,
            "surface": self.surface,
            "timestamp": self.timestamp,
            "last_market_price": self.last_market_price,
        }

    def __str__(self) -> str:
        return (
            f"Sets({self.sets_a}-{self.sets_b}) "
            f"Games({self.games_a}-{self.games_b}) "
            f"Points({self.point_a}-{self.point_b}) "
            f"Server={self.server_id} "
            f"{'TB' if self.is_tiebreak else ''}"
            f"{'BP!' if self.is_break_point else ''}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Point Event — raw input from data feed
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TennisPointEvent:
    """A single point result from the data feed."""
    match_id: str
    point_winner_id: str          # ID of player who won this point
    new_sets_a: int
    new_sets_b: int
    new_games_a: int
    new_games_b: int
    new_point_a: str              # "0", "15", "30", "40", "AD"
    new_point_b: str
    new_server_id: str
    is_tiebreak: bool = False
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
#  State Transition
# ═══════════════════════════════════════════════════════════════════════

def update_from_point(state: TennisState, event: TennisPointEvent) -> TennisState:
    """Create a new TennisState from the current state + a point event.

    Since TennisState is frozen, this returns a *new* instance.
    The recent_points window is capped at 20 entries.
    """
    recent = state.recent_points + (event.point_winner_id,)
    if len(recent) > 20:
        recent = recent[-20:]

    return TennisState(
        match_id=state.match_id,
        sets_a=event.new_sets_a,
        sets_b=event.new_sets_b,
        games_a=event.new_games_a,
        games_b=event.new_games_b,
        point_a=PointScore.from_str(event.new_point_a),
        point_b=PointScore.from_str(event.new_point_b),
        server_id=event.new_server_id,
        receiver_id=(state.player_b_id if event.new_server_id == state.player_a_id
                     else state.player_a_id),
        pregame_favorite_id=state.pregame_favorite_id,
        player_a_id=state.player_a_id,
        player_b_id=state.player_b_id,
        best_of=state.best_of,
        is_tiebreak=event.is_tiebreak,
        player_p_anchor=state.player_p_anchor,
        surface=state.surface,
        timestamp=event.timestamp or time.time(),
        last_market_price=state.last_market_price,
        recent_points=recent,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Derived Computations
# ═══════════════════════════════════════════════════════════════════════

def compute_break_point_flag(state: TennisState) -> bool:
    """True if the returner can win this game on the current point.

    Break point occurs when:
      - Returner is at 40 (or AD) and server is at 30 or below, OR
      - Returner has AD in a deuce game.

    In tiebreaks there are no "break points" in the traditional sense,
    but we still track when the returner can win a mini-break.
    """
    pa, pb = state.point_a, state.point_b

    # Determine which point score belongs to server vs returner
    if state.server_id == state.player_a_id:
        s_pts, r_pts = pa, pb
    else:
        s_pts, r_pts = pb, pa

    if state.is_tiebreak:
        # In tiebreak: returner can win a "mini-break" point
        # if they're ahead or at match point. We simplify: not a
        # traditional break point.
        return False

    # Returner at 40 and server at 30 or below
    if r_pts == PointScore.P40 and s_pts.value <= PointScore.P30.value:
        return True
    # Returner has AD
    if r_pts == PointScore.AD:
        return True

    return False


def compute_momentum_delta(state: TennisState) -> float:
    """Momentum indicator for Player A: fraction of last N points won.

    Returns a value in [-1.0, +1.0]:
      +1.0 = Player A won all recent points
      -1.0 = Player B won all recent points
       0.0 = even split or no data
    """
    if not state.recent_points:
        return 0.0
    n = len(state.recent_points)
    a_wins = sum(1 for pid in state.recent_points if pid == state.player_a_id)
    # Scale to [-1, 1]: (a_wins/n - 0.5) * 2
    return (a_wins / n - 0.5) * 2.0


# ═══════════════════════════════════════════════════════════════════════
#  Model Output
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TennisModelOutput:
    """Fair match-win probabilities with model internals for logging."""
    p_a: float                    # P(Player A wins match)
    p_b: float                    # P(Player B wins match)
    p_serve: float                # service point-win probability used
    game_win_prob: float          # P(server wins current game)
    set_win_prob_a: float         # P(A wins current set)
    method: str = "markov"

    @property
    def p_favorite(self) -> float:
        """Convenience — returns the higher probability."""
        return max(self.p_a, self.p_b)

    def as_dict(self) -> dict:
        return {
            "p_a": round(self.p_a, 6),
            "p_b": round(self.p_b, 6),
            "p_serve": round(self.p_serve, 4),
            "game_win_prob": round(self.game_win_prob, 6),
            "set_win_prob_a": round(self.set_win_prob_a, 6),
            "method": self.method,
        }

    def __str__(self) -> str:
        return (
            f"A={self.p_a:.4f}  B={self.p_b:.4f} | "
            f"p_serve={self.p_serve:.3f}  G_win={self.game_win_prob:.4f} | "
            f"S_win_A={self.set_win_prob_a:.4f}"
        )
