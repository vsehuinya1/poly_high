"""
Football in-play state and probability output dataclasses.

These are pure data containers — no logic, no side effects.
Designed to be immutable snapshots passed between layers.
"""
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════
#  Input: Game State Snapshot
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FootballState:
    """Immutable snapshot of a football match at a single point in time.

    Attributes:
        minute:           Current match minute (0–95+). Supports stoppage time.
        home_goals:       Goals scored by the home team.
        away_goals:       Goals scored by the away team.
        pre_match_home:   Pre-match P(Home Win)  from 1X2 market.
        pre_match_draw:   Pre-match P(Draw)      from 1X2 market.
        pre_match_away:   Pre-match P(Away Win)  from 1X2 market.
        red_cards_home:   Red cards received by home team (0, 1, 2, …).
        red_cards_away:   Red cards received by away team (0, 1, 2, …).
    """
    minute: float
    home_goals: int
    away_goals: int
    pre_match_home: float
    pre_match_draw: float
    pre_match_away: float
    red_cards_home: int = 0
    red_cards_away: int = 0


# ═══════════════════════════════════════════════════════════════════════
#  Output: Model Probabilities
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Probabilities:
    """Full-time outcome probabilities with model internals for logging.

    Designed to mirror the SNAP output format used by the NBA engine,
    giving full transparency into the model's intermediate state.
    """
    home: float
    draw: float
    away: float

    # ── Context (for logging / SNAP) ─────────────────────────────
    goal_diff: int
    remaining_minutes: float
    lambda_home_remaining: float
    lambda_away_remaining: float

    # ── Extension: base rates from inversion ─────────────────────
    lambda_home_90: float = 0.0
    lambda_away_90: float = 0.0

    def as_dict(self) -> dict:
        """Flat dict for CSV logging or JSON serialization."""
        return {
            "home": round(self.home, 6),
            "draw": round(self.draw, 6),
            "away": round(self.away, 6),
            "goal_diff": self.goal_diff,
            "remaining_minutes": round(self.remaining_minutes, 2),
            "lambda_home_remaining": round(self.lambda_home_remaining, 4),
            "lambda_away_remaining": round(self.lambda_away_remaining, 4),
            "lambda_home_90": round(self.lambda_home_90, 4),
            "lambda_away_90": round(self.lambda_away_90, 4),
        }

    def __str__(self) -> str:
        return (
            f"H={self.home:.4f}  D={self.draw:.4f}  A={self.away:.4f} | "
            f"diff={self.goal_diff:+d}  rem={self.remaining_minutes:.0f}' | "
            f"λH_rem={self.lambda_home_remaining:.3f}  λA_rem={self.lambda_away_remaining:.3f} | "
            f"λH_90={self.lambda_home_90:.3f}  λA_90={self.lambda_away_90:.3f}"
        )
