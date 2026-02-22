"""
In-play fair value probability models for NBA and Football.

NBA: P(home_win) ≈ Φ(S / (σ√T)) — point-spread normal approximation
Football: Poisson-based goal model with red card adjustments
"""
import math
import logging
from typing import Optional
from dataclasses import dataclass

log = logging.getLogger("sports.models")

# ── Constants ────────────────────────────────────────────────────────
# NBA: empirical std dev of scoring rate — ~1.70 points per root-minute
# Calibrated against ESPN/538 win probability curves:
#   +7 with 24min → ~80%, +4 with 2min → ~95%, +10 with 12min → ~96%
NBA_SIGMA = 1.70
NBA_TOTAL_MINUTES = 48.0

# Football: average goals per 90 minutes (league averages)
FOOTBALL_HOME_GOAL_RATE = 1.55 / 90.0   # per minute
FOOTBALL_AWAY_GOAL_RATE = 1.20 / 90.0   # per minute

# Poisson lookup cache
_poisson_cache: dict[tuple[float, int], float] = {}


def _phi(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    # Accurate to ~1e-7, no scipy needed
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    d = 0.3989422804014327  # 1/sqrt(2π)
    poly = (0.319381530 * t - 0.356563782 * t2 + 1.781477937 * t3
            - 1.821255978 * t4 + 1.330274429 * t5)
    cdf = 1.0 - d * math.exp(-0.5 * x * x) * poly
    return 0.5 + sign * (cdf - 0.5)


def _inv_phi(p: float) -> float:
    """Inverse normal CDF approximation (Abramowitz & Stegun)."""
    # Tail approximation for S0 anchor
    p = max(0.0001, min(0.9999, p))
    sign = 1.0
    if p < 0.5:
        sign = -1.0
        t = math.sqrt(-2.0 * math.log(p))
    else:
        sign = 1.0
        t = math.sqrt(-2.0 * math.log(1.0 - p))
    
    num = 2.515517 + 0.802853*t + 0.010328*t*t
    den = 1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t
    val = t - (num / den)
    return sign * val


def _poisson_pmf(lam: float, k: int) -> float:
    """Poisson probability mass function."""
    key = (round(lam, 6), k)
    if key in _poisson_cache:
        return _poisson_cache[key]
    if lam <= 0:
        result = 1.0 if k == 0 else 0.0
    else:
        result = math.exp(-lam + k * math.log(lam) - sum(math.log(i) for i in range(1, k + 1)))
    _poisson_cache[key] = result
    return result


# ═══════════════════════════════════════════════════════════════════════
#  NBA Model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModelOutput:
    """Fair value and intermediate metrics for logging."""
    p_home: float
    p_away: float
    p_draw: float = 0.0
    confidence: float = 1.0
    method: str = ""
    details: str = ""
    # Intermediate metrics for snapshots.csv
    sigma: float = 0.0
    strength_adjustment: float = 0.0
    s_eff: float = 0.0
    z: float = 0.0


def nba_win_probability(
    home_score: int,
    away_score: int,
    adj_seconds: float,
    period: str = "",
    pregame_home_prob: float = 0.5,
) -> ModelOutput:
    """
    NBA win probability with pre-game anchoring and dynamic sigma.
    
    1. Time scaling: use adjusted seconds to derive minutes.
    2. Strength anchor: convert pre-game prob to point-equivalent (S0).
    3. Dynamic sigma: regime-based volatility (2.10 late, 1.70 early).
    4. S_eff: (H-A) + S0 * (T/48).
    """
    t_min = max(adj_seconds / 60.0, 0.01)
    
    # Dynamic Sigma Regime
    if adj_seconds <= 60:
        sigma = 2.10
    elif adj_seconds <= 180:
        sigma = 1.95
    else:
        sigma = 1.70
        
    # Strength Anchor (sigma_base=1.70, T_full=48)
    s0 = 1.70 * math.sqrt(48) * _inv_phi(pregame_home_prob)
    strength_adj = s0 * (t_min / 48.0)
    
    # Effective Score and Z-Score
    s_eff = (home_score - away_score) + strength_adj
    z = s_eff / (sigma * math.sqrt(t_min))
    
    p_home = max(0.001, min(0.999, _phi(z)))
    p_away = 1.0 - p_home
    
    details = f"S_eff={s_eff:.2f}, adj={strength_adj:.2f}, σ={sigma:.2f}, z={z:.3f}"
    
    return ModelOutput(
        p_home=p_home, p_away=p_away, 
        confidence=1.0 - (t_min/48.0)*0.5,
        method="nba_anchor_diffusion",
        details=details,
        sigma=sigma,
        strength_adjustment=strength_adj,
        s_eff=s_eff,
        z=z
    )


# ═══════════════════════════════════════════════════════════════════════
#  Football Model
# ═══════════════════════════════════════════════════════════════════════

def football_win_probability(
    home_goals: int,
    away_goals: int,
    minutes_remaining: float,
    home_red_cards: int = 0,
    away_red_cards: int = 0,
    home_goal_rate: Optional[float] = None,
    away_goal_rate: Optional[float] = None,
) -> ModelOutput:
    """
    Football in-play win probability using Poisson goal model.

    Models remaining goals as independent Poisson processes, then
    sums over all possible goal combinations to get P(H), P(D), P(A).

    Red card adjustment: team with red card has goal rate reduced by ~30%
    per card, opponent's rate increases by ~15%.
    """
    # Default goal rates
    h_rate = home_goal_rate or FOOTBALL_HOME_GOAL_RATE
    a_rate = away_goal_rate or FOOTBALL_AWAY_GOAL_RATE

    # Red card adjustments
    if home_red_cards > 0:
        h_rate *= (0.70 ** home_red_cards)
        a_rate *= (1.15 ** home_red_cards)
    if away_red_cards > 0:
        a_rate *= (0.70 ** away_red_cards)
        h_rate *= (1.15 ** away_red_cards)

    t = max(minutes_remaining, 0.01)

    # Game finished
    if minutes_remaining <= 0:
        if home_goals > away_goals:
            return ModelOutput(1.0, 0.0, 0.0, method="deterministic")
        elif away_goals > home_goals:
            return ModelOutput(0.0, 1.0, 0.0, method="deterministic")
        else:
            return ModelOutput(0.0, 0.0, 1.0, method="deterministic",
                             details="Draw")

    # Expected remaining goals
    lambda_h = h_rate * t
    lambda_a = a_rate * t

    # Sum over possible remaining goals (cap at 8 each for speed)
    max_goals = 8
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    for h_extra in range(max_goals + 1):
        p_h = _poisson_pmf(lambda_h, h_extra)
        for a_extra in range(max_goals + 1):
            p_a = _poisson_pmf(lambda_a, a_extra)
            p_joint = p_h * p_a

            final_h = home_goals + h_extra
            final_a = away_goals + a_extra

            if final_h > final_a:
                p_home += p_joint
            elif final_a > final_h:
                p_away += p_joint
            else:
                p_draw += p_joint

    # Normalize (should be ~1.0 already)
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    # Clip extremes
    p_home = max(0.001, min(0.998, p_home))
    p_away = max(0.001, min(0.998, p_away))
    p_draw = max(0.001, min(0.998, p_draw))

    # Confidence
    pct_complete = 1.0 - (t / 90.0)
    confidence = 0.5 + 0.5 * pct_complete

    goal_diff = home_goals - away_goals
    details = (f"Score {home_goals}-{away_goals}, {t:.0f}min left, "
               f"λH={lambda_h:.2f}, λA={lambda_a:.2f}, "
               f"reds H={home_red_cards} A={away_red_cards}")

    return ModelOutput(p_home, p_away, p_draw, confidence=confidence,
                      method="football_poisson", details=details)


# ═══════════════════════════════════════════════════════════════════════
#  Edge Computation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EdgeSignal:
    """Detected edge between model and market."""
    timestamp: float
    game_id: str
    sport: str
    market_title: str
    token_id: str
    outcome: str            # "home", "away", "draw"
    model_prob: float       # our fair value
    market_prob: float      # polymarket mid price
    edge: float             # model_prob - market_prob (positive = buy signal)
    confidence: float
    game_state: str         # human-readable game state
    direction: str          # "BUY" or "SELL"


def compute_edge(
    model: ModelOutput,
    home_mid: float,
    away_mid: float,
    draw_mid: float = 0.0,
) -> list[tuple[str, float, float, float]]:
    """
    Compare model probabilities to market mid prices.
    Returns list of (outcome, model_prob, market_mid, edge).
    """
    edges = []

    if home_mid > 0:
        e = model.p_home - home_mid
        edges.append(("home", model.p_home, home_mid, e))

    if away_mid > 0:
        e = model.p_away - away_mid
        edges.append(("away", model.p_away, away_mid, e))

    if draw_mid > 0 and model.p_draw > 0:
        e = model.p_draw - draw_mid
        edges.append(("draw", model.p_draw, draw_mid, e))

    return edges
