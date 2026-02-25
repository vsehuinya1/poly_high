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

# Football: model constants
FOOTBALL_TOTAL_MINUTES = 90.0
FOOTBALL_GOAL_CAP = 8         # max additional goals per team in Poisson sum
FOOTBALL_RED_CARD_SELF_PENALTY = 0.30   # reduce own λ by 30% per net red card
FOOTBALL_RED_CARD_OPP_BOOST = 0.15      # boost opponent λ by 15% per net red card
FOOTBALL_FALLBACK_TOTAL_GOALS = 2.65    # league-average total xG (safety fallback)
FOOTBALL_FALLBACK_HOME_SHARE = 0.57     # home share of total goals in fallback

# ── Precomputed log-factorial table (k = 0..8) ──────────────────────
# Eliminates repeated math.log(i) calls inside Poisson PMF hot path.
_LOG_FACTORIAL = [0.0] * 9
for _k in range(1, 9):
    _LOG_FACTORIAL[_k] = _LOG_FACTORIAL[_k - 1] + math.log(_k)

# ── Football λ inversion cache ───────────────────────────────────────
# Key: (round(p_home, 4), round(p_draw, 4), round(p_away, 4))
# Value: (λ_home_90, λ_away_90)
_lambda_cache: dict[tuple[float, float, float], tuple[float, float]] = {}

# Grid search bounds for inversion
_GRID_LO = 0.30
_GRID_HI = 3.51
_GRID_STEP = 0.01


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
    p = max(0.0001, min(0.9999, p))
    # Exact zero for 50/50 — avoids spurious anchor bias
    if abs(p - 0.5) < 0.001:
        return 0.0
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
        num = 2.515517 + 0.802853*t + 0.010328*t*t
        den = 1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t
        return -(t - (num / den))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        num = 2.515517 + 0.802853*t + 0.010328*t*t
        den = 1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t
        return t - (num / den)


def _poisson_pmf(lam: float, k: int) -> float:
    """Poisson probability mass function using precomputed log-factorials."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    # Use precomputed table for k <= 8, fall back to lgamma for larger k
    log_fact = _LOG_FACTORIAL[k] if k < len(_LOG_FACTORIAL) else math.lgamma(k + 1)
    return math.exp(-lam + k * math.log(lam) - log_fact)


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
    3. Dynamic sigma: regime-based volatility.
       - Last 60s:  σ=2.10 (fouling, 3pt heaves → high variance)
       - Last 180s: σ=1.95 (intentional fouls begin)
       - Default:   σ=1.70 (calibrated baseline)
    4. S_eff: (H-A) + S0 * (T/48).
    """
    score_diff = home_score - away_score

    # ── Deterministic: game finished ──────────────────────────────
    if adj_seconds <= 0:
        if score_diff > 0:
            return ModelOutput(1.0, 0.0, method="deterministic",
                             details=f"Home wins by {score_diff}")
        elif score_diff < 0:
            return ModelOutput(0.0, 1.0, method="deterministic",
                             details=f"Away wins by {-score_diff}")
        else:
            return ModelOutput(0.5, 0.5, method="tied_end",
                             details="Tied at end of regulation, OT likely")

    # ── OT handling ───────────────────────────────────────────────
    # OT is 5 minutes. If period contains "OT", cap T at 5 min.
    is_ot = "OT" in str(period).upper()
    if is_ot:
        adj_seconds = min(adj_seconds, 300.0)  # cap at 5 min

    t_min = adj_seconds / 60.0
    
    # Dynamic Sigma Regime
    if adj_seconds <= 60:
        sigma = 2.10
    elif adj_seconds <= 180:
        sigma = 1.95
    else:
        sigma = 1.70
        
    # Strength Anchor (sigma_base=1.70, T_full=48)
    s0 = 1.70 * math.sqrt(48.0) * _inv_phi(pregame_home_prob)
    strength_adj = s0 * (t_min / 48.0)
    
    # Effective Score and Z-Score
    s_eff = score_diff + strength_adj
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
#  Football: Pre-match → λ Inversion (grid search)
# ═══════════════════════════════════════════════════════════════════════

def _poisson_1x2(lam_h: float, lam_a: float, cap: int = 5) -> tuple[float, float, float]:
    """Compute P(H), P(D), P(A) for a full match under Poisson model."""
    p_h = p_d = p_a = 0.0
    for h in range(cap + 1):
        ph = _poisson_pmf(lam_h, h)
        for a in range(cap + 1):
            pa = _poisson_pmf(lam_a, a)
            joint = ph * pa
            if h > a:
                p_h += joint
            elif h == a:
                p_d += joint
            else:
                p_a += joint
    total = p_h + p_d + p_a
    if total > 0:
        p_h /= total
        p_d /= total
        p_a /= total
    return p_h, p_d, p_a


def invert_1x2_to_lambdas(
    p_home: float,
    p_draw: float,
    p_away: float,
) -> tuple[float, float, float]:
    """Reverse-engineer implied 90-minute goal intensities from 1X2 odds.

    Two-pass grid search:
        Pass 1 (coarse): step=0.10 over [0.30, 3.50]² — ~1k evaluations
        Pass 2 (fine):   step=0.01 in ±0.15 around coarse best — ~900 evaluations
    Total ~2k evaluations, runs in <500ms.

    MUST be called at startup / link initialization — never during
    live tick processing.

    Args:
        p_home: Pre-match P(Home Win).
        p_draw: Pre-match P(Draw).
        p_away: Pre-match P(Away Win).

    Returns:
        (λ_home_90, λ_away_90, sse) — implied goals per 90 minutes
        and the fitting error.
    """
    key = (round(p_home, 4), round(p_draw, 4), round(p_away, 4))
    if key in _lambda_cache:
        lh, la = _lambda_cache[key]
        fh, fd, fa = _poisson_1x2(lh, la)
        sse = (fh - p_home)**2 + (fd - p_draw)**2 + (fa - p_away)**2
        return lh, la, sse

    # Validate inputs
    total = p_home + p_draw + p_away
    if total < 0.5 or p_home <= 0 or p_draw <= 0 or p_away <= 0:
        log.warning(
            "degenerate pre-match probs (H=%.3f D=%.3f A=%.3f, sum=%.3f) "
            "— using fallback rates",
            p_home, p_draw, p_away, total,
        )
        lh = FOOTBALL_FALLBACK_TOTAL_GOALS * FOOTBALL_FALLBACK_HOME_SHARE
        la = FOOTBALL_FALLBACK_TOTAL_GOALS * (1 - FOOTBALL_FALLBACK_HOME_SHARE)
        _lambda_cache[key] = (lh, la)
        return lh, la, 999.0

    # Normalize targets
    ph_t, pd_t, pa_t = p_home / total, p_draw / total, p_away / total

    cap = 5  # goals cap for fitting

    # Helper: build list of λ values
    def _grid(lo, hi, step):
        out, v = [], lo
        while v <= hi + step * 0.01:
            out.append(round(v, 4))
            v += step
        return out

    # Helper: precompute PMF vectors, then search all pairs
    def _search(h_vals, a_vals, b_lh, b_la, b_sse):
        # Build PMF lookup — O(unique_vals × cap) math.exp calls
        pmf = {}
        for v in h_vals + a_vals:
            k = round(v, 4)
            if k not in pmf:
                pmf[k] = [_poisson_pmf(v, g) for g in range(cap + 1)]
        # Grid search using precomputed vectors — O(grid² × cap²) float ops only
        for lhv in h_vals:
            ph_vec = pmf[round(lhv, 4)]
            for lav in a_vals:
                pa_vec = pmf[round(lav, 4)]
                sh = sd = sa = 0.0
                for h in range(cap + 1):
                    pv = ph_vec[h]
                    for a in range(cap + 1):
                        j = pv * pa_vec[a]
                        if h > a:    sh += j
                        elif h == a: sd += j
                        else:        sa += j
                t = sh + sd + sa
                if t > 0:
                    sh /= t; sd /= t; sa /= t
                e = (sh - ph_t)**2 + (sd - pd_t)**2 + (sa - pa_t)**2
                if e < b_sse:
                    b_sse = e; b_lh = lhv; b_la = lav
        return b_lh, b_la, b_sse

    best_lh = FOOTBALL_FALLBACK_TOTAL_GOALS * FOOTBALL_FALLBACK_HOME_SHARE
    best_la = FOOTBALL_FALLBACK_TOTAL_GOALS * (1 - FOOTBALL_FALLBACK_HOME_SHARE)
    best_sse = float("inf")

    # Pass 1: coarse (step=0.10, ~33 values → ~1k pairs)
    coarse = _grid(_GRID_LO, _GRID_HI, 0.10)
    best_lh, best_la, best_sse = _search(coarse, coarse, best_lh, best_la, best_sse)

    # Pass 2: fine (step=0.01, ±0.15 → ~31 values per axis → ~961 pairs)
    fine_h = _grid(max(_GRID_LO, best_lh - 0.15), min(_GRID_HI, best_lh + 0.15), 0.01)
    fine_a = _grid(max(_GRID_LO, best_la - 0.15), min(_GRID_HI, best_la + 0.15), 0.01)
    best_lh, best_la, best_sse = _search(fine_h, fine_a, best_lh, best_la, best_sse)

    _lambda_cache[key] = (best_lh, best_la)
    log.info(
        "INVERSION | target=(%.3f, %.3f, %.3f) → λ=(%.2f, %.2f) SSE=%.6f",
        ph_t, pd_t, pa_t, best_lh, best_la, best_sse,
    )
    return best_lh, best_la, best_sse


# ═══════════════════════════════════════════════════════════════════════
#  Football Model (forward Poisson only — λ must be pre-warmed)
# ═══════════════════════════════════════════════════════════════════════

def football_win_probability(
    home_goals: int,
    away_goals: int,
    minutes_remaining: float,
    home_red_cards: int = 0,
    away_red_cards: int = 0,
    lambda_home_90: Optional[float] = None,
    lambda_away_90: Optional[float] = None,
) -> ModelOutput:
    """Football in-play win probability using forward Poisson only.

    λ values MUST be pre-warmed via invert_1x2_to_lambdas() at link
    initialization time. This function never performs grid search.

    Args:
        home_goals:       Current home score.
        away_goals:       Current away score.
        minutes_remaining: Minutes left in regulation (0–90+).
        home_red_cards:   Red cards received by home team.
        away_red_cards:   Red cards received by away team.
        lambda_home_90:   Pre-warmed home expected goals per 90 min. REQUIRED.
        lambda_away_90:   Pre-warmed away expected goals per 90 min. REQUIRED.

    Returns:
        ModelOutput with P(H), P(A), P(D) and diagnostics.

    Raises:
        RuntimeError: If λ values are not provided (not pre-warmed).
    """
    if lambda_home_90 is None or lambda_away_90 is None:
        raise RuntimeError(
            "Football λ not pre-warmed. Call invert_1x2_to_lambdas() at "
            "link initialization, not during live tick processing."
        )

    # ── Game finished ─────────────────────────────────────────────
    if minutes_remaining <= 0:
        if home_goals > away_goals:
            return ModelOutput(1.0, 0.0, 0.0, method="deterministic")
        elif away_goals > home_goals:
            return ModelOutput(0.0, 1.0, 0.0, method="deterministic")
        else:
            return ModelOutput(0.0, 0.0, 1.0, method="deterministic",
                             details="Draw")

    # ── Scale to remaining time ───────────────────────────────────
    t = max(minutes_remaining, 0.01)
    lambda_h = lambda_home_90 * (t / FOOTBALL_TOTAL_MINUTES)
    lambda_a = lambda_away_90 * (t / FOOTBALL_TOTAL_MINUTES)

    # ── Red card adjustments (net disadvantage) ───────────────────
    net = home_red_cards - away_red_cards
    if net > 0:
        lambda_h *= (1.0 - FOOTBALL_RED_CARD_SELF_PENALTY) ** net
        lambda_a *= (1.0 + FOOTBALL_RED_CARD_OPP_BOOST) ** net
    elif net < 0:
        abs_net = abs(net)
        lambda_a *= (1.0 - FOOTBALL_RED_CARD_SELF_PENALTY) ** abs_net
        lambda_h *= (1.0 + FOOTBALL_RED_CARD_OPP_BOOST) ** abs_net

    # ── Poisson convolution ───────────────────────────────────────
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    for h_extra in range(FOOTBALL_GOAL_CAP + 1):
        p_h = _poisson_pmf(lambda_h, h_extra)
        for a_extra in range(FOOTBALL_GOAL_CAP + 1):
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

    # ── Normalize + clip ──────────────────────────────────────────
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    p_home = max(0.001, min(0.998, p_home))
    p_away = max(0.001, min(0.998, p_away))
    p_draw = max(0.001, min(0.998, p_draw))

    # ── Confidence + diagnostics ──────────────────────────────────
    pct_complete = 1.0 - (t / FOOTBALL_TOTAL_MINUTES)
    confidence = 0.5 + 0.5 * pct_complete

    details = (f"Score {home_goals}-{away_goals}, {t:.0f}min left, "
               f"λH_rem={lambda_h:.3f}, λA_rem={lambda_a:.3f}, "
               f"λH_90={lambda_home_90:.2f}, λA_90={lambda_away_90:.2f}, "
               f"reds H={home_red_cards} A={away_red_cards}")

    return ModelOutput(p_home, p_away, p_draw, confidence=confidence,
                      method="football_poisson_prewarmed", details=details)


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
