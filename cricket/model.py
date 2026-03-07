"""
Cricket in-play win probability model — DLS-inspired resource model.

Pure structural model: resources remaining = f(overs, wickets).
No ML. No external data. Deterministic given match state.

The model computes win probabilities by comparing remaining
batting resources against required scoring rate, using a
simplified Duckworth-Lewis-Stern resource table.

References:
    - Duckworth & Lewis (1998), "A fair method for resetting
      the target in interrupted one-day cricket matches"
    - Stern (2009), updated DLS methodology
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache

from cricket.state import CricketState, CricketModelOutput, InningsPhase

log = logging.getLogger("cricket.model")


# ═══════════════════════════════════════════════════════════════════════
#  DLS Resource Table (T20 simplified)
# ═══════════════════════════════════════════════════════════════════════

# Resource % remaining given (overs_remaining, wickets_lost)
# Rows: overs remaining (0-20), Columns: wickets lost (0-10)
# Values: percentage of batting resources still available
# Source: Simplified from official DLS tables, tuned for T20 scoring rates

_DLS_RESOURCES = {
    # overs_remaining: {wickets_lost: resource_pct}
    20: {0: 100.0, 1: 93.4, 2: 85.1, 3: 74.9, 4: 62.7, 5: 49.0, 6: 34.9, 7: 22.0, 8: 11.9, 9: 4.7, 10: 0.0},
    18: {0: 93.6, 1: 87.9, 2: 80.5, 3: 71.3, 4: 60.0, 5: 47.2, 6: 33.8, 7: 21.4, 8: 11.6, 9: 4.6, 10: 0.0},
    16: {0: 86.7, 1: 81.8, 2: 75.4, 3: 67.2, 4: 56.9, 5: 45.1, 6: 32.5, 7: 20.7, 8: 11.3, 9: 4.5, 10: 0.0},
    14: {0: 79.0, 1: 75.1, 2: 69.6, 3: 62.5, 4: 53.3, 5: 42.5, 6: 30.9, 7: 19.9, 8: 10.9, 9: 4.4, 10: 0.0},
    12: {0: 70.6, 1: 67.5, 2: 63.0, 3: 57.0, 4: 49.1, 5: 39.5, 6: 29.0, 7: 18.8, 8: 10.4, 9: 4.2, 10: 0.0},
    10: {0: 61.6, 1: 59.2, 2: 55.8, 3: 50.8, 4: 44.1, 5: 35.9, 6: 26.6, 7: 17.5, 8: 9.8,  9: 4.0, 10: 0.0},
    8:  {0: 51.8, 1: 50.1, 2: 47.6, 3: 43.7, 4: 38.4, 5: 31.7, 6: 23.9, 7: 15.9, 8: 9.0,  9: 3.7, 10: 0.0},
    6:  {0: 41.1, 1: 40.0, 2: 38.3, 3: 35.6, 4: 31.8, 5: 26.6, 6: 20.4, 7: 13.9, 8: 8.0,  9: 3.4, 10: 0.0},
    4:  {0: 29.4, 1: 28.8, 2: 27.8, 3: 26.2, 4: 23.8, 5: 20.4, 6: 16.1, 7: 11.3, 8: 6.6,  9: 2.9, 10: 0.0},
    2:  {0: 16.3, 1: 16.1, 2: 15.7, 3: 15.1, 4: 14.0, 5: 12.4, 6: 10.2, 7: 7.5,  8: 4.6,  9: 2.1, 10: 0.0},
    1:  {0: 8.9,  1: 8.8,  2: 8.6,  3: 8.3,  4: 7.9,  5: 7.1,  6: 6.0,  7: 4.6,  8: 2.9,  9: 1.4, 10: 0.0},
    0:  {0: 0.0,  1: 0.0,  2: 0.0,  3: 0.0,  4: 0.0,  5: 0.0,  6: 0.0,  7: 0.0,  8: 0.0,  9: 0.0, 10: 0.0},
}


def _interpolate_resource(overs_remaining: float, wickets_lost: int) -> float:
    """Linearly interpolate DLS resource percentage.

    Args:
        overs_remaining: Fractional overs remaining (e.g. 12.3).
        wickets_lost: Wickets lost (0-10).

    Returns:
        Resource percentage (0-100).
    """
    wickets_lost = min(10, max(0, wickets_lost))
    overs_remaining = max(0.0, min(20.0, overs_remaining))

    # Find bounding rows
    table_overs = sorted(_DLS_RESOURCES.keys())
    lower = 0
    upper = 0
    for o in table_overs:
        if o <= overs_remaining:
            lower = o
        if o >= overs_remaining:
            upper = o
            break

    if lower == upper:
        return _DLS_RESOURCES[lower].get(wickets_lost, 0.0)

    # Linear interpolation between rows
    r_lower = _DLS_RESOURCES[lower].get(wickets_lost, 0.0)
    r_upper = _DLS_RESOURCES[upper].get(wickets_lost, 0.0)

    frac = (overs_remaining - lower) / (upper - lower) if upper != lower else 0.0
    return r_lower + frac * (r_upper - r_lower)


# ═══════════════════════════════════════════════════════════════════════
#  Win Probability Model
# ═══════════════════════════════════════════════════════════════════════

class CricketWinModel:
    """Stateless, deterministic in-play win probability model.

    For FIRST INNINGS:
        Uses current run rate vs average T20 score to estimate
        probability of setting a competitive total.

    For SECOND INNINGS (chase):
        Compares resources consumed vs resources required using
        the DLS resource table, adjusted for momentum.

    Usage:
        model = CricketWinModel()
        output = model.get_win_prob(state)
        print(output.p_batting, output.p_bowling)
    """

    # Average T20I first innings total (2020-2024)
    AVG_T20_TOTAL = 165.0

    def get_win_prob(self, state: CricketState) -> CricketModelOutput:
        """Compute win probability for both teams.

        Pipeline:
            1. Calculate resources remaining via DLS table.
            2. Compute par score at current stage.
            3. Compare actual score vs par to derive probability.
            4. Apply momentum adjustment from rolling run rate.

        Args:
            state: Immutable match snapshot.

        Returns:
            CricketModelOutput with probability and diagnostics.
        """
        if state.innings == 1:
            return self._first_innings(state)
        else:
            return self._second_innings(state)

    def _first_innings(self, state: CricketState) -> CricketModelOutput:
        """First innings — estimate projected total vs average."""
        resource_at_start = _interpolate_resource(20.0, 0)
        resource_now = _interpolate_resource(state.overs_remaining, state.wickets)
        resource_used = resource_at_start - resource_now

        if resource_used <= 0:
            return CricketModelOutput(
                p_batting=0.5, p_bowling=0.5,
                resource_pct=100.0, par_score=0.0,
            )

        # Project total based on resources consumed
        projected_total = (state.runs / resource_used) * resource_at_start

        # Compare projected total to average
        z = (projected_total - self.AVG_T20_TOTAL) / 25.0  # 25 runs ≈ 1 SD
        z = max(-4.0, min(4.0, z))

        # Sigmoid mapping: above-average total → higher batting team probability
        # But in 1st innings, "batting team wins" means they set a good total
        p_batting = 1.0 / (1.0 + math.exp(-0.5 * z))

        # Momentum factor
        momentum = self._momentum_factor(state)

        return CricketModelOutput(
            p_batting=round(p_batting, 4),
            p_bowling=round(1.0 - p_batting, 4),
            resource_pct=round(resource_now, 1),
            par_score=round(projected_total, 0),
            momentum_factor=round(momentum, 3),
        )

    def _second_innings(self, state: CricketState) -> CricketModelOutput:
        """Second innings — chase probability via DLS resource comparison."""
        resource_now = _interpolate_resource(state.overs_remaining, state.wickets)
        resource_at_start = _interpolate_resource(20.0, 0)

        # Par score = target × (resources used / total resources)
        resource_used = resource_at_start - resource_now
        if resource_at_start <= 0:
            par_score = 0.0
        else:
            par_score = state.target_score * (resource_used / resource_at_start)

        # How far ahead or behind the chasing team is vs par
        runs_above_par = state.runs - par_score

        # Normalize: ±30 runs ≈ ±2 SD in T20
        z = runs_above_par / 15.0
        z = max(-4.0, min(4.0, z))

        # Momentum adjustment: recent run rate vs required
        momentum = self._momentum_factor(state)

        # Combined z-score including momentum
        z_adj = z + (momentum * 0.3)  # momentum contributes 30% weight
        z_adj = max(-4.0, min(4.0, z_adj))

        # Sigmoid
        p_chasing = 1.0 / (1.0 + math.exp(-0.8 * z_adj))

        # Pitch bias hypothesis (Gemini) — log-only
        pitch_bias = self._pitch_bias(state)

        return CricketModelOutput(
            p_batting=round(p_chasing, 4),
            p_bowling=round(1.0 - p_chasing, 4),
            resource_pct=round(resource_now, 1),
            par_score=round(par_score, 0),
            momentum_factor=round(momentum, 3),
            pitch_bias=round(pitch_bias, 3),
        )

    @staticmethod
    def _momentum_factor(state: CricketState) -> float:
        """Calculate momentum from recent run rate vs required.

        Returns:
            Positive = batting team has momentum.
            Negative = batting team is behind.
            Range: approximately -3 to +3.
        """
        if not state.recent_over_runs or len(state.recent_over_runs) < 2:
            return 0.0

        rolling_rr = state.rolling_run_rate_3

        if state.is_chasing and state.required_run_rate > 0:
            delta = rolling_rr - state.required_run_rate
        elif state.run_rate > 0:
            delta = rolling_rr - state.run_rate  # momentum vs own average
        else:
            return 0.0

        return delta / 3.0  # normalize: ±3 rpo = ±1.0

    @staticmethod
    def _pitch_bias(state: CricketState) -> float:
        """Gemini Hypothesis: Pitch Bias (log-only).

        Ahmedabad red-soil pitch: if spinner economy < 6.5 in
        first 10 overs, reduce chase probability.

        Returns adjustment factor (negative = harder chase).
        """
        if state.innings != 2:
            return 0.0
        if state.overs < 5.0:
            return 0.0  # not enough data
        if state.spinner_economy > 0 and state.spinner_economy < 6.5:
            bias = -0.05 * (6.5 - state.spinner_economy) / 6.5
            log.debug("PITCH_BIAS | venue=%s spin_eco=%.1f bias=%.3f",
                       state.venue, state.spinner_economy, bias)
            return bias
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Convenience — module-level function
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_MODEL = CricketWinModel()


def get_win_prob(state: CricketState) -> CricketModelOutput:
    """Module-level convenience wrapper."""
    return _DEFAULT_MODEL.get_win_prob(state)
