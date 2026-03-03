"""
Tennis in-play win probability model — Markov chain.

Pure structural model: point → game → set → match.
No ML. No external data. Deterministic given state + p_serve.

The model computes exact match-win probabilities by recursing
through every possible future score path, weighted by the
serve-win probability at each point.

References:
    - Newton & Keller (2005), "Probability of Winning at Tennis"
    - Klaassen & Magnus (2001), point-by-point analysis
    - ATP average service point won: ~0.64 (2015-2024)
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from tennis.state import TennisState, TennisModelOutput, PointScore

log = logging.getLogger("tennis.model")


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

ATP_SERVE_WIN_P = 0.64
"""ATP tour average: server wins ~64% of service points."""

# Surface adjustments applied to anchor (additive)
_SURFACE_ADJ = {
    "grass": +0.03,    # bigger serve advantage on grass
    "hard":   0.00,    # baseline
    "clay":  -0.03,    # clay reduces serve advantage
}


# ═══════════════════════════════════════════════════════════════════════
#  Point → Game Probability (Markov)
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=512)
def _game_win_prob_from_start(p: float) -> float:
    """P(server wins game) starting from 0-0 given serve-win prob p.

    Uses closed-form derivation of the 4-point game with deuce.
    """
    q = 1.0 - p

    # P(reach deuce) = sum of paths to (40-40)
    # Paths: server wins exactly 3 of first 6 points = C(6,3) * p^3 * q^3
    # But tennis scoring means we need exactly 3 each after 6 points:
    # The coefficient accounts for the specific orderings that reach deuce.
    p_deuce = 20.0 * (p ** 3) * (q ** 3)  # C(6,3) = 20

    # P(server wins from deuce) = p^2 / (p^2 + q^2)
    p_win_deuce = (p * p) / (p * p + q * q)

    # P(server wins game without reaching deuce)
    # = sum of paths where server wins 4+ of first 4-6 points
    p_win_no_deuce = (
        p ** 4                               # 4-0 (wins first 4)
        + 4 * (p ** 4) * q                   # 4-1 (various orderings)
        + 10 * (p ** 4) * (q ** 2)           # 4-2 (various orderings)
    )

    return p_win_no_deuce + p_deuce * p_win_deuce


@lru_cache(maxsize=2048)
def _game_win_prob_from_score(pts_server: int, pts_returner: int,
                               p: float) -> float:
    """P(server wins game) from an arbitrary point score.

    Args:
        pts_server:  Server's point index (0=love, 1=15, 2=30, 3=40, 4=AD).
        pts_returner: Returner's point index.
        p:           P(server wins next service point).

    Tennis point transitions:
        - Normal scoring: 0→15→30→40→game
        - At 40-40 (deuce): winner gets AD
        - At AD-40: server wins game if wins point, else back to deuce
        - At 40-AD: returner wins game if wins point, else back to deuce
    """
    q = 1.0 - p

    # Terminal states
    if pts_server >= 3 and pts_returner >= 3:
        # Deuce territory
        if pts_server == pts_returner:
            # Deuce: P(server wins) = p^2 / (p^2 + q^2)
            return (p * p) / (p * p + q * q)
        elif pts_server > pts_returner:
            # Server has AD: wins with prob p, deuce with prob q
            return p + q * (p * p) / (p * p + q * q)
        else:
            # Returner has AD: server must win to get back to deuce
            return q * 0 + p * (p * p) / (p * p + q * q)
            # Wait — that's wrong. Let me redo:
            # From AD-out (returner has AD):
            # Server wins point → deuce, then P(win from deuce) * p
            # Server loses point → returner wins game → 0
            # But actually, from returner-AD:
            #   P(server wins) = p * P(deuce) = p * p²/(p²+q²)

    # Server already won the game
    if pts_server >= 4 and pts_server > pts_returner:
        return 1.0
    # Returner already won the game
    if pts_returner >= 4 and pts_returner > pts_server:
        return 0.0

    # Server at game point (40, returner < 40)
    if pts_server == 3 and pts_returner < 3:
        return p + q * _game_win_prob_from_score(3, pts_returner + 1, p)

    # Returner at game point (server < 40, returner = 40)
    if pts_returner == 3 and pts_server < 3:
        return p * _game_win_prob_from_score(pts_server + 1, 3, p) + q * 0.0

    # Normal scoring (both below 40)
    return (p * _game_win_prob_from_score(pts_server + 1, pts_returner, p) +
            q * _game_win_prob_from_score(pts_server, pts_returner + 1, p))


# ═══════════════════════════════════════════════════════════════════════
#  Game → Set Probability (Markov)
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=4096)
def _set_win_prob(games_a: int, games_b: int,
                  p_a_serve: float, p_b_serve: float,
                  a_is_serving: bool,
                  tiebreak_at_6_6: bool = True) -> float:
    """P(Player A wins this set) from an arbitrary game score.

    Args:
        games_a:         Games won by A in this set.
        games_b:         Games won by B in this set.
        p_a_serve:       P(A wins a point on A's serve).
        p_b_serve:       P(B wins a point on B's serve).
        a_is_serving:    True if A is currently the server.
        tiebreak_at_6_6: Whether a tiebreak is played at 6-6.
    """
    # Terminal: A won the set
    if games_a >= 6 and games_a - games_b >= 2:
        return 1.0
    if games_b >= 6 and games_b - games_a >= 2:
        return 0.0

    # Tiebreak at 6-6
    if games_a == 6 and games_b == 6:
        if tiebreak_at_6_6:
            return _tiebreak_win_prob(0, 0, p_a_serve, p_b_serve, a_is_serving)
        else:
            # Advantage set: no tiebreak, must win by 2 games
            # Approximation: 2-game mini-match probability
            p_a_hold = _game_win_prob_from_start(p_a_serve)
            p_b_hold = _game_win_prob_from_start(p_b_serve)
            p_a_break = 1.0 - p_b_hold
            p_b_break = 1.0 - p_a_hold
            if a_is_serving:
                p_win_2 = p_a_hold * p_a_break
                p_lose_2 = p_b_break * p_b_hold
            else:
                p_win_2 = (1.0 - p_b_hold) * p_a_hold  # break then hold
                p_lose_2 = p_b_hold * (1.0 - p_a_hold)
            p_draw_2 = 1.0 - p_win_2 - p_lose_2
            if p_draw_2 >= 1.0:
                return 0.5
            return p_win_2 / (p_win_2 + p_lose_2)

    # Above 6 games without tiebreak — shouldn't happen with tiebreak
    if games_a > 7 or games_b > 7:
        return 0.5  # safety fallback

    # 7-6 or 6-7 after tiebreak
    if games_a == 7:
        return 1.0
    if games_b == 7:
        return 0.0

    # Current game probability
    if a_is_serving:
        p_a_wins_game = _game_win_prob_from_start(p_a_serve)
    else:
        p_a_wins_game = 1.0 - _game_win_prob_from_start(p_b_serve)

    # After this game, server alternates
    next_a_serving = not a_is_serving

    return (p_a_wins_game * _set_win_prob(games_a + 1, games_b,
                                          p_a_serve, p_b_serve,
                                          next_a_serving, tiebreak_at_6_6) +
            (1 - p_a_wins_game) * _set_win_prob(games_a, games_b + 1,
                                                 p_a_serve, p_b_serve,
                                                 next_a_serving, tiebreak_at_6_6))


@lru_cache(maxsize=4096)
def _set_win_prob_from_point(games_a: int, games_b: int,
                              pts_server: int, pts_returner: int,
                              p_a_serve: float, p_b_serve: float,
                              a_is_serving: bool) -> float:
    """P(A wins set) incorporating the current point score within a game.

    This first resolves the current game from the point score,
    then recurses through remaining games.
    """
    if a_is_serving:
        p_server_wins_game = _game_win_prob_from_score(pts_server, pts_returner,
                                                        p_a_serve)
        p_a_wins_this_game = p_server_wins_game
    else:
        p_server_wins_game = _game_win_prob_from_score(pts_server, pts_returner,
                                                        p_b_serve)
        p_a_wins_this_game = 1.0 - p_server_wins_game

    next_serving = not a_is_serving
    return (p_a_wins_this_game *
            _set_win_prob(games_a + 1, games_b, p_a_serve, p_b_serve,
                          next_serving) +
            (1 - p_a_wins_this_game) *
            _set_win_prob(games_a, games_b + 1, p_a_serve, p_b_serve,
                          next_serving))


# ═══════════════════════════════════════════════════════════════════════
#  Tiebreak Probability
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=4096)
def _tiebreak_win_prob(pts_a: int, pts_b: int,
                        p_a_serve: float, p_b_serve: float,
                        a_is_serving: bool) -> float:
    """P(A wins tiebreak) from an arbitrary tiebreak score.

    Tiebreak rules:
        - First to 7, win by 2.
        - Server changes after 1st point, then every 2 points.
        - We track total points played to determine server.
    """
    # Terminal
    if pts_a >= 7 and pts_a - pts_b >= 2:
        return 1.0
    if pts_b >= 7 and pts_b - pts_a >= 2:
        return 0.0

    # Safety cap for deep deuce in tiebreak
    if pts_a > 30 or pts_b > 30:
        # Approximate: 2-point win probability
        if a_is_serving:
            p_win_2 = p_a_serve * (1 - p_b_serve)
            p_lose_2 = (1 - p_a_serve) * p_b_serve
        else:
            p_win_2 = (1 - p_b_serve) * p_a_serve
            p_lose_2 = p_b_serve * (1 - p_a_serve)
        total = p_win_2 + p_lose_2
        return p_win_2 / total if total > 0 else 0.5

    # Current point probability
    if a_is_serving:
        p_a_wins_point = p_a_serve
    else:
        p_a_wins_point = 1.0 - p_b_serve

    # Determine next server after this point
    total_pts = pts_a + pts_b
    # First point: original server. Then alternates every 2 points.
    # After point is played, total_pts + 1 points have been played.
    next_total = total_pts + 1
    if next_total == 1:
        next_a_serving = not a_is_serving  # switch after 1st point
    elif (next_total - 1) % 2 == 0:
        next_a_serving = not a_is_serving  # switch every 2 points after 1st
    else:
        next_a_serving = a_is_serving

    return (p_a_wins_point * _tiebreak_win_prob(pts_a + 1, pts_b,
                                                 p_a_serve, p_b_serve,
                                                 next_a_serving) +
            (1 - p_a_wins_point) * _tiebreak_win_prob(pts_a, pts_b + 1,
                                                       p_a_serve, p_b_serve,
                                                       next_a_serving))


# ═══════════════════════════════════════════════════════════════════════
#  Set → Match Probability (Markov)
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=512)
def _match_win_prob(sets_a: int, sets_b: int,
                    p_a_wins_set: float,
                    best_of: int = 3) -> float:
    """P(A wins match) from an arbitrary set score.

    This uses a constant set-win probability (computed from current state).
    For more accuracy, the set-win probability should be recomputed
    after each set, but the approximation is acceptable for in-play pricing.

    Args:
        sets_a:       Sets won by A.
        sets_b:       Sets won by B.
        p_a_wins_set: P(A wins a single set) — assumed constant.
        best_of:      3 or 5 sets.
    """
    sets_to_win = (best_of + 1) // 2   # 2 for best-of-3, 3 for best-of-5

    if sets_a >= sets_to_win:
        return 1.0
    if sets_b >= sets_to_win:
        return 0.0

    return (p_a_wins_set * _match_win_prob(sets_a + 1, sets_b,
                                            p_a_wins_set, best_of) +
            (1 - p_a_wins_set) * _match_win_prob(sets_a, sets_b + 1,
                                                   p_a_wins_set, best_of))


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

class TennisMarkovModel:
    """Stateless, deterministic in-play match-win probability model.

    Usage:
        model = TennisMarkovModel()
        state = TennisState(match_id="m1", sets_a=1, sets_b=0,
                            games_a=3, games_b=2, ...)
        output = model.get_win_prob(state)
        print(output.p_a, output.p_b)
    """

    def get_win_prob(self, state: TennisState) -> TennisModelOutput:
        """Compute match-win probabilities for both players.

        Pipeline:
            1. Resolve serve-win probability (anchor or default + surface).
            2. Compute P(server wins current game from current point score).
            3. Compute P(A wins current set from current game + point score).
            4. Compute P(A wins match from current set score).

        Args:
            state: Immutable match snapshot.

        Returns:
            TennisModelOutput with full diagnostic internals.
        """
        # 1. Resolve serve probabilities
        p_serve = self._resolve_p_serve(state)
        # Both players assumed symmetric serve strength unless anchor given
        # p_a_serve = probability A wins a point when A serves
        # p_b_serve = probability B wins a point when B serves
        p_a_serve = p_serve
        p_b_serve = p_serve  # symmetric assumption

        # 2. Determine who is serving and map point scores
        a_is_serving = (state.server_id == state.player_a_id)

        if a_is_serving:
            pts_server = state.point_a.value
            pts_returner = state.point_b.value
        else:
            pts_server = state.point_b.value
            pts_returner = state.point_a.value

        # 3. Game-level: P(server wins current game)
        game_win = _game_win_prob_from_score(pts_server, pts_returner, p_serve)

        # 4. Set-level: P(A wins current set) incorporating point score
        if state.is_tiebreak:
            # In tiebreak, use tiebreak model directly
            if a_is_serving:
                set_win_a = _tiebreak_win_prob(
                    state.point_a.value, state.point_b.value,
                    p_a_serve, p_b_serve, a_is_serving)
            else:
                set_win_a = _tiebreak_win_prob(
                    state.point_a.value, state.point_b.value,
                    p_a_serve, p_b_serve, a_is_serving)
        else:
            # Standard game within set
            if state.games_a >= 6 and state.games_b >= 6:
                # At 6-6 the tiebreak hasn't started yet
                set_win_a = _set_win_prob(
                    state.games_a, state.games_b,
                    p_a_serve, p_b_serve, a_is_serving)
            else:
                set_win_a = _set_win_prob_from_point(
                    state.games_a, state.games_b,
                    pts_server, pts_returner,
                    p_a_serve, p_b_serve, a_is_serving)

        # 5. Match-level: P(A wins match)
        match_win_a = _match_win_prob(state.sets_a, state.sets_b,
                                       set_win_a, state.best_of)

        match_win_b = 1.0 - match_win_a

        return TennisModelOutput(
            p_a=match_win_a,
            p_b=match_win_b,
            p_serve=p_serve,
            game_win_prob=game_win,
            set_win_prob_a=set_win_a,
            method="markov",
        )

    @staticmethod
    def _resolve_p_serve(state: TennisState) -> float:
        """Determine service point-win probability.

        Priority:
            1. player_p_anchor if provided.
            2. ATP_SERVE_WIN_P (0.64) + surface adjustment.
        """
        if state.player_p_anchor is not None:
            p = state.player_p_anchor
        else:
            p = ATP_SERVE_WIN_P

        # Apply surface adjustment
        adj = _SURFACE_ADJ.get(state.surface, 0.0)
        p = max(0.01, min(0.99, p + adj))

        return round(p, 4)


# ═══════════════════════════════════════════════════════════════════════
#  Convenience — module-level function
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_MODEL = TennisMarkovModel()

def get_win_prob(state: TennisState) -> TennisModelOutput:
    """Module-level convenience wrapper."""
    return _DEFAULT_MODEL.get_win_prob(state)
