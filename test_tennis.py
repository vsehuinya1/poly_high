#!/usr/bin/env python3
"""
Tennis module integration test.

Validates:
    1. All imports resolve.
    2. Markov model produces correct probabilities for known states.
    3. Strategy B triggers fire on synthetic scenarios.
    4. Execution guards block correctly.
    5. CSV logger writes proper format.
    6. State transitions work via update_from_point.
"""
import os
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("  TENNIS MODULE INTEGRATION TEST")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # 1. Import Check
    # ──────────────────────────────────────────────────────────
    print("\n[1] IMPORTS...", end=" ")
    from tennis.state import (
        TennisState, TennisPointEvent, TennisModelOutput,
        PointScore, update_from_point,
        compute_break_point_flag, compute_momentum_delta,
    )
    from tennis.model import (
        TennisMarkovModel, get_win_prob,
        _game_win_prob_from_start, _game_win_prob_from_score,
        _set_win_prob, _tiebreak_win_prob, _match_win_prob,
    )
    from tennis.strategy import InflectionStrategy, TennisSignal
    from tennis.execution import TennisExecutionGuard, ExecutionDecision
    from tennis.logger import TennisCSVLogger
    from tennis.feeds import TennisDataFeed, FlashscoreFeed
    print("OK ✓")

    # ──────────────────────────────────────────────────────────
    # 2. Markov Model — Known Boundary Cases
    # ──────────────────────────────────────────────────────────
    print("\n[2] MARKOV MODEL...")

    # 2a. Game win from 0-0 with p=0.64
    g = _game_win_prob_from_start(0.64)
    print(f"  Game win (0-0, p=0.64):     {g:.4f}")
    assert 0.80 < g < 0.88, f"Expected ~0.84, got {g:.4f}"

    # 2b. Game win from 0-0 with p=0.50 → must be exactly 0.50
    g50 = _game_win_prob_from_start(0.50)
    print(f"  Game win (0-0, p=0.50):     {g50:.4f}")
    assert abs(g50 - 0.50) < 0.001, f"Expected 0.500, got {g50:.4f}"

    # 2c. Deuce probability: p²/(p²+q²)
    deuce_p = _game_win_prob_from_score(3, 3, 0.64)
    expected_deuce = 0.64**2 / (0.64**2 + 0.36**2)
    print(f"  Deuce win (p=0.64):         {deuce_p:.4f} (expected {expected_deuce:.4f})")
    assert abs(deuce_p - expected_deuce) < 0.001, f"Deuce mismatch"

    # 2d. Game win from 40-0 → almost certain
    g40_0 = _game_win_prob_from_score(3, 0, 0.64)
    print(f"  Game win (40-0, p=0.64):    {g40_0:.4f}")
    assert g40_0 > 0.95, f"40-0 should be > 0.95"

    # 2e. Game win from 0-40 → very low
    g0_40 = _game_win_prob_from_score(0, 3, 0.64)
    print(f"  Game win (0-40, p=0.64):    {g0_40:.4f}")
    assert g0_40 < 0.20, f"0-40 should be < 0.20"

    # 2f. Set win from 0-0 with equal serve
    s = _set_win_prob(0, 0, 0.64, 0.64, True)
    print(f"  Set win (0-0, both p=0.64): {s:.4f}")
    assert 0.45 < s < 0.55, f"Symmetric should be ~0.50"

    # 2g. Match win from 0-0 sets with equal serve
    m = _match_win_prob(0, 0, 0.50, 3)
    print(f"  Match win (0-0 sets, 50%):  {m:.4f}")
    assert abs(m - 0.50) < 0.001, f"50% set win should give 50% match"

    # 2h. Full model call from match start
    model = TennisMarkovModel()
    state_start = TennisState(
        match_id="test_1",
        player_a_id="player_a", player_b_id="player_b",
        server_id="player_a", receiver_id="player_b",
        pregame_favorite_id="player_a",
        timestamp=time.time(),
    )
    out = model.get_win_prob(state_start)
    print(f"  Full model (match start):   A={out.p_a:.4f} B={out.p_b:.4f}")
    assert abs(out.p_a + out.p_b - 1.0) < 0.001, "Probabilities must sum to 1"
    assert 0.45 < out.p_a < 0.55, "Equal players at start should be ~50/50"

    # 2i. A is up 1-0 in sets → should have > 50%
    state_up = TennisState(
        match_id="test_2", sets_a=1, sets_b=0,
        player_a_id="player_a", player_b_id="player_b",
        server_id="player_a", receiver_id="player_b",
        pregame_favorite_id="player_a",
        timestamp=time.time(),
    )
    out_up = model.get_win_prob(state_up)
    print(f"  A leads 1-0 sets:           A={out_up.p_a:.4f} B={out_up.p_b:.4f}")
    assert out_up.p_a > 0.60, "Leading 1-0 should give >60%"

    print("  Model tests PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 3. Break Point Detection
    # ──────────────────────────────────────────────────────────
    print("\n[3] BREAK POINT DETECTION...")

    # 3a. Server at 30, returner at 40 → break point
    bp_state = TennisState(
        match_id="bp_test",
        player_a_id="sv", player_b_id="rt",
        server_id="sv", receiver_id="rt",
        point_a=PointScore.P30, point_b=PointScore.P40,
        timestamp=time.time(),
    )
    assert bp_state.is_break_point, "30-40 should be break point"
    print("  30-40 (server-returner):    BP=True ✓")

    # 3b. Both at 40 (deuce) → NOT break point
    deuce_state = TennisState(
        match_id="deuce_test",
        player_a_id="sv", player_b_id="rt",
        server_id="sv", receiver_id="rt",
        point_a=PointScore.P40, point_b=PointScore.P40,
        timestamp=time.time(),
    )
    assert not deuce_state.is_break_point, "Deuce is not break point"
    print("  40-40 (deuce):              BP=False ✓")

    # 3c. Returner has AD → break point
    ad_state = TennisState(
        match_id="ad_test",
        player_a_id="sv", player_b_id="rt",
        server_id="sv", receiver_id="rt",
        point_a=PointScore.P40, point_b=PointScore.AD,
        timestamp=time.time(),
    )
    assert ad_state.is_break_point, "40-AD should be break point"
    print("  40-AD (returner has AD):    BP=True ✓")

    # 3d. Tiebreak → never break point
    tb_state = TennisState(
        match_id="tb_test",
        player_a_id="sv", player_b_id="rt",
        server_id="sv", receiver_id="rt",
        point_a=PointScore.P30, point_b=PointScore.P40,
        is_tiebreak=True,
        timestamp=time.time(),
    )
    assert not tb_state.is_break_point, "Tiebreak should never flag as BP"
    print("  30-40 in tiebreak:          BP=False ✓")

    print("  Break point tests PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 4. Strategy B Triggers
    # ──────────────────────────────────────────────────────────
    print("\n[4] STRATEGY B TRIGGERS...")

    strategy = InflectionStrategy(panic_edge_threshold=0.06,
                                   reversion_edge_threshold=0.05)

    # 4a. Panic Discount — break point, favorite serving, underpriced
    #     Model gives p_a ≈ 0.35 at 30-40 bp, so market must be < 0.29
    panic_state = TennisState(
        match_id="panic_test",
        player_a_id="fav", player_b_id="dog",
        server_id="fav", receiver_id="dog",
        pregame_favorite_id="fav",
        point_a=PointScore.P30, point_b=PointScore.P40,  # break point
        timestamp=time.time(),
    )
    # Set market price well below fair (~0.35) to create 0.10+ edge
    sig = strategy.evaluate(panic_state, market_price=0.25)
    assert sig is not None, "Panic Discount should fire at market=0.25 (fair~0.35)"
    assert sig.trigger_type == "PANIC_DISCOUNT"
    print(f"  Panic Discount fired:       edge={sig.edge:+.4f} ✓")

    # 4b. Set Mean Reversion — favorite down 0-1
    #     Model gives p_a ≈ 0.38 when down 0-1 in sets, so market must be < 0.33
    reversion_state = TennisState(
        match_id="reversion_test",
        sets_a=0, sets_b=1,  # favorite (A) is down
        player_a_id="fav", player_b_id="dog",
        server_id="fav", receiver_id="dog",
        pregame_favorite_id="fav",
        timestamp=time.time(),
    )
    sig2 = strategy.evaluate(reversion_state, market_price=0.15)
    assert sig2 is not None, "Set Mean Reversion should fire at market=0.15 (fair~0.25)"
    assert sig2.trigger_type == "SET_MEAN_REVERSION"
    print(f"  Set Mean Reversion fired:   edge={sig2.edge:+.4f} ✓")

    # 4c. No trigger when not break point and not down 0-1
    clean_state = TennisState(
        match_id="clean_test",
        player_a_id="fav", player_b_id="dog",
        server_id="fav", receiver_id="dog",
        pregame_favorite_id="fav",
        point_a=PointScore.P15, point_b=PointScore.LOVE,
        timestamp=time.time(),
    )
    sig3 = strategy.evaluate(clean_state, market_price=0.50)
    assert sig3 is None, "No trigger should fire on clean state"
    print("  No false trigger on clean state ✓")

    print("  Strategy B tests PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 5. Execution Guard
    # ──────────────────────────────────────────────────────────
    print("\n[5] EXECUTION GUARD...")

    guard = TennisExecutionGuard(price_cap=0.85, staleness_s=3.0, cooldown_s=120.0)

    # Build a dummy signal for testing
    dummy_model_out = TennisModelOutput(
        p_a=0.55, p_b=0.45, p_serve=0.64,
        game_win_prob=0.84, set_win_prob_a=0.52,
    )
    now = time.time()

    # 5a. Price cap block
    high_price_signal = TennisSignal(
        timestamp=now, match_id="exec_test", trigger_type="PANIC_DISCOUNT",
        edge=0.10, fair_price=0.60, market_price=0.90,
        state_snapshot=TennisState(match_id="exec_test", timestamp=now,
                                    player_a_id="a", player_b_id="b"),
        model_output=dummy_model_out,
    )
    d = guard.can_execute(high_price_signal,
                           high_price_signal.state_snapshot)
    assert not d.can_execute and d.reason == "BLOCK_PRICE_CAP"
    print(f"  Price cap block (0.90):     {d} ✓")

    # 5b. Tiebreak block
    tb_signal = TennisSignal(
        timestamp=now, match_id="exec_tb", trigger_type="PANIC_DISCOUNT",
        edge=0.10, fair_price=0.60, market_price=0.50,
        state_snapshot=TennisState(match_id="exec_tb", is_tiebreak=True,
                                    timestamp=now, player_a_id="a", player_b_id="b"),
        model_output=dummy_model_out,
    )
    d2 = guard.can_execute(tb_signal, tb_signal.state_snapshot)
    assert not d2.can_execute and d2.reason == "BLOCK_TIEBREAK"
    print(f"  Tiebreak block:             {d2} ✓")

    # 5c. Staleness block
    stale_signal = TennisSignal(
        timestamp=now, match_id="exec_stale", trigger_type="SET_MEAN_REVERSION",
        edge=0.10, fair_price=0.60, market_price=0.50,
        state_snapshot=TennisState(match_id="exec_stale",
                                    timestamp=now - 10.0,  # 10s old
                                    player_a_id="a", player_b_id="b"),
        model_output=dummy_model_out,
    )
    d3 = guard.can_execute(stale_signal, stale_signal.state_snapshot)
    assert not d3.can_execute and d3.reason == "BLOCK_STALE"
    print(f"  Staleness block (10s old):  {d3} ✓")

    # 5d. Valid signal passes
    valid_signal = TennisSignal(
        timestamp=now, match_id="exec_valid", trigger_type="PANIC_DISCOUNT",
        edge=0.10, fair_price=0.60, market_price=0.50,
        state_snapshot=TennisState(match_id="exec_valid", timestamp=now,
                                    player_a_id="a", player_b_id="b"),
        model_output=dummy_model_out,
    )
    d4 = guard.can_execute(valid_signal, valid_signal.state_snapshot)
    assert d4.can_execute
    print(f"  Valid signal passes:         {d4} ✓")

    # 5e. Position limit — after entry, second signal blocked
    guard.record_entry("exec_valid")
    d5 = guard.can_execute(valid_signal, valid_signal.state_snapshot)
    assert not d5.can_execute and d5.reason == "BLOCK_POSITION"
    print(f"  Position limit block:       {d5} ✓")

    print("  Execution guard tests PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 6. CSV Logger
    # ──────────────────────────────────────────────────────────
    print("\n[6] CSV LOGGER...")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_logger = TennisCSVLogger(Path(tmpdir))

        # Log the signals from Strategy B tests (guaranteed non-None now)
        csv_logger.log_signal(sig)
        csv_logger.log_trade_entry(sig, market_price_at_bp=0.25)
        csv_logger.log_trade_exit("panic_test", exit_reason="convergence",
                                   r_multiple=1.5)
        csv_logger.log_signal(sig2)
        csv_logger.close()

        # Verify files exist
        csv_files = list(Path(tmpdir).glob("*.csv"))
        print(f"  CSV files created:          {len(csv_files)}")
        for f in csv_files:
            lines = f.read_text().strip().split("\n")
            print(f"    {f.name}: {len(lines)} lines (header + {len(lines)-1} data)")
        assert len(csv_files) >= 1, "At least 1 CSV should be created"

    print("  CSV logger tests PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 7. State Transition (update_from_point)
    # ──────────────────────────────────────────────────────────
    print("\n[7] STATE TRANSITION...")

    initial = TennisState(
        match_id="transition_test",
        player_a_id="alice", player_b_id="bob",
        server_id="alice", receiver_id="bob",
        pregame_favorite_id="alice",
        timestamp=time.time(),
    )

    event = TennisPointEvent(
        match_id="transition_test",
        point_winner_id="alice",
        new_sets_a=0, new_sets_b=0,
        new_games_a=0, new_games_b=0,
        new_point_a="15", new_point_b="0",
        new_server_id="alice",
        timestamp=time.time(),
    )

    next_state = update_from_point(initial, event)
    assert next_state.point_a == PointScore.P15
    assert next_state.point_b == PointScore.LOVE
    assert len(next_state.recent_points) == 1
    assert next_state.recent_points[0] == "alice"
    print(f"  0-0 → 15-0 after alice wins: {next_state.point_a}-{next_state.point_b} ✓")

    # Momentum check
    m_delta = compute_momentum_delta(next_state)
    assert m_delta > 0, "Alice winning should give positive momentum"
    print(f"  Momentum delta:             {m_delta:+.4f} ✓")

    print("  State transition tests PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 8. Tiebreak Model
    # ──────────────────────────────────────────────────────────
    print("\n[8] TIEBREAK MODEL...")

    # Tiebreak from 0-0 with equal serve → ~50%
    from tennis.model import _tiebreak_win_prob
    tb = _tiebreak_win_prob(0, 0, 0.64, 0.64, True)
    print(f"  Tiebreak (0-0, symmetric):  {tb:.4f}")
    assert 0.45 < tb < 0.55, "Symmetric tiebreak should be ~50%"

    # A leading 6-3 in tiebreak → overwhelming favorite
    tb_lead = _tiebreak_win_prob(6, 3, 0.64, 0.64, True)
    print(f"  Tiebreak (6-3, A serving):  {tb_lead:.4f}")
    assert tb_lead > 0.80, "6-3 in tiebreak should be >80%"

    print("  Tiebreak model tests PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 9. Surface Adjustments
    # ──────────────────────────────────────────────────────────
    print("\n[9] SURFACE ADJUSTMENTS...")

    grass_state = TennisState(
        match_id="grass_test", surface="grass",
        player_a_id="a", player_b_id="b",
        server_id="a", receiver_id="b",
        pregame_favorite_id="a",
        timestamp=time.time(),
    )
    clay_state = TennisState(
        match_id="clay_test", surface="clay",
        player_a_id="a", player_b_id="b",
        server_id="a", receiver_id="b",
        pregame_favorite_id="a",
        timestamp=time.time(),
    )
    hard_state = TennisState(
        match_id="hard_test", surface="hard",
        player_a_id="a", player_b_id="b",
        server_id="a", receiver_id="b",
        pregame_favorite_id="a",
        timestamp=time.time(),
    )

    g_out = model.get_win_prob(grass_state)
    c_out = model.get_win_prob(clay_state)
    h_out = model.get_win_prob(hard_state)
    print(f"  p_serve — grass={g_out.p_serve} hard={h_out.p_serve} clay={c_out.p_serve}")
    assert g_out.p_serve > h_out.p_serve > c_out.p_serve, \
        "Grass > Hard > Clay for serve advantage"
    print("  Surface ordering correct ✓")

    print("  Surface adjustment tests PASSED ✓")

    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ALL TENNIS TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
