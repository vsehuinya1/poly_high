#!/usr/bin/env python3
"""
Tennis execution hardening v2.0 — guard-specific tests.

Tests:
    1. Dead market block (mkt=0.001)
    2. State dedup suppression
    3. Rolling rate limiter (11 signals in 1hr)
    4. Stale auto-disable (5 consecutive stales)
    5. Position loop breaker (open position → skip eval)
    6. TennisHealthStats counters
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("  TENNIS HARDENING v2.0 — GUARD TESTS")
    print("=" * 60)

    from tennis.state import TennisState, TennisModelOutput, PointScore
    from tennis.strategy import InflectionStrategy, TennisSignal
    from tennis.execution import TennisExecutionGuard, ExecutionDecision, TennisHealthStats

    # ──────────────────────────────────────────────────────────
    # 1. Dead Market Block (Fix 1)
    # ──────────────────────────────────────────────────────────
    print("\n[1] DEAD MARKET BLOCK...")

    strategy = InflectionStrategy(
        panic_edge_threshold=0.06,
        reversion_edge_threshold=0.05,
        price_floor=0.05,
    )

    # Reversion scenario: favorite down 0-1
    dead_state = TennisState(
        match_id="dead_mkt_test",
        sets_a=0, sets_b=1,
        player_a_id="fav", player_b_id="dog",
        server_id="fav", receiver_id="dog",
        pregame_favorite_id="fav",
        timestamp=time.time(),
    )

    # Market price below floor → should return None
    sig = strategy.evaluate(dead_state, market_price=0.001)
    assert sig is None, "Dead market (0.001) should be suppressed"
    print("  mkt=0.001 suppressed:       ✓")

    sig2 = strategy.evaluate(dead_state, market_price=0.04)
    assert sig2 is None, "Dead market (0.04 < 0.05 floor) should be suppressed"
    print("  mkt=0.04 < 0.05 floor:      ✓")

    # Call again — should still return None (log-once behavior, no crash)
    sig3 = strategy.evaluate(dead_state, market_price=0.001)
    assert sig3 is None, "Second dead market call should still suppress"
    print("  log-once (no double log):    ✓")

    # Market above floor should fire
    sig4 = strategy.evaluate(dead_state, market_price=0.15)
    assert sig4 is not None, "Price 0.15 >= 0.05 floor should fire SET_MEAN_REVERSION"
    assert sig4.trigger_type == "SET_MEAN_REVERSION"
    print(f"  mkt=0.15 fires normally:     edge={sig4.edge:+.4f} ✓")

    print("  Dead market block PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 2. State Dedup Suppression (Fix 2)
    # ──────────────────────────────────────────────────────────
    print("\n[2] STATE DEDUP...")

    strat2 = InflectionStrategy(
        panic_edge_threshold=0.06,
        reversion_edge_threshold=0.05,
        price_floor=0.05,
    )

    dedup_state = TennisState(
        match_id="dedup_test",
        sets_a=0, sets_b=1,
        player_a_id="fav", player_b_id="dog",
        server_id="fav", receiver_id="dog",
        pregame_favorite_id="fav",
        timestamp=time.time(),
    )

    # First call should fire
    s1 = strat2.evaluate(dedup_state, market_price=0.15, selection_id="token_A")
    assert s1 is not None, "First call should fire"
    print(f"  First call fires:            edge={s1.edge:+.4f} ✓")

    # Same state + same selection → suppress
    s2 = strat2.evaluate(dedup_state, market_price=0.15, selection_id="token_A")
    assert s2 is None, "Same state should be suppressed by dedup"
    print("  Same state suppressed:       ✓")

    # Different selection_id → should fire (cross-runner safety)
    s3 = strat2.evaluate(dedup_state, market_price=0.15, selection_id="token_B")
    assert s3 is not None, "Different selection_id should fire"
    print("  Different selection fires:    ✓")

    # Change score state → should fire again for original selection
    dedup_state_changed = TennisState(
        match_id="dedup_test",
        sets_a=0, sets_b=1,
        games_a=1, games_b=0,  # games changed
        player_a_id="fav", player_b_id="dog",
        server_id="fav", receiver_id="dog",
        pregame_favorite_id="fav",
        timestamp=time.time(),
    )
    s4 = strat2.evaluate(dedup_state_changed, market_price=0.15, selection_id="token_A")
    assert s4 is not None, "State change should re-enable signal"
    print(f"  State change re-fires:       edge={s4.edge:+.4f} ✓")

    print("  State dedup PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 3. Rolling Rate Limiter (Fix 5)
    # ──────────────────────────────────────────────────────────
    print("\n[3] ROLLING RATE LIMITER...")

    guard = TennisExecutionGuard(
        price_cap=0.85, staleness_s=3.0, cooldown_s=0.0,
        max_signals_per_hour=10, stale_disable_count=5,
    )

    dummy_model = TennisModelOutput(
        p_a=0.55, p_b=0.45, p_serve=0.64,
        game_win_prob=0.84, set_win_prob_a=0.52,
    )
    now = time.time()

    # Send 10 valid signals (should all pass)
    passed = 0
    for i in range(10):
        sig = TennisSignal(
            timestamp=now, match_id="rate_test",
            trigger_type="SET_MEAN_REVERSION",
            edge=0.10, fair_price=0.60, market_price=0.50,
            state_snapshot=TennisState(
                match_id="rate_test", timestamp=now,
                player_a_id="a", player_b_id="b",
            ),
            model_output=dummy_model,
        )
        d = guard.can_execute(sig, sig.state_snapshot)
        if d.can_execute:
            passed += 1
    assert passed == 10, f"First 10 should pass, got {passed}"
    print(f"  Signals 1-10 pass:           {passed}/10 ✓")

    # 11th signal → rate limited
    sig11 = TennisSignal(
        timestamp=now, match_id="rate_test",
        trigger_type="SET_MEAN_REVERSION",
        edge=0.10, fair_price=0.60, market_price=0.50,
        state_snapshot=TennisState(
            match_id="rate_test", timestamp=now,
            player_a_id="a", player_b_id="b",
        ),
        model_output=dummy_model,
    )
    d11 = guard.can_execute(sig11, sig11.state_snapshot)
    assert not d11.can_execute and d11.reason == "BLOCK_RATE_LIMIT"
    print(f"  Signal 11 blocked:           {d11} ✓")

    assert guard.stats.rate_limited >= 1, "Rate limit counter should increment"
    print(f"  Rate limit counter:          {guard.stats.rate_limited} ✓")

    print("  Rolling rate limiter PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 4. Stale Auto-Disable (Fix 4)
    # ──────────────────────────────────────────────────────────
    print("\n[4] STALE AUTO-DISABLE...")

    guard2 = TennisExecutionGuard(
        price_cap=0.85, staleness_s=3.0, cooldown_s=0.0,
        max_signals_per_hour=100,
        stale_disable_count=5, stale_disable_s=300.0,
    )

    stale_time = time.time() - 10.0  # 10s old state

    # Fire 5 consecutive stale signals
    for i in range(5):
        sig = TennisSignal(
            timestamp=now, match_id="stale_test",
            trigger_type="SET_MEAN_REVERSION",
            edge=0.10, fair_price=0.60, market_price=0.50,
            state_snapshot=TennisState(
                match_id="stale_test", timestamp=stale_time,
                player_a_id="a", player_b_id="b",
            ),
            model_output=dummy_model,
        )
        d = guard2.can_execute(sig, sig.state_snapshot)
        assert not d.can_execute and d.reason == "BLOCK_STALE"

    print(f"  5 consecutive stales:        count={guard2._stale_counts.get('stale_test', 0)} ✓")

    # Check that match is now disabled
    assert "stale_test" in guard2._disabled_until
    assert guard2._disabled_until["stale_test"] > time.time()
    print(f"  Match auto-disabled:         until +{guard2._disabled_until['stale_test'] - time.time():.0f}s ✓")

    # should_evaluate should return False for disabled match
    can = guard2.should_evaluate("stale_test")
    assert can is False, "Disabled match should fail should_evaluate"
    print("  should_evaluate→False:       ✓")

    assert guard2.stats.stale_block >= 5, "Stale block counter"
    assert guard2.stats.max_staleness_ms > 0, "Max staleness tracked"
    print(f"  Stats: stale_block={guard2.stats.stale_block} max_ms={guard2.stats.max_staleness_ms:.0f} ✓")

    # Simulate recovery: fresh state should reset stale counter
    guard3 = TennisExecutionGuard(
        price_cap=0.85, staleness_s=3.0, cooldown_s=0.0,
        max_signals_per_hour=100, stale_disable_count=5,
    )
    # 3 stale events
    for i in range(3):
        sig = TennisSignal(
            timestamp=now, match_id="recovery_test",
            trigger_type="SET_MEAN_REVERSION",
            edge=0.10, fair_price=0.60, market_price=0.50,
            state_snapshot=TennisState(
                match_id="recovery_test", timestamp=stale_time,
                player_a_id="a", player_b_id="b",
            ),
            model_output=dummy_model,
        )
        guard3.can_execute(sig, sig.state_snapshot)

    assert guard3._stale_counts.get("recovery_test") == 3

    # Now a fresh signal — should pass and reset counter
    fresh_sig = TennisSignal(
        timestamp=now, match_id="recovery_test",
        trigger_type="SET_MEAN_REVERSION",
        edge=0.10, fair_price=0.60, market_price=0.50,
        state_snapshot=TennisState(
            match_id="recovery_test", timestamp=time.time(),
            player_a_id="a", player_b_id="b",
        ),
        model_output=dummy_model,
    )
    d_fresh = guard3.can_execute(fresh_sig, fresh_sig.state_snapshot)
    assert d_fresh.can_execute, "Fresh signal should pass after stales"
    assert guard3._stale_counts.get("recovery_test") == 0, "Counter reset on fresh"
    print("  Stale counter resets:        ✓")

    print("  Stale auto-disable PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 5. Position Loop Breaker (Fix 3)
    # ──────────────────────────────────────────────────────────
    print("\n[5] POSITION LOOP BREAKER...")

    guard4 = TennisExecutionGuard(
        price_cap=0.85, staleness_s=3.0, cooldown_s=0.0,
        max_signals_per_hour=100,
    )

    # Record a position entry
    guard4.record_entry("pos_test", state_key="key1", edge=0.10)
    ms = guard4.get_match_state("pos_test")
    assert ms.has_open_position is True
    assert ms.entry_state_key == "key1"
    assert ms.entry_edge_sign == 1.0

    # should_evaluate with SAME state key → False (skip)
    can1 = guard4.should_evaluate("pos_test", state_key="key1", edge=0.10)
    assert can1 is False, "Same state while position open → skip"
    print("  Same state + open pos:       skipped ✓")

    # should_evaluate with different state key but SAME edge sign → False
    can2 = guard4.should_evaluate("pos_test", state_key="key2", edge=0.10)
    assert can2 is False, "State changed but edge same sign → skip"
    print("  State changed, same edge:    skipped ✓")

    # should_evaluate with different state key AND flipped edge → True
    can3 = guard4.should_evaluate("pos_test", state_key="key2", edge=-0.10)
    assert can3 is True, "State changed + edge flipped → allow"
    print("  State changed + edge flip:   allowed ✓")

    # No position → should always evaluate
    can4 = guard4.should_evaluate("no_pos_test")
    assert can4 is True, "No position → always evaluate"
    print("  No position → evaluate:      ✓")

    assert guard4.stats.position_block >= 2, "Position block counter"
    print(f"  Position block count:        {guard4.stats.position_block} ✓")

    print("  Position loop breaker PASSED ✓")

    # ──────────────────────────────────────────────────────────
    # 6. Health Stats Integration (Fix 7)
    # ──────────────────────────────────────────────────────────
    print("\n[6] HEALTH STATS...")

    stats = TennisHealthStats()
    stats.signals_fired = 42
    stats.dead_market = 100
    stats.stale_block = 10
    stats.trades_executed = 3
    stats.max_staleness_ms = 8500

    d = stats.as_dict()
    assert d["signals_fired"] == 42
    assert d["dead_market"] == 100
    assert d["max_staleness_ms"] == 8500
    print(f"  as_dict() keys:              {len(d)} ✓")

    # log_summary should not crash
    stats.log_summary()
    print("  log_summary() runs:          ✓")

    print("  Health stats PASSED ✓")

    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ALL HARDENING TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
