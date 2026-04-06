#!/usr/bin/env python3
"""
Failure Mode Simulation Tests — v2.0

Validates ALL non-bypassable safety guarantees in sports/guards.py:

    1. BLOCK_ZERO_EDGE       — edge ≤ 0 or None → blocked
    2. BLOCK_INSANE_EDGE     — |edge| ≥ 1.0 → blocked
    3. BLOCK_INVALID_PRICE   — price outside [0.01, 0.99] → blocked
    4. BLOCK_STALE_DATA      — book age > 30s → blocked
    5. BLOCK_EMPTY_BOOK      — bid=0 ask=0 → blocked
    6. BLOCK_EDGE_DRIFT      — signal vs execution drift > 0.05 → blocked
    7. BLOCK_CIRCUIT_BREAKER  — 5 consecutive losses → blocked
    8. Edge drift < 0.05 → allowed
    9. Valid trade → passes all checks
   10. Edge quality tracker logs distribution

Also tests:
   11. Circuit breaker: rolling R < -2.0
   12. Circuit breaker: reset works
   13. Negative edge (SELL direction) with valid magnitude → allowed
"""
import os
import sys
import logging

# Setup logging so we can see guard output
logging.basicConfig(level=logging.INFO, format="  %(name)-18s %(message)s")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("  FAILURE MODE SIMULATION v2.0 — GUARD TESTS")
    print("=" * 60)

    from sports.guards import (
        validate_trade_execution,
        circuit_breaker,
        TradingCircuitBreaker,
        EdgeQualityTracker,
    )

    passed = 0
    total = 0

    def check(name, result, expected_pass, expected_reason=""):
        nonlocal passed, total
        total += 1
        can_exec, reason = result
        ok = can_exec == expected_pass
        if expected_reason and not ok:
            ok = False
        elif expected_reason and reason != expected_reason:
            ok = False
        status = "✓" if ok else "✗ FAIL"
        if ok:
            passed += 1
        reason_str = f" ({reason})" if reason else ""
        print(f"  [{total:2d}] {name:40s} → {'PASS' if can_exec else 'BLOCK':5s}{reason_str:30s} {status}")

    # ──────────────────────────────────────────────────────────
    # Reset circuit breaker for clean testing
    # ──────────────────────────────────────────────────────────
    circuit_breaker.reset()
    circuit_breaker._r_values.clear()
    circuit_breaker._recent_outcomes.clear()
    circuit_breaker._signal_results.clear()
    circuit_breaker._disabled = False

    # ──────────────────────────────────────────────────────────
    # 1. Zero edge → blocked
    # ──────────────────────────────────────────────────────────
    print("\n[1] ZERO EDGE...")
    check("edge=0.0", validate_trade_execution(
        edge=0.0, price=0.50, sport="test", context="test"
    ), False, "BLOCK_ZERO_EDGE")

    check("edge=0.0005 (below threshold)", validate_trade_execution(
        edge=0.0005, price=0.50, sport="test", context="test"
    ), False, "BLOCK_ZERO_EDGE")

    check("edge=None", validate_trade_execution(
        edge=None, price=0.50, sport="test", context="test"
    ), False, "BLOCK_ZERO_EDGE")

    # ──────────────────────────────────────────────────────────
    # 2. Insane edge → blocked
    # ──────────────────────────────────────────────────────────
    print("\n[2] INSANE EDGE...")
    check("edge=1.5", validate_trade_execution(
        edge=1.5, price=0.50, sport="test", context="test"
    ), False, "BLOCK_INSANE_EDGE")

    check("edge=-1.2 (sell direction)", validate_trade_execution(
        edge=-1.2, price=0.50, sport="test", context="test"
    ), False, "BLOCK_INSANE_EDGE")

    # ──────────────────────────────────────────────────────────
    # 3. Invalid price → blocked
    # ──────────────────────────────────────────────────────────
    print("\n[3] INVALID PRICE...")
    check("price=0.00", validate_trade_execution(
        edge=0.05, price=0.00, sport="test", context="test"
    ), False, "BLOCK_INVALID_PRICE")

    check("price=1.00", validate_trade_execution(
        edge=0.05, price=1.00, sport="test", context="test"
    ), False, "BLOCK_INVALID_PRICE")

    check("price=-0.10", validate_trade_execution(
        edge=0.05, price=-0.10, sport="test", context="test"
    ), False, "BLOCK_INVALID_PRICE")

    check("price=None", validate_trade_execution(
        edge=0.05, price=None, sport="test", context="test"
    ), False, "BLOCK_INVALID_PRICE")

    # ──────────────────────────────────────────────────────────
    # 4. Stale data → blocked
    # ──────────────────────────────────────────────────────────
    print("\n[4] STALE DATA...")
    check("book_age=60s", validate_trade_execution(
        edge=0.05, price=0.50, sport="test", context="test",
        book_age=60.0,
    ), False, "BLOCK_STALE_DATA")

    check("book_age=31s", validate_trade_execution(
        edge=0.05, price=0.50, sport="test", context="test",
        book_age=31.0,
    ), False, "BLOCK_STALE_DATA")

    check("book_age=29s (ok)", validate_trade_execution(
        edge=0.05, price=0.50, sport="test", context="test",
        book_age=29.0,
    ), True)

    # ──────────────────────────────────────────────────────────
    # 5. Empty book → blocked
    # ──────────────────────────────────────────────────────────
    print("\n[5] EMPTY BOOK...")
    check("bid=0 ask=0", validate_trade_execution(
        edge=0.05, price=0.50, sport="test", context="test",
        book_bid=0.0, book_ask=0.0,
    ), False, "BLOCK_EMPTY_BOOK")

    check("bid=0.01 ask=0.0 (partial ok)", validate_trade_execution(
        edge=0.05, price=0.50, sport="test", context="test",
        book_bid=0.01, book_ask=0.0,
    ), True)  # partial book is acceptable — at least one side exists

    # ──────────────────────────────────────────────────────────
    # 6. Edge drift → blocked
    # ──────────────────────────────────────────────────────────
    print("\n[6] EDGE DRIFT...")
    check("drift=0.08 (blocked)", validate_trade_execution(
        edge=0.05, price=0.50, sport="test", context="test",
        signal_edge=0.13,
    ), False, "BLOCK_EDGE_DRIFT")

    check("drift=0.03 (allowed)", validate_trade_execution(
        edge=0.10, price=0.50, sport="test", context="test",
        signal_edge=0.13,
    ), True)

    check("drift=0.05 exactly (allowed)", validate_trade_execution(
        edge=0.08, price=0.50, sport="test", context="test",
        signal_edge=0.13,
    ), True)

    # Edge drift with sell direction (negative edge)
    check("sell drift=0.08 (blocked)", validate_trade_execution(
        edge=-0.05, price=0.50, sport="test", context="test",
        signal_edge=-0.13,
    ), False, "BLOCK_EDGE_DRIFT")

    # ──────────────────────────────────────────────────────────
    # 7. Circuit breaker — loss streak
    # ──────────────────────────────────────────────────────────
    print("\n[7] CIRCUIT BREAKER — LOSS STREAK...")
    # Use a fresh breaker for isolated testing
    cb = TradingCircuitBreaker(loss_streak_limit=5, rolling_r_window=10, rolling_r_floor=-2.0)
    
    # 4 losses → not tripped yet
    for i in range(4):
        cb.record_trade_outcome(-0.10)
    assert not cb.is_disabled, "4 losses should not trip"
    print(f"  4 consecutive losses:        not tripped ✓")
    
    # 5th loss → tripped
    cb.record_trade_outcome(-0.10)
    assert cb.is_disabled, "5 losses should trip"
    assert "LOSS_STREAK_5" in cb.disable_reason
    print(f"  5 consecutive losses:        tripped ({cb.disable_reason}) ✓")

    # Reset
    cb.reset()
    assert not cb.is_disabled, "Reset should clear"
    print(f"  Reset:                       cleared ✓")

    # ──────────────────────────────────────────────────────────
    # 8. Circuit breaker — rolling R
    # ──────────────────────────────────────────────────────────
    print("\n[8] CIRCUIT BREAKER — ROLLING R...")
    cb2 = TradingCircuitBreaker(loss_streak_limit=5, rolling_r_window=10, rolling_r_floor=-2.0)
    
    # Mix of wins and losses with net R < -2.0
    outcomes = [-0.8, +0.2, -0.5, -0.3, +0.1, -0.4, -0.2, -0.3, +0.1, -0.15]
    total_r = sum(outcomes)
    for r in outcomes:
        cb2.record_trade_outcome(r)
    
    if total_r < -2.0:
        assert cb2.is_disabled, f"Rolling R {total_r:.2f} < -2.0 should trip"
        print(f"  Rolling R={total_r:.2f}:         tripped ({cb2.disable_reason}) ✓")
    else:
        assert not cb2.is_disabled, f"Rolling R {total_r:.2f} >= -2.0 shouldn't trip"
        print(f"  Rolling R={total_r:.2f}:         not tripped (above floor) ✓")

    # ──────────────────────────────────────────────────────────
    # 9. Circuit breaker integration with guard
    # ──────────────────────────────────────────────────────────
    print("\n[9] CIRCUIT BREAKER + GUARD...")
    # Reset global breaker and force-trip it
    circuit_breaker.reset()
    circuit_breaker._r_values.clear()
    circuit_breaker._recent_outcomes.clear()
    for _ in range(5):
        circuit_breaker.record_trade_outcome(-0.50)
    
    check("valid trade + tripped breaker", validate_trade_execution(
        edge=0.10, price=0.50, sport="test", context="test",
    ), False)  # should be blocked by circuit breaker

    # Reset for remaining tests
    circuit_breaker.reset()
    circuit_breaker._r_values.clear()
    circuit_breaker._recent_outcomes.clear()
    circuit_breaker._disabled = False

    # ──────────────────────────────────────────────────────────
    # 10. Valid trade → passes all checks
    # ──────────────────────────────────────────────────────────
    print("\n[10] VALID TRADE (all checks pass)...")
    check("full valid trade", validate_trade_execution(
        edge=0.08, price=0.50, sport="tennis", context="test",
        book_age=5.0, book_bid=0.49, book_ask=0.51, signal_edge=0.09,
    ), True)

    # Negative edge with valid magnitude (SELL direction)
    check("sell direction edge=-0.08", validate_trade_execution(
        edge=-0.08, price=0.50, sport="tennis", context="sell test",
        book_age=5.0, book_bid=0.49, book_ask=0.51,
    ), True)

    # ──────────────────────────────────────────────────────────
    # 11. Edge quality tracker
    # ──────────────────────────────────────────────────────────
    print("\n[11] EDGE QUALITY TRACKER...")
    tracker = EdgeQualityTracker(window=10, log_interval=5)
    for e in [0.05, 0.08, 0.12, 0.03, 0.15]:
        tracker.record(e)
    # Should have logged after 5 entries
    print(f"  Tracked {len(tracker._edges)} edges:     ✓")

    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{total} tests passed")
    print(f"{'=' * 60}")

    if passed == total:
        print("\n  ✅ ALL FAILURE MODE TESTS PASSED — system is safe under failure")
    else:
        print(f"\n  ❌ {total - passed} TESTS FAILED — review before deployment")
        sys.exit(1)


if __name__ == "__main__":
    main()
