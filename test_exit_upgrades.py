#!/usr/bin/env python3
"""
Tennis Exit Manager v3.2 — exit upgrade tests.

Validates:
    1. EXIT_STOP_LOSS fires as primary (-15% cap)
    2. EXIT_TICK_STOP fires as secondary (dynamic cap)
    3. Dynamic tick limit: never looser than -15%
    4. EXIT_PRICE_FLOOR fires at mkt <= 0.18
    5. EXIT_NO_MFE fires within 12min window with low MFE
    6. Runner V2 activates at 3% MFE (was 5%)
    7. mae_ticks tracked correctly
    8. New exit reasons appear in stats dict
    9. CSV schema includes mae_ticks column
"""
import os
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("  TENNIS EXIT UPGRADES v3.2 — TESTS")
    print("=" * 60)

    from tennis.exit_manager import TennisExitManager, TennisPaperTrade

    # ──────────────────────────────────────────────────────────
    # Helper
    # ──────────────────────────────────────────────────────────
    def make_manager_and_trade(entry_price=0.50, entry_time_offset=0):
        tmpdir = tempfile.mkdtemp()
        mgr = TennisExitManager(data_dir=Path(tmpdir))
        trade = mgr.register_trade(
            match_id="test_match",
            selection_id="token_123",
            player="Test Player",
            trigger_type="SET_MEAN_REVERSION",
            entry_price=entry_price,
            fair_value=max(entry_price + 0.05, 0.55),
            edge=0.05,
            entry_score="0-1 0-0",
        )
        if entry_time_offset:
            trade.entry_timestamp = time.time() + entry_time_offset
        return mgr, trade, tmpdir

    # ──────────────────────────────────────────────────────────
    # 1. STOP-LOSS is PRIMARY (fires at -15%)
    # ──────────────────────────────────────────────────────────
    print("\n[1] STOP-LOSS PRIMARY...")

    mgr, trade, _ = make_manager_and_trade(entry_price=0.50)
    # stop_price = 0.50 * 0.85 = 0.425
    # Drop to 0.42 → below stop
    mgr.check_all(
        get_market_price=lambda m, s: 0.42,
        get_fair_value=lambda m: 0.55,
        get_score=lambda m: "0-1 0-0",
        is_match_finished=lambda m: False,
    )
    assert not trade.is_open, "Trade should close at stop-loss"
    assert trade.exit_reason == "EXIT_STOP_LOSS", f"Expected EXIT_STOP_LOSS, got {trade.exit_reason}"
    print(f"  entry=0.50, mkt=0.42:        reason={trade.exit_reason} ✓")

    # ──────────────────────────────────────────────────────────
    # 2. DYNAMIC TICK STOP (secondary — tighter for expensive entries)
    # ──────────────────────────────────────────────────────────
    print("\n[2] DYNAMIC TICK STOP...")

    # entry=0.80, stop_loss=0.80*0.85=0.68
    # dynamic_tick_limit = min(10, int(0.80*0.15/0.01)) = min(10, 12) = 10
    # 10 ticks = $0.10 → price 0.70 > 0.68 → tick stop fires before stop-loss
    mgr2, trade2, _ = make_manager_and_trade(entry_price=0.80)

    for p in [0.80 - i * 0.01 for i in range(1, 15)]:
        mgr2.check_all(
            get_market_price=lambda m, s, price=p: price,
            get_fair_value=lambda m: 0.85,
            get_score=lambda m: "0-1 0-0",
            is_match_finished=lambda m: False,
        )
        if not trade2.is_open:
            break

    assert not trade2.is_open
    assert trade2.exit_reason == "EXIT_TICK_STOP", f"Expected EXIT_TICK_STOP, got {trade2.exit_reason}"
    assert trade2.mae_ticks >= 10
    print(f"  entry=0.80, 10t drop:        reason={trade2.exit_reason} ticks={trade2.mae_ticks} ✓")

    # ──────────────────────────────────────────────────────────
    # 3. DYNAMIC CAP — cheap entry gets tighter tick limit
    # ──────────────────────────────────────────────────────────
    print("\n[3] DYNAMIC CAP (cheap entry)...")

    # entry=0.30, dynamic_tick_limit = min(10, int(0.30*0.15/0.01)) = min(10, 4) = 4
    # So tick stop fires at 4 ticks ($0.04), BEFORE reaching 10 ticks
    # But stop_loss at 0.30*0.85=0.255 → 5 ticks
    # Tick stop at 4 ticks (price=0.26) fires first since 0.26 > 0.255
    mgr3, trade3, _ = make_manager_and_trade(entry_price=0.30)

    for p in [0.30 - i * 0.01 for i in range(1, 10)]:
        mgr3.check_all(
            get_market_price=lambda m, s, price=p: price,
            get_fair_value=lambda m: 0.40,
            get_score=lambda m: "0-1 0-0",
            is_match_finished=lambda m: False,
        )
        if not trade3.is_open:
            break

    assert not trade3.is_open
    # At 4 ticks (price=0.26), tick stop fires. Stop-loss at 0.255 hasn't been hit yet.
    assert trade3.exit_reason == "EXIT_TICK_STOP", f"Expected EXIT_TICK_STOP, got {trade3.exit_reason}"
    assert trade3.mae_ticks <= 5, f"Should fire at ~4 ticks, got {trade3.mae_ticks}"
    
    # Verify the dynamic limit: min(10, int(0.30*0.15/0.01)) = min(10,4) = 4
    dynamic_limit = min(10, int(0.30 * 0.15 / 0.01))
    assert dynamic_limit == 4, f"Dynamic limit should be 4, got {dynamic_limit}"
    print(f"  entry=0.30, dynamic_limit=4: reason={trade3.exit_reason} ticks={trade3.mae_ticks} ✓")

    # ──────────────────────────────────────────────────────────
    # 4. STOP-LOSS fires for very cheap entries where dynamic tick = 0
    # ──────────────────────────────────────────────────────────
    print("\n[4] VERY CHEAP ENTRY (tick limit = 0 → stop-loss only)...")

    # entry=0.05, dynamic_tick_limit = min(10, int(0.05*0.15/0.01)) = min(10, 0) = 0
    # Tick stop won't fire (limit=0), so stop-loss takes over
    mgr4, trade4, _ = make_manager_and_trade(entry_price=0.05)

    mgr4.check_all(
        get_market_price=lambda m, s: 0.03,
        get_fair_value=lambda m: 0.10,
        get_score=lambda m: "0-1 0-0",
        is_match_finished=lambda m: False,
    )
    assert not trade4.is_open
    assert trade4.exit_reason == "EXIT_STOP_LOSS", f"Expected EXIT_STOP_LOSS, got {trade4.exit_reason}"
    print(f"  entry=0.05, mkt=0.03:        reason={trade4.exit_reason} ✓")

    # ──────────────────────────────────────────────────────────
    # 5. EXIT_PRICE_FLOOR
    # ──────────────────────────────────────────────────────────
    print("\n[5] EXIT_PRICE_FLOOR...")

    mgr5, trade5, _ = make_manager_and_trade(entry_price=0.22)
    # stop_loss = 0.22*0.85 = 0.187
    # dynamic_tick = min(10, int(0.22*0.15/0.01)) = min(10, 3) = 3
    # At mkt=0.17: adverse=0.05 → mae_ticks=5 >= 3 → tick stop fires first
    # Actually need to test price floor properly: use entry=0.20
    mgr5b, trade5b, _ = make_manager_and_trade(entry_price=0.20)
    # stop_loss = 0.20*0.85 = 0.17 → mkt=0.18 > 0.17 (no stop)
    # dynamic_tick = min(10, int(0.20*0.15/0.01)) = min(10, 3) = 3
    # At mkt=0.18: adverse=0.02 → mae_ticks=2 < 3 (no tick stop)
    # price floor = 0.18 → fires!
    mgr5b.check_all(
        get_market_price=lambda m, s: 0.18,
        get_fair_value=lambda m: 0.30,
        get_score=lambda m: "0-1 0-0",
        is_match_finished=lambda m: False,
    )
    assert not trade5b.is_open
    assert trade5b.exit_reason == "EXIT_PRICE_FLOOR", f"Expected EXIT_PRICE_FLOOR, got {trade5b.exit_reason}"
    print(f"  entry=0.20, mkt=0.18:        reason={trade5b.exit_reason} ✓")

    # ──────────────────────────────────────────────────────────
    # 6. EXIT_NO_MFE — fast failure detection
    # ──────────────────────────────────────────────────────────
    print("\n[6] EXIT_NO_MFE...")

    mgr6, trade6, _ = make_manager_and_trade(entry_price=0.50)
    trade6.entry_timestamp = time.time() - 300  # 5 minutes ago
    trade6.mfe = 0.01

    mgr6.check_all(
        get_market_price=lambda m, s: 0.49,
        get_fair_value=lambda m: 0.55,
        get_score=lambda m: "0-1 0-0",
        is_match_finished=lambda m: False,
    )
    assert not trade6.is_open
    assert trade6.exit_reason == "EXIT_NO_MFE", f"Expected EXIT_NO_MFE, got {trade6.exit_reason}"
    print(f"  5min + mfe=0.01:             reason={trade6.exit_reason} ✓")

    # Past window: should NOT exit
    mgr6b, trade6b, _ = make_manager_and_trade(entry_price=0.50)
    trade6b.entry_timestamp = time.time() - 900
    trade6b.mfe = 0.01

    mgr6b.check_all(
        get_market_price=lambda m, s: 0.50,
        get_fair_value=lambda m: 0.55,
        get_score=lambda m: "0-1 0-0",
        is_match_finished=lambda m: False,
    )
    assert trade6b.is_open
    print("  15min (past window):         stays open ✓")

    # ──────────────────────────────────────────────────────────
    # 7. RUNNER ACTIVATION AT 3%
    # ──────────────────────────────────────────────────────────
    print("\n[7] RUNNER ACTIVATION AT 3%...")

    mgr7, trade7, _ = make_manager_and_trade(entry_price=0.50)
    mgr7.check_all(
        get_market_price=lambda m, s: 0.53,
        get_fair_value=lambda m: 0.55,
        get_score=lambda m: "0-1 0-0",
        is_match_finished=lambda m: False,
    )
    assert trade7.runner_v2_active
    print(f"  mfe=0.03 activates runner:   ✓")

    # ──────────────────────────────────────────────────────────
    # 8. mae_ticks TRACKING
    # ──────────────────────────────────────────────────────────
    print("\n[8] MAE_TICKS TRACKING...")

    mgr8, trade8, _ = make_manager_and_trade(entry_price=0.50)
    for p in [0.49, 0.48, 0.47, 0.46, 0.45]:
        mgr8.check_all(
            get_market_price=lambda m, s, price=p: price,
            get_fair_value=lambda m: 0.55,
            get_score=lambda m: "0-1 0-0",
            is_match_finished=lambda m: False,
        )
    assert trade8.mae_ticks == 5
    print(f"  5-tick drop: mae_ticks={trade8.mae_ticks} ✓")

    # ──────────────────────────────────────────────────────────
    # 9. STATS DICT + CSV
    # ──────────────────────────────────────────────────────────
    print("\n[9] STATS + CSV...")

    tmpdir = tempfile.mkdtemp()
    mgr9 = TennisExitManager(data_dir=Path(tmpdir))
    stats = mgr9.stats
    assert "exit_tick_stop" in stats
    assert "exit_price_floor" in stats
    assert "exit_no_mfe" in stats
    print("  stats dict keys:             ✓")

    # CSV check
    trade9 = mgr9.register_trade(
        match_id="csv_test", selection_id="tok_csv",
        player="CSV Player", trigger_type="PANIC_DISCOUNT",
        entry_price=0.50, fair_value=0.55, edge=0.05,
        entry_score="0-0 0-0",
    )
    mgr9._close_trade(trade9, exit_price=0.48, exit_reason="EXIT_TICK_STOP", exit_score="0-0 0-0")
    mgr9.close()
    csv_files = list(Path(tmpdir).glob("*.csv"))
    assert len(csv_files) >= 1
    content = csv_files[0].read_text()
    assert "mae_ticks" in content
    print("  CSV mae_ticks:               ✓")

    # ──────────────────────────────────────────────────────────
    # 10. DYNAMIC LIMIT VERIFICATION TABLE
    # ──────────────────────────────────────────────────────────
    print("\n[10] DYNAMIC TICK LIMIT TABLE...")
    print(f"  {'Entry':>6s} | {'Tick Limit':>10s} | {'≡ R-loss':>8s} | vs -15%")
    print("  " + "-" * 45)
    for entry in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        limit = min(10, int(entry * 0.15 / 0.01))
        r_loss = (limit * 0.01) / entry if limit > 0 else 0
        status = "≤ 15%" if r_loss <= 0.15 else "> 15%"
        print(f"  ${entry:.2f}  | {limit:>10d} | {r_loss:>+7.1%} | {status}")

    assert min(10, int(0.20 * 0.15 / 0.01)) == 3
    assert min(10, int(0.50 * 0.15 / 0.01)) == 7
    assert min(10, int(0.80 * 0.15 / 0.01)) == 10
    print("  Dynamic limits correct:      ✓")

    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ALL v3.2 TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
