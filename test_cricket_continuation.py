#!/usr/bin/env python3
"""Test: Cricket Continuation Strategy v2.0"""
import os, sys, time, logging

logging.basicConfig(level=logging.INFO, format="  %(name)-25s %(message)s")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sports.guards import circuit_breaker
# Reset breaker
circuit_breaker.reset()
circuit_breaker._r_values.clear()
circuit_breaker._recent_outcomes.clear()
circuit_breaker._disabled = False

from cricket.tick_strategy import CricketTickDetector, CricketTickSignal

passed = 0
total = 0

def check(name, condition):
    global passed, total
    total += 1
    status = "✓" if condition else "✗ FAIL"
    if condition: passed += 1
    print(f"  [{total:2d}] {name:50s} {status}")

print("=" * 65)
print("  CRICKET CONTINUATION v2.0 — STRATEGY TESTS")
print("=" * 65)

# ──────────────────────────────────────────────────────────────
# 1. IPL Filter — non-IPL markets rejected
# ──────────────────────────────────────────────────────────────
print("\n[1] IPL MARKET FILTER...")
det = CricketTickDetector()
t0 = time.time()

# Feed non-IPL ticks with a genuine spike
for i in range(20):
    det.on_tick("non_ipl_1", 0.50 + i * 0.003, 0.01, t0 + i,
                market_title="T20 World Cup: Ghana vs Nigeria")
sig = det.on_tick("non_ipl_1", 0.58, 0.01, t0 + 20,
                  market_title="T20 World Cup: Ghana vs Nigeria")
check("Non-IPL market blocked", sig is None)

# ──────────────────────────────────────────────────────────────
# 2. Spike Continuation — basic signal flow
# ──────────────────────────────────────────────────────────────
print("\n[2] SPIKE CONTINUATION SIGNAL...")
det2 = CricketTickDetector()
t0 = time.time()
title = "Indian Premier League: RCB vs CSK"

# Build up 10 baseline ticks at 0.50 (t0+0 through t0+9)
for i in range(10):
    det2.on_tick("ipl_1", 0.50, 0.02, t0 + i, market_title=title)

# Simulate spike UP: 0.50 → 0.55 (move=0.05 > 0.04)
# spike_age measured from last baseline at t0+9
det2.on_tick("ipl_1", 0.52, 0.04, t0 + 11, market_title=title)  # spread expanding
det2.on_tick("ipl_1", 0.54, 0.03, t0 + 12, market_title=title)  # spread contracting tick 1
sig = det2.on_tick("ipl_1", 0.55, 0.02, t0 + 13, market_title=title)  # spike_age=4s, contraction tick 2
# spike_age = 13 - 9 = 4s → < 5s confirmation → too early
check("Too early (4s) → no signal", sig is None)

# 1 more tick → spike_age = 14 - 9 = 5s, should fire
sig = det2.on_tick("ipl_1", 0.55, 0.01, t0 + 14, market_title=title)
check("After 5s confirmation → signal fires", sig is not None)

if sig:
    check("Signal type = SPIKE_CONTINUATION", sig.signal_type == "SPIKE_CONTINUATION")
    check("Direction = LONG (same as spike UP)", sig.direction == "LONG")
    check("Move ≥ 0.04", sig.move >= 0.04)
    check("Edge > 0", sig.edge > 0)
    check("Edge ≥ 0.01 (min edge)", sig.edge >= 0.01)
    # projected_move = max(0.02, min(0.05, 0.05 * 0.6)) = 0.03
    check("Projected move = 0.03", abs(sig.fair_price - sig.entry_price - 0.03) < 0.001)

# ──────────────────────────────────────────────────────────────
# 3. Retrace rejection — >33% retrace blocks signal
# ──────────────────────────────────────────────────────────────
print("\n[3] RETRACE REJECTION...")
det3 = CricketTickDetector()
t0 = time.time()

# Baseline
for i in range(10):
    det3.on_tick("ipl_2", 0.50, 0.01, t0 + i, market_title=title)

# Spike UP to 0.55 (move = 0.05)
det3.on_tick("ipl_2", 0.55, 0.03, t0 + 11, market_title=title)
# Retrace 40% → back to 0.53 (retrace = 0.02 / 0.05 = 40% > 33%)
det3.on_tick("ipl_2", 0.53, 0.02, t0 + 13, market_title=title)
det3.on_tick("ipl_2", 0.53, 0.01, t0 + 14, market_title=title)
sig = det3.on_tick("ipl_2", 0.53, 0.01, t0 + 16, market_title=title)
check("40% retrace → blocked", sig is None)

# ──────────────────────────────────────────────────────────────
# 4. Spread contraction required
# ──────────────────────────────────────────────────────────────
print("\n[4] SPREAD CONTRACTION...")
det4 = CricketTickDetector()
t0 = time.time()

for i in range(10):
    det4.on_tick("ipl_3", 0.50, 0.01, t0 + i, market_title=title)

# Spike with expanding spread (no contraction)
det4.on_tick("ipl_3", 0.55, 0.02, t0 + 11, market_title=title)
det4.on_tick("ipl_3", 0.55, 0.03, t0 + 13, market_title=title)
sig = det4.on_tick("ipl_3", 0.55, 0.04, t0 + 15, market_title=title)
check("No spread contraction → blocked", sig is None)

# ──────────────────────────────────────────────────────────────
# 5. Small spike rejected
# ──────────────────────────────────────────────────────────────
print("\n[5] SMALL SPIKE FILTER...")
det5 = CricketTickDetector()
t0 = time.time()

for i in range(10):
    det5.on_tick("ipl_4", 0.50, 0.01, t0 + i, market_title=title)

# Move of 0.03 < 0.04 threshold
det5.on_tick("ipl_4", 0.53, 0.02, t0 + 11, market_title=title)
det5.on_tick("ipl_4", 0.53, 0.01, t0 + 13, market_title=title)
sig = det5.on_tick("ipl_4", 0.53, 0.01, t0 + 16, market_title=title)
check("Move 0.03 < 0.04 → no signal", sig is None)

# ──────────────────────────────────────────────────────────────
# 6. Exit: STOP_LOSS
# ──────────────────────────────────────────────────────────────
print("\n[6] EXIT — STOP LOSS...")
det6 = CricketTickDetector()
t0 = time.time()

# Create signal manually
sig_long = CricketTickSignal(
    signal_type="SPIKE_CONTINUATION", match_id="exit_test_1",
    direction="LONG", move=0.05, entry_price=0.50,
    fair_price=0.53, edge=0.025, spread=0.01
)
det6.register_trade(sig_long)
check("Trade registered", "exit_test_1" in det6._trades)

# Price drops to stop (0.50 - 0.06 = 0.44)
exits = det6.check_exits(lambda mid: (0.44, 0.01))
check("Stop loss triggered at 0.44", len(exits) == 1 and exits[0][4] == "EXIT_STOP_LOSS")

# ──────────────────────────────────────────────────────────────
# 7. Exit: EXIT_MOMENTUM_FAIL
# ──────────────────────────────────────────────────────────────
print("\n[7] EXIT — MOMENTUM FAIL...")
det7 = CricketTickDetector()

sig_long2 = CricketTickSignal(
    signal_type="SPIKE_CONTINUATION", match_id="exit_test_2",
    direction="LONG", move=0.05, entry_price=0.50,
    fair_price=0.53, edge=0.025, spread=0.01
)
det7.register_trade(sig_long2)

# Price flat at 0.50 for 45+ seconds — no new high
# Simulate by manipulating trade timestamp
det7._trades["exit_test_2"].entry_timestamp = time.time() - 50  # 50s ago
det7._trades["exit_test_2"].last_extreme_ts = time.time() - 50  # no new extreme

exits = det7.check_exits(lambda mid: (0.50, 0.01))
check("Momentum fail after 45s flat", len(exits) == 1 and exits[0][4] == "EXIT_MOMENTUM_FAIL")

# ──────────────────────────────────────────────────────────────
# 8. Exit: EXIT_RUNNER
# ──────────────────────────────────────────────────────────────
print("\n[8] EXIT — RUNNER TRAILING...")
det8 = CricketTickDetector()

sig_long3 = CricketTickSignal(
    signal_type="SPIKE_CONTINUATION", match_id="exit_test_3",
    direction="LONG", move=0.05, entry_price=0.50,
    fair_price=0.53, edge=0.025, spread=0.01
)
det8.register_trade(sig_long3)

# Price rises to 0.54 (MFE = 0.04 ≥ 0.03 activation)
exits = det8.check_exits(lambda mid: (0.54, 0.01))
check("Runner activated at MFE 0.04", det8._trades.get("exit_test_3") and det8._trades["exit_test_3"].runner_active)
check("No exit yet (price above trail)", len(exits) == 0)

# Price drops to trail (0.54 - 0.02 = 0.52)
exits = det8.check_exits(lambda mid: (0.52, 0.01))
check("Runner exit at trail distance", len(exits) == 1 and exits[0][4] == "EXIT_RUNNER")

# ──────────────────────────────────────────────────────────────
# 9. DOWN spike → SHORT direction
# ──────────────────────────────────────────────────────────────
print("\n[9] SHORT DIRECTION...")
det9 = CricketTickDetector()
t0 = time.time()

for i in range(10):
    det9.on_tick("ipl_5", 0.60, 0.01, t0 + i, market_title=title)

# Spike DOWN: 0.60 → 0.54 (move = -0.06)
det9.on_tick("ipl_5", 0.56, 0.03, t0 + 11, market_title=title)
det9.on_tick("ipl_5", 0.54, 0.02, t0 + 13, market_title=title)
sig = det9.on_tick("ipl_5", 0.54, 0.01, t0 + 16, market_title=title)
if sig:
    check("DOWN spike → SHORT direction", sig.direction == "SHORT")
    check("Signal type SPIKE_CONTINUATION", sig.signal_type == "SPIKE_CONTINUATION")
else:
    check("DOWN spike → signal generated", False)
    check("Signal type", False)

# ──────────────────────────────────────────────────────────────
# 10. No old signal types remain
# ──────────────────────────────────────────────────────────────
print("\n[10] REGRESSION — NO OLD SIGNAL TYPES...")
with open(os.path.join(os.path.dirname(__file__), "cricket", "tick_strategy.py")) as f:
    src = f.read()
check("No SPIKE_REVERSION in tick_strategy.py", "SPIKE_REVERSION" not in src)
check("No DRIFT_REVERSION in tick_strategy.py", "DRIFT_REVERSION" not in src)
check("SPIKE_CONTINUATION present", "SPIKE_CONTINUATION" in src)
check("MOMENTUM_DRIFT present", "MOMENTUM_DRIFT" in src)
check("EXIT_MOMENTUM_FAIL present", "EXIT_MOMENTUM_FAIL" in src)
check("EXIT_RUNNER present", "EXIT_RUNNER" in src)

# ──────────────────────────────────────────────────────────────
# 11. Guard integration still works
# ──────────────────────────────────────────────────────────────
print("\n[11] GUARD INTEGRATION...")
det11 = CricketTickDetector()
bad_sig = CricketTickSignal(
    signal_type="SPIKE_CONTINUATION", match_id="guard_test",
    direction="LONG", move=0.05, entry_price=0.50,
    fair_price=0.53, edge=0.0005,  # edge too low → guard will block
    spread=0.01
)
det11.register_trade(bad_sig)
check("Zero-edge trade blocked by guard", "guard_test" not in det11._trades)

# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print(f"  RESULTS: {passed}/{total} tests passed")
print(f"{'=' * 65}")

if passed == total:
    print("\n  ✅ ALL TESTS PASSED — cricket continuation v2.0 verified")
else:
    print(f"\n  ❌ {total - passed} TESTS FAILED")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# EXAMPLE LOG OUTPUT (simulated trade)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  EXAMPLE TRADE LOG")
print("=" * 65)
print("""
  CRICKET_SIGNAL_CONTINUATION | type=SPIKE_CONTINUATION |
    spike_size=0.0500 | delay_s=5.0 | retrace_pct=0% |
    spread=0.0100 | edge=0.0250 | projected=0.0300 |
    dir=LONG | mid=0.5500 | ipl_match_123

  CRICKET_TICK_TRADE | SPIKE_CONTINUATION | LONG |
    entry=0.5500 | stop=0.4900 | timeout=300s | ipl_match_123

  CRICKET_RUNNER_ACTIVE | mfe=0.0300 |
    entry=0.5500 mid=0.5800 | ipl_match_123

  CRICKET_TICK_EXIT | SPIKE_CONTINUATION | EXIT_RUNNER |
    entry=0.5500 exit=0.5700 | pnl=+0.0200 R=+0.333 |
    mfe=0.0300 | runner=True | hold=85s | ipl_match_123

  --- OR ---

  CRICKET_EXIT_MOMENTUM_FAIL | duration=46s |
    mfe=0.0050 | entry=0.5500 mid=0.5490 |
    time_since_extreme=46s | ipl_match_123
""")
