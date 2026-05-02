"""TSLB Exit Optimization — sweep hold time, trailing stop, take-profit"""
import sqlite3
from collections import defaultdict

conn = sqlite3.connect("sports_data/tick_history.db")
cur = conn.cursor()

MAX_PRICE = 0.20
MAX_SPREAD = 0.03
MIN_PRICE = 0.03
DEDUP_S = 90.0

# Load all TSLB candidates
cur.execute("""
    SELECT t.token_id, t.timestamp, t.mid, t.spread
    FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id
    WHERE tl.sport='tennis' AND t.mid>? AND t.mid<? AND t.spread>0 AND t.spread<=?
    ORDER BY t.timestamp ASC
""", (MIN_PRICE, MAX_PRICE, MAX_SPREAD))
candidates = cur.fetchall()

# Dedup
last_entry = {}
entries = []
for tid, ts, mid, sp in candidates:
    if ts - last_entry.get(tid, 0) < DEDUP_S:
        continue
    last_entry[tid] = ts
    entries.append((tid, ts, mid, sp))

print(f"Entries after dedup: {len(entries)}")

# Pre-fetch forward paths for all entries (batch)
print("Loading forward paths...")
paths = {}
for tid, ts, mid, sp in entries:
    cur.execute("""
        SELECT timestamp - ?, mid FROM ticks
        WHERE token_id=? AND timestamp>? AND timestamp<=?
        ORDER BY timestamp
    """, (ts, tid, ts, ts + 610))
    p = cur.fetchall()
    if len(p) >= 3:
        paths[(tid, ts)] = (mid, p)

print(f"Paths loaded: {len(paths)}")

# Sweep
hold_times = [60, 120, 180, 300, 420, 600]
trail_stops = [0.01, 0.015, 0.02, 0.03, 0.05, 999]  # 999 = no trailing
take_profits = [0.5, 1.0, 1.5, 2.0, 3.0, 999]  # multiples of entry (999 = no TP)

print(f"\nSweeping {len(hold_times)}x{len(trail_stops)}x{len(take_profits)} = {len(hold_times)*len(trail_stops)*len(take_profits)} combos...\n")

results = []
for ht in hold_times:
    for ts_val in trail_stops:
        for tp in take_profits:
            pnls = []
            for key, (entry, path) in paths.items():
                peak = entry
                exit_p = entry
                for dt, price in path:
                    if price > peak:
                        peak = price
                    # Take profit
                    if tp < 999 and price >= entry * (1 + tp):
                        exit_p = price
                        break
                    # Trailing stop
                    if ts_val < 999 and peak > entry and peak - price >= ts_val:
                        exit_p = price
                        break
                    # Timeout
                    if dt >= ht:
                        exit_p = price
                        break
                pnl_r = (exit_p - entry) / entry if entry > 0 else 0
                pnls.append(pnl_r)

            if not pnls:
                continue
            wins = [p for p in pnls if p > 0.005]
            losses = [p for p in pnls if p < -0.005]
            avg = sum(pnls) / len(pnls)
            total = sum(pnls)
            wr = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0

            # Kelly
            if wins and losses:
                p = len(wins) / (len(wins) + len(losses))
                avg_w = sum(wins) / len(wins)
                avg_l = abs(sum(losses) / len(losses))
                b = avg_w / avg_l if avg_l > 0 else 1
                kelly = max(0, (p * b - (1 - p)) / b)
            else:
                kelly = 0

            # Simulated equity (quarter kelly)
            eq = 100.0
            for r in pnls:
                eq += (kelly / 4) * eq * r
                if eq <= 0: eq = 0; break

            results.append({
                "hold": ht, "trail": ts_val, "tp": tp,
                "n": len(pnls), "avg_r": avg, "wr": wr,
                "kelly": kelly, "equity": eq, "total_r": total,
            })

# Sort by equity
results.sort(key=lambda x: -x["equity"])

print("=" * 90)
print(f"{'Hold':>5} {'Trail':>6} {'TP':>5} | {'N':>6} {'Avg_R':>8} {'WR':>6} {'Kelly':>7} | {'QK Equity':>12}")
print("=" * 90)
for r in results[:30]:
    trail_s = f"{r['trail']:.3f}" if r['trail'] < 999 else "NONE"
    tp_s = f"{r['tp']:.1f}x" if r['tp'] < 999 else "NONE"
    print(f"{r['hold']:>5}s {trail_s:>6} {tp_s:>5} | {r['n']:>6} {r['avg_r']:>+7.2%} {r['wr']:>5.0%} {r['kelly']:>6.1%} | ${r['equity']:>11,.2f}")

# Best by category
print(f"\n{'='*70}")
print("BEST BY CATEGORY")
print(f"{'='*70}")
best_eq = results[0]
best_wr = max(results, key=lambda x: x["wr"])
best_kelly = max(results, key=lambda x: x["kelly"])
best_sharpe = max(results, key=lambda x: x["avg_r"])

for label, r in [("Max Equity", best_eq), ("Max WR", best_wr), ("Max Kelly", best_kelly), ("Max Avg R", best_sharpe)]:
    trail_s = f"{r['trail']:.3f}" if r['trail'] < 999 else "NONE"
    tp_s = f"{r['tp']:.1f}x" if r['tp'] < 999 else "NONE"
    print(f"  {label:>12}: hold={r['hold']}s trail={trail_s} tp={tp_s} | equity=${r['equity']:,.0f} kelly={r['kelly']:.1%} wr={r['wr']:.0%}")

conn.close()
print("\nDONE")
