"""Tennis tick data miner — extract alpha patterns from tick_history.db"""
import sqlite3
import json
from collections import defaultdict
from datetime import datetime, timezone

DB = "sports_data/tick_history.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

# 1. How much tennis tick data do we have?
cur.execute("""
    SELECT COUNT(*) as cnt, COUNT(DISTINCT t.token_id) as tokens
    FROM ticks t
    JOIN token_labels tl ON t.token_id = tl.token_id
    WHERE tl.sport = 'tennis'
""")
row = cur.fetchone()
print(f"=== TENNIS TICK DATA ===")
print(f"Total ticks: {row[0]:,}")
print(f"Unique tokens: {row[1]}")

# 2. Top 20 tennis markets by tick count
print(f"\n=== TOP 20 MARKETS BY ACTIVITY ===")
cur.execute("""
    SELECT tl.market_title, COUNT(*) as cnt, 
           MIN(t.timestamp) as t0, MAX(t.timestamp) as t1,
           MIN(t.mid) as lo, MAX(t.mid) as hi, AVG(t.mid) as avg_mid
    FROM ticks t
    JOIN token_labels tl ON t.token_id = tl.token_id
    WHERE tl.sport = 'tennis'
    GROUP BY tl.market_title
    ORDER BY cnt DESC
    LIMIT 20
""")
for title, cnt, t0, t1, lo, hi, avg in cur.fetchall():
    d0 = datetime.fromtimestamp(t0, tz=timezone.utc).strftime('%m-%d')
    d1 = datetime.fromtimestamp(t1, tz=timezone.utc).strftime('%m-%d')
    hrs = (t1 - t0) / 3600
    print(f"  {cnt:>7,} ticks | {d0}→{d1} ({hrs:.0f}h) | lo={lo:.3f} hi={hi:.3f} | {title[:60]}")

# 3. Price distribution — where do most ticks fall?
print(f"\n=== PRICE DISTRIBUTION (ALL TENNIS) ===")
buckets = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), 
           (0.30, 0.50), (0.50, 0.70), (0.70, 0.80), (0.80, 0.90),
           (0.90, 0.95), (0.95, 1.0)]
for lo, hi in buckets:
    cur.execute("""
        SELECT COUNT(*) FROM ticks t
        JOIN token_labels tl ON t.token_id = tl.token_id
        WHERE tl.sport = 'tennis' AND t.mid >= ? AND t.mid < ?
    """, (lo, hi))
    cnt = cur.fetchone()[0]
    print(f"  [{lo:.2f}, {hi:.2f}): {cnt:>8,} ticks")

# 4. LOWER BAND DEEP DIVE (< 0.20)
print(f"\n=== LOWER BAND (<0.20) — PRICE MOVEMENT PATTERNS ===")
# Find tokens that spent time below 0.20 and then moved
cur.execute("""
    SELECT t.token_id, tl.market_title, t.timestamp, t.mid
    FROM ticks t
    JOIN token_labels tl ON t.token_id = tl.token_id
    WHERE tl.sport = 'tennis' AND t.mid > 0.02 AND t.mid < 0.20
    ORDER BY t.token_id, t.timestamp
""")
lb_ticks = cur.fetchall()
print(f"  Lower-band ticks: {len(lb_ticks):,}")

# Group by token, find price trajectories
by_token = defaultdict(list)
for tid, title, ts, mid in lb_ticks:
    by_token[tid].append((ts, mid, title))

# For each token that enters LB, track what happens next
print(f"  Tokens entering lower band: {len(by_token)}")

reversals = []
collapses = []
for tid, ticks in by_token.items():
    title = ticks[0][2]
    # Find entry point (first tick < 0.20)
    entry_ts, entry_mid = ticks[0][0], ticks[0][1]
    
    # Get ALL ticks for this token (including above 0.20) for forward path
    cur.execute("""
        SELECT timestamp, mid FROM ticks 
        WHERE token_id = ? AND timestamp >= ? 
        ORDER BY timestamp LIMIT 500
    """, (tid, entry_ts))
    full_path = cur.fetchall()
    
    if len(full_path) < 10:
        continue
    
    # Forward prices at various intervals
    for snap_s in [30, 60, 120, 300, 600]:
        target_ts = entry_ts + snap_s
        closest = min(full_path, key=lambda x: abs(x[0] - target_ts))
        if abs(closest[0] - target_ts) < snap_s * 0.3:  # within 30% tolerance
            fwd_price = closest[1]
            pnl = fwd_price - entry_mid
            r_mult = pnl / entry_mid if entry_mid > 0 else 0
            if snap_s == 300:
                if r_mult > 0.10:
                    reversals.append((tid, title, entry_mid, fwd_price, r_mult))
                elif r_mult < -0.20:
                    collapses.append((tid, title, entry_mid, fwd_price, r_mult))

    # Find max price in next 10 minutes
    max_fwd = max(p[1] for p in full_path[:100]) if full_path else entry_mid
    max_r = (max_fwd - entry_mid) / entry_mid if entry_mid > 0 else 0

print(f"\n  REVERSALS (R_300s > +10%): {len(reversals)}")
for tid, title, entry, fwd, r in sorted(reversals, key=lambda x: -x[4])[:10]:
    print(f"    entry={entry:.3f} → fwd={fwd:.3f} | R={r:+.1%} | {title[:55]}")

print(f"\n  COLLAPSES (R_300s < -20%): {len(collapses)}")
for tid, title, entry, fwd, r in sorted(collapses, key=lambda x: x[4])[:10]:
    print(f"    entry={entry:.3f} → fwd={fwd:.3f} | R={r:+.1%} | {title[:55]}")

# 5. SPREAD ANALYSIS — do tight spreads predict movement?
print(f"\n=== SPREAD ↔ MOVEMENT CORRELATION ===")
cur.execute("""
    SELECT t.spread, t.mid, t.token_id, t.timestamp
    FROM ticks t
    JOIN token_labels tl ON t.token_id = tl.token_id
    WHERE tl.sport = 'tennis' AND t.mid > 0.03 AND t.mid < 0.20
      AND t.spread > 0 AND t.spread < 0.5
    ORDER BY t.spread ASC
    LIMIT 5000
""")
spread_ticks = cur.fetchall()
tight_wins = 0
tight_total = 0
wide_wins = 0
wide_total = 0
for spread, mid, tid, ts in spread_ticks:
    cur.execute("""
        SELECT mid FROM ticks WHERE token_id = ? 
        AND timestamp BETWEEN ? AND ? ORDER BY timestamp DESC LIMIT 1
    """, (tid, ts + 250, ts + 350))
    r = cur.fetchone()
    if not r:
        continue
    fwd = r[0]
    pnl = fwd - mid
    if spread <= 0.02:
        tight_total += 1
        if pnl > 0.005:
            tight_wins += 1
    elif spread >= 0.05:
        wide_total += 1
        if pnl > 0.005:
            wide_wins += 1

print(f"  Tight spread (≤0.02): {tight_wins}/{tight_total} positive at +300s")
print(f"  Wide spread (≥0.05): {wide_wins}/{wide_total} positive at +300s")

# 6. MOMENTUM — consecutive price moves in same direction
print(f"\n=== MOMENTUM PATTERNS (price sequences) ===")
cur.execute("""
    SELECT t.token_id, tl.market_title, t.timestamp, t.mid, t.spread
    FROM ticks t
    JOIN token_labels tl ON t.token_id = tl.token_id
    WHERE tl.sport = 'tennis' AND t.mid > 0.03 AND t.mid < 0.30
    ORDER BY t.token_id, t.timestamp
""")
all_ticks = cur.fetchall()
by_token2 = defaultdict(list)
for tid, title, ts, mid, spread in all_ticks:
    by_token2[tid].append((ts, mid, spread))

momentum_up = []
momentum_dn = []
for tid, ticks in by_token2.items():
    for i in range(3, len(ticks)):
        # 3 consecutive up moves
        if (ticks[i][1] > ticks[i-1][1] > ticks[i-2][1] > ticks[i-3][1]
            and ticks[i][1] - ticks[i-3][1] >= 0.01):
            # Check forward
            target_ts = ticks[i][0] + 120
            fwd = [(t, m) for t, m, s in ticks[i:i+50] if abs(t - target_ts) < 30]
            if fwd:
                pnl = fwd[0][1] - ticks[i][1]
                momentum_up.append(pnl)
        # 3 consecutive down
        if (ticks[i][1] < ticks[i-1][1] < ticks[i-2][1] < ticks[i-3][1]
            and ticks[i-3][1] - ticks[i][1] >= 0.01):
            target_ts = ticks[i][0] + 120
            fwd = [(t, m) for t, m, s in ticks[i:i+50] if abs(t - target_ts) < 30]
            if fwd:
                pnl = ticks[i][1] - fwd[0][1]
                momentum_dn.append(pnl)

if momentum_up:
    avg_up = sum(momentum_up) / len(momentum_up)
    wr_up = sum(1 for p in momentum_up if p > 0) / len(momentum_up)
    print(f"  After 3 consecutive UP moves (n={len(momentum_up)}):")
    print(f"    Continuation at +120s: avg={avg_up:+.4f} | wr={wr_up:.0%}")
if momentum_dn:
    avg_dn = sum(momentum_dn) / len(momentum_dn)
    wr_dn = sum(1 for p in momentum_dn if p > 0) / len(momentum_dn)
    print(f"  After 3 consecutive DOWN moves (n={len(momentum_dn)}):")
    print(f"    Mean reversion at +120s: avg={avg_dn:+.4f} | wr={wr_dn:.0%}")

# 7. QUICK DUMP patterns
print(f"\n=== QUICK DUMP REVERSALS (price drops >3c in <60s) ===")
dump_reversals = []
for tid, ticks in by_token2.items():
    for i in range(1, len(ticks)):
        dt = ticks[i][0] - ticks[i-1][0]
        dp = ticks[i][1] - ticks[i-1][1]
        if dt < 60 and dp < -0.03 and ticks[i][1] > 0.03:
            # Quick dump — check if it reverses
            target_ts = ticks[i][0] + 120
            fwd = [(t, m) for t, m, s in ticks[i:i+50] if abs(t - target_ts) < 30]
            if fwd:
                bounce = fwd[0][1] - ticks[i][1]
                dump_reversals.append((dp, bounce, ticks[i][1]))

if dump_reversals:
    avg_bounce = sum(r[1] for r in dump_reversals) / len(dump_reversals)
    wr = sum(1 for r in dump_reversals if r[1] > 0) / len(dump_reversals)
    print(f"  Quick dumps found: {len(dump_reversals)}")
    print(f"  Avg bounce at +120s: {avg_bounce:+.4f}")
    print(f"  Win rate (any bounce): {wr:.0%}")
    # By size of dump
    big_dumps = [r for r in dump_reversals if r[0] < -0.05]
    if big_dumps:
        avg_big = sum(r[1] for r in big_dumps) / len(big_dumps)
        wr_big = sum(1 for r in big_dumps if r[1] > 0) / len(big_dumps)
        print(f"  BIG dumps (>5c, n={len(big_dumps)}): bounce={avg_big:+.4f} wr={wr_big:.0%}")

conn.close()
print("\n=== MINING COMPLETE ===")
