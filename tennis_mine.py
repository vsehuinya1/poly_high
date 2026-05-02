"""Compact tennis alpha scanner — runs against live WAL-mode DB"""
import sqlite3, time
conn = sqlite3.connect("sports_data/tick_history.db")
cur = conn.cursor()

print("=== 1. PRICE DISTRIBUTION ===")
for lo,hi in [(0,0.05),(0.05,0.10),(0.10,0.20),(0.20,0.50),(0.50,0.80),(0.80,0.95),(0.95,1)]:
    cur.execute("SELECT COUNT(*) FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id WHERE tl.sport='tennis' AND t.mid>=? AND t.mid<?", (lo,hi))
    print(f"  [{lo:.2f},{hi:.2f}): {cur.fetchone()[0]:>8,}")

print("\n=== 2. LOWER BAND PRICE PATHS (<0.15) ===")
# Sample 200 random LB ticks and check their +120s forward price
cur.execute("""
    SELECT t.token_id, t.timestamp, t.mid, t.spread
    FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id
    WHERE tl.sport='tennis' AND t.mid>0.03 AND t.mid<0.15 AND t.spread<0.05
    ORDER BY RANDOM() LIMIT 200
""")
lb_samples = cur.fetchall()
results_60 = []
results_120 = []
results_300 = []
for tid, ts, mid, spread in lb_samples:
    for delay, bucket in [(60, results_60), (120, results_120), (300, results_300)]:
        cur.execute("SELECT mid FROM ticks WHERE token_id=? AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp-?) LIMIT 1",
                    (tid, ts+delay-15, ts+delay+15, ts+delay))
        r = cur.fetchone()
        if r and r[0] > 0:
            bucket.append(r[0] - mid)

for label, bucket in [("t+60s", results_60), ("t+120s", results_120), ("t+300s", results_300)]:
    if bucket:
        avg = sum(bucket)/len(bucket)
        wr = sum(1 for p in bucket if p > 0.005)/len(bucket)
        big = sum(1 for p in bucket if p > 0.03)/len(bucket)
        print(f"  {label}: n={len(bucket)} avg={avg:+.4f} wr(>0.5c)={wr:.0%} big(>3c)={big:.0%}")

print("\n=== 3. SPREAD TIGHTENING SIGNAL ===")
# When spread goes from wide to tight, does price move?
cur.execute("""
    SELECT t.token_id, t.timestamp, t.mid, t.spread
    FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id
    WHERE tl.sport='tennis' AND t.mid>0.05 AND t.mid<0.30 AND t.spread<=0.02
    ORDER BY RANDOM() LIMIT 300
""")
tight_ticks = cur.fetchall()
tight_fwd = []
for tid, ts, mid, sp in tight_ticks:
    cur.execute("SELECT mid FROM ticks WHERE token_id=? AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp-?) LIMIT 1",
                (tid, ts+110, ts+130, ts+120))
    r = cur.fetchone()
    if r and r[0] > 0:
        tight_fwd.append(r[0] - mid)

# Compare: wide spread
cur.execute("""
    SELECT t.token_id, t.timestamp, t.mid, t.spread
    FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id
    WHERE tl.sport='tennis' AND t.mid>0.05 AND t.mid<0.30 AND t.spread>=0.06
    ORDER BY RANDOM() LIMIT 300
""")
wide_ticks = cur.fetchall()
wide_fwd = []
for tid, ts, mid, sp in wide_ticks:
    cur.execute("SELECT mid FROM ticks WHERE token_id=? AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp-?) LIMIT 1",
                (tid, ts+110, ts+130, ts+120))
    r = cur.fetchone()
    if r and r[0] > 0:
        wide_fwd.append(r[0] - mid)

if tight_fwd:
    print(f"  TIGHT (sp≤0.02): n={len(tight_fwd)} avg={sum(tight_fwd)/len(tight_fwd):+.4f} wr={sum(1 for p in tight_fwd if p>0)/len(tight_fwd):.0%}")
if wide_fwd:
    print(f"  WIDE  (sp≥0.06): n={len(wide_fwd)} avg={sum(wide_fwd)/len(wide_fwd):+.4f} wr={sum(1 for p in wide_fwd if p>0)/len(wide_fwd):.0%}")

print("\n=== 4. BIG DROP REVERSAL (>3c drop in <30s) ===")
cur.execute("""
    SELECT t1.token_id, t1.timestamp, t1.mid, t2.mid as prev_mid
    FROM ticks t1
    JOIN ticks t2 ON t1.token_id = t2.token_id 
        AND t2.timestamp BETWEEN t1.timestamp - 30 AND t1.timestamp - 1
    JOIN token_labels tl ON t1.token_id = tl.token_id
    WHERE tl.sport = 'tennis' AND t1.mid > 0.03 AND t1.mid < 0.40
        AND (t2.mid - t1.mid) > 0.03
    ORDER BY RANDOM() LIMIT 200
""")
drops = cur.fetchall()
drop_fwd60 = []
drop_fwd120 = []
for tid, ts, mid, prev in drops:
    drop_size = prev - mid
    for delay, bucket in [(60, drop_fwd60), (120, drop_fwd120)]:
        cur.execute("SELECT mid FROM ticks WHERE token_id=? AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp-?) LIMIT 1",
                    (tid, ts+delay-10, ts+delay+10, ts+delay))
        r = cur.fetchone()
        if r and r[0] > 0:
            bucket.append((r[0] - mid, drop_size))

for label, bucket in [("t+60s", drop_fwd60), ("t+120s", drop_fwd120)]:
    if bucket:
        bounces = [b[0] for b in bucket]
        avg = sum(bounces)/len(bounces)
        wr = sum(1 for b in bounces if b > 0.005)/len(bounces)
        avg_drop = sum(b[1] for b in bucket)/len(bucket)
        print(f"  {label}: n={len(bucket)} avg_bounce={avg:+.4f} wr={wr:.0%} avg_drop={avg_drop:.3f}")

print("\n=== 5. MICRO BREAKOUT (price crosses above recent resistance) ===")
cur.execute("""
    SELECT t.token_id, t.timestamp, t.mid, t.spread
    FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id
    WHERE tl.sport='tennis' AND t.mid>0.05 AND t.mid<0.25 AND t.spread<=0.03
    ORDER BY RANDOM() LIMIT 400
""")
samples = cur.fetchall()
breakout_fwd = []
for tid, ts, mid, sp in samples:
    # Check if price was lower in the last 60s (this is a breakout)
    cur.execute("SELECT MIN(mid), MAX(mid), AVG(mid) FROM ticks WHERE token_id=? AND timestamp BETWEEN ? AND ?",
                (tid, ts-120, ts-5))
    r = cur.fetchone()
    if not r or not r[0]:
        continue
    prev_lo, prev_hi, prev_avg = r[0], r[1], r[2]
    # Breakout: current > prev_hi AND recent range was tight
    if mid > prev_hi and (prev_hi - prev_lo) < 0.03 and mid - prev_avg > 0.01:
        cur.execute("SELECT mid FROM ticks WHERE token_id=? AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp-?) LIMIT 1",
                    (tid, ts+110, ts+130, ts+120))
        fwd = cur.fetchone()
        if fwd and fwd[0] > 0:
            breakout_fwd.append(fwd[0] - mid)

if breakout_fwd:
    avg = sum(breakout_fwd)/len(breakout_fwd)
    wr = sum(1 for p in breakout_fwd if p > 0)/len(breakout_fwd)
    print(f"  Micro breakouts: n={len(breakout_fwd)} avg={avg:+.4f} wr={wr:.0%}")
else:
    print("  No breakout patterns found in sample")

conn.close()
print("\n=== DONE ===")
