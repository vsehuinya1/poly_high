"""
Full-Spectrum Polymarket Alpha Scanner

Scans ALL tennis price levels to find where:
  movement magnitude > round-trip spread cost

Tests:
  1. Spread-to-movement ratio by price band
  2. Momentum continuation (3 consecutive moves)
  3. Large single-tick moves (volatility events)
  4. Limit-order edge (buy at bid, sell at ask)
  5. Score-change price dislocation (big moves > 5c)
"""
import sqlite3
from collections import defaultdict

conn = sqlite3.connect("sports_data/tick_history.db")
cur = conn.cursor()

print("=" * 70)
print("FULL-SPECTRUM ALPHA SCANNER")
print("=" * 70)

# ── 1. Spread & Movement by Price Band ──
print("\n=== 1. SPREAD vs MOVEMENT BY PRICE BAND ===")
bands = [(0.03,0.10),(0.10,0.20),(0.20,0.30),(0.30,0.50),(0.50,0.70),(0.70,0.85),(0.85,0.97)]
for lo, hi in bands:
    cur.execute("""
        SELECT t.token_id, t.timestamp, t.mid, t.spread
        FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id
        WHERE tl.sport='tennis' AND t.mid>=? AND t.mid<? AND t.spread>0
        ORDER BY RANDOM() LIMIT 500
    """, (lo, hi))
    samples = cur.fetchall()
    if not samples:
        continue
    
    avg_spread = sum(s[3] for s in samples) / len(samples)
    spread_pct = avg_spread / ((lo+hi)/2) * 100
    
    # Check movement at +60s, +300s
    moves_60 = []
    moves_300 = []
    for tid, ts, mid, sp in samples:
        for delay, bucket in [(60, moves_60), (300, moves_300)]:
            cur.execute("""SELECT mid FROM ticks WHERE token_id=? 
                AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp-?) LIMIT 1""",
                (tid, ts+delay-15, ts+delay+15, ts+delay))
            r = cur.fetchone()
            if r and r[0] > 0:
                bucket.append(abs(r[0] - mid))
    
    avg_move_60 = sum(moves_60)/len(moves_60) if moves_60 else 0
    avg_move_300 = sum(moves_300)/len(moves_300) if moves_300 else 0
    rt_cost = avg_spread  # round trip = buy at ask, sell at bid = 1 spread
    
    edge_60 = avg_move_60 - rt_cost
    edge_300 = avg_move_300 - rt_cost
    
    print(f"  [{lo:.2f},{hi:.2f}) n={len(samples):>3} | spread={avg_spread:.4f} ({spread_pct:.0f}% of price)")
    print(f"      move@60s={avg_move_60:.4f} move@300s={avg_move_300:.4f} | edge_60={edge_60:+.4f} edge_300={edge_300:+.4f}")

# ── 2. Big Single-Tick Moves (Score Changes) ──
print("\n=== 2. BIG MOVES (>5c in <30s) — SCORE CHANGE EVENTS ===")
for lo, hi in [(0.20,0.40),(0.40,0.60),(0.60,0.80)]:
    cur.execute("""
        SELECT t1.token_id, t1.timestamp, t1.mid, t1.spread,
               t2.mid as prev_mid, (t1.mid - t2.mid) as delta
        FROM ticks t1
        JOIN ticks t2 ON t1.token_id=t2.token_id
            AND t2.timestamp BETWEEN t1.timestamp-30 AND t1.timestamp-1
        JOIN token_labels tl ON t1.token_id=tl.token_id
        WHERE tl.sport='tennis' AND t1.mid>=? AND t1.mid<?
            AND ABS(t1.mid - t2.mid) > 0.05
        ORDER BY RANDOM() LIMIT 200
    """, (lo, hi))
    big_moves = cur.fetchall()
    if not big_moves:
        print(f"  [{lo:.2f},{hi:.2f}): no big moves found")
        continue
    
    # After a big UP move, does it continue or revert?
    up_cont = []
    dn_cont = []
    for tid, ts, mid, sp, prev, delta in big_moves:
        cur.execute("""SELECT mid FROM ticks WHERE token_id=?
            AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp-?) LIMIT 1""",
            (tid, ts+55, ts+65, ts+60))
        r = cur.fetchone()
        if not r: continue
        fwd = r[0] - mid
        if delta > 0:
            up_cont.append(fwd)
        else:
            dn_cont.append(fwd)
    
    if up_cont:
        avg_up = sum(up_cont)/len(up_cont)
        wr_up = sum(1 for x in up_cont if x > 0)/len(up_cont)
        print(f"  [{lo:.2f},{hi:.2f}) AFTER BIG UP (n={len(up_cont)}): continuation@60s={avg_up:+.4f} wr={wr_up:.0%}")
    if dn_cont:
        avg_dn = sum(dn_cont)/len(dn_cont)
        wr_dn = sum(1 for x in dn_cont if x < 0)/len(dn_cont)
        print(f"  [{lo:.2f},{hi:.2f}) AFTER BIG DN (n={len(dn_cont)}): continuation@60s={avg_dn:+.4f} sell_wr={wr_dn:.0%}")

# ── 3. Momentum Continuation (mid range, tight spread) ──
print("\n=== 3. MOMENTUM (3+ consecutive moves, mid-range, tight spread) ===")
for lo, hi in [(0.20,0.40),(0.40,0.60),(0.60,0.80)]:
    cur.execute("""
        SELECT t.token_id, t.timestamp, t.mid, t.spread
        FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id
        WHERE tl.sport='tennis' AND t.mid>=? AND t.mid<? AND t.spread>0
        ORDER BY t.token_id, t.timestamp
    """, (lo, hi))
    ticks = cur.fetchall()
    by_token = defaultdict(list)
    for tid, ts, mid, sp in ticks:
        by_token[tid].append((ts, mid, sp))
    
    mom_up = []
    mom_dn = []
    for tid, tlist in by_token.items():
        for i in range(3, len(tlist)-10):
            # 3 consecutive up
            if (tlist[i][1] > tlist[i-1][1] > tlist[i-2][1] > tlist[i-3][1]
                and tlist[i][1] - tlist[i-3][1] >= 0.02
                and tlist[i][2] <= 0.03):  # tight spread at signal
                # Forward 60s
                entry_ask = tlist[i][1] + tlist[i][2]/2
                target_ts = tlist[i][0] + 60
                fwd = [(t,m) for t,m,s in tlist[i:i+30] if abs(t - target_ts) < 15]
                if fwd:
                    exit_bid = fwd[0][1] - tlist[i][2]/2
                    mom_up.append(exit_bid - entry_ask)
            # 3 consecutive down
            if (tlist[i][1] < tlist[i-1][1] < tlist[i-2][1] < tlist[i-3][1]
                and tlist[i-3][1] - tlist[i][1] >= 0.02
                and tlist[i][2] <= 0.03):
                entry_bid = tlist[i][1] - tlist[i][2]/2  # sell at bid
                target_ts = tlist[i][0] + 60
                fwd = [(t,m) for t,m,s in tlist[i:i+30] if abs(t - target_ts) < 15]
                if fwd:
                    exit_ask = fwd[0][1] + tlist[i][2]/2
                    mom_dn.append(entry_bid - exit_ask)  # short PnL
    
    if mom_up:
        avg = sum(mom_up)/len(mom_up)
        wr = sum(1 for x in mom_up if x > 0)/len(mom_up)
        print(f"  [{lo:.2f},{hi:.2f}) UP momentum: n={len(mom_up)} avg_pnl={avg:+.4f} wr={wr:.0%} (after slippage)")
    if mom_dn:
        avg = sum(mom_dn)/len(mom_dn)
        wr = sum(1 for x in mom_dn if x > 0)/len(mom_dn)
        print(f"  [{lo:.2f},{hi:.2f}) DN momentum: n={len(mom_dn)} avg_pnl={avg:+.4f} wr={wr:.0%} (after slippage)")

# ── 4. Limit Order Edge (Market Making) ──
print("\n=== 4. LIMIT ORDER EDGE (buy at bid, sell at ask) ===")
for lo, hi in [(0.20,0.40),(0.40,0.60),(0.60,0.80)]:
    cur.execute("""
        SELECT t.token_id, t.timestamp, t.mid, t.spread
        FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id
        WHERE tl.sport='tennis' AND t.mid>=? AND t.mid<? AND t.spread>0
        ORDER BY RANDOM() LIMIT 500
    """, (lo, hi))
    samples = cur.fetchall()
    
    # Simulate: post bid order, check if mid drops to our bid (fill), then sell at ask
    fills = []
    for tid, ts, mid, sp in samples:
        our_bid = mid - sp/2  # post at best bid
        # Check if mid drops to our bid within 300s
        cur.execute("""SELECT MIN(mid), MAX(mid) FROM ticks 
            WHERE token_id=? AND timestamp>? AND timestamp<=?""",
            (tid, ts, ts + 300))
        r = cur.fetchone()
        if not r or not r[0]: continue
        lo_price, hi_price = r[0], r[1]
        
        if lo_price <= our_bid:
            # Filled! Now check if price goes back up to our ask
            our_ask = our_bid + sp  # sell at ask (spread is our profit)
            if hi_price >= our_ask:
                fills.append(sp)  # full spread captured
            else:
                # Partial: we got filled but price didn't reach our ask
                # Exit at mid after 300s
                cur.execute("""SELECT mid FROM ticks WHERE token_id=?
                    AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp-?) LIMIT 1""",
                    (tid, ts+290, ts+310, ts+300))
                exit_r = cur.fetchone()
                if exit_r:
                    fills.append(exit_r[0] - our_bid)
    
    if fills:
        avg_fill = sum(fills)/len(fills)
        wr = sum(1 for f in fills if f > 0)/len(fills)
        fill_rate = len(fills)/len(samples)
        print(f"  [{lo:.2f},{hi:.2f}) fill_rate={fill_rate:.0%} | avg_pnl={avg_fill:+.4f} | wr={wr:.0%} | n={len(fills)}")

# ── 5. Directional after sustained move (trend) ──
print("\n=== 5. TREND FOLLOWING — price moved >10c in last 300s ===")
for lo, hi in [(0.20,0.50),(0.50,0.80)]:
    cur.execute("""
        SELECT t.token_id, t.timestamp, t.mid, t.spread
        FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id
        WHERE tl.sport='tennis' AND t.mid>=? AND t.mid<? AND t.spread>0 AND t.spread<=0.03
        ORDER BY RANDOM() LIMIT 500
    """, (lo, hi))
    samples = cur.fetchall()
    
    trend_up = []
    trend_dn = []
    for tid, ts, mid, sp in samples:
        # Check price 300s ago
        cur.execute("""SELECT mid FROM ticks WHERE token_id=?
            AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp-?) LIMIT 1""",
            (tid, ts-310, ts-290, ts-300))
        r = cur.fetchone()
        if not r or not r[0]: continue
        past_mid = r[0]
        trend = mid - past_mid
        
        if abs(trend) < 0.10: continue  # need >10c trend
        
        # Forward 120s
        cur.execute("""SELECT mid FROM ticks WHERE token_id=?
            AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp-?) LIMIT 1""",
            (tid, ts+110, ts+130, ts+120))
        fwd_r = cur.fetchone()
        if not fwd_r: continue
        
        entry_cost = sp  # round trip slippage
        fwd_move = fwd_r[0] - mid
        
        if trend > 0:
            trend_up.append(fwd_move - entry_cost)  # buy continuation
        else:
            trend_dn.append(-fwd_move - entry_cost)  # sell continuation
    
    if trend_up:
        avg = sum(trend_up)/len(trend_up)
        wr = sum(1 for x in trend_up if x > 0)/len(trend_up)
        print(f"  [{lo:.2f},{hi:.2f}) TREND UP cont: n={len(trend_up)} avg={avg:+.4f} wr={wr:.0%}")
    if trend_dn:
        avg = sum(trend_dn)/len(trend_dn)
        wr = sum(1 for x in trend_dn if x > 0)/len(trend_dn)
        print(f"  [{lo:.2f},{hi:.2f}) TREND DN cont: n={len(trend_dn)} avg={avg:+.4f} wr={wr:.0%}")

conn.close()
print(f"\n{'='*70}")
print("SCAN COMPLETE")
print(f"{'='*70}")
