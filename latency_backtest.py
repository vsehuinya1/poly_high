"""
Score-Change Latency Edge Backtest

Finds all big price moves (score changes) in tennis tick data,
then simulates catching them at different latency delays.

Question: If we react in 1s vs 3s vs 6s, what does $100 become?
"""
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

conn = sqlite3.connect("sports_data/tick_history.db")
cur = conn.cursor()

print("=" * 70)
print("SCORE-CHANGE LATENCY EDGE BACKTEST")
print("=" * 70)

# Step 1: Find all big moves (>5c within 30s) in tradeable range
print("\nFinding score-change events...")
cur.execute("""
    SELECT t1.token_id, t1.timestamp, t1.mid, t1.spread,
           t2.mid as prev_mid, tl.market_title
    FROM ticks t1
    JOIN ticks t2 ON t1.token_id=t2.token_id
        AND t2.timestamp BETWEEN t1.timestamp-30 AND t1.timestamp-3
    JOIN token_labels tl ON t1.token_id=tl.token_id
    WHERE tl.sport='tennis' AND t1.mid>0.10 AND t1.mid<0.90
      AND t1.spread>0 AND t1.spread<=0.05
      AND ABS(t1.mid - t2.mid) > 0.05
    ORDER BY t1.timestamp
""")
raw_events = cur.fetchall()
print(f"  Raw big-move ticks: {len(raw_events)}")

# Deduplicate: one event per token per 120s window
events = []
last_event = {}
for tid, ts, mid, sp, prev, title in raw_events:
    if ts - last_event.get(tid, 0) < 120:
        continue
    last_event[tid] = ts
    delta = mid - prev
    direction = "UP" if delta > 0 else "DN"
    events.append({
        "tid": tid, "ts": ts, "mid": mid, "spread": sp,
        "prev": prev, "delta": delta, "dir": direction,
        "title": title,
    })

print(f"  Deduped events: {len(events)}")

if not events:
    print("NO EVENTS"); conn.close(); exit()

# Step 2: For each event, simulate entry at different latencies
# Key insight: the move is HAPPENING. We detect it and try to ride it.
latencies = [1, 3, 6, 10]  # seconds after the move tick
hold_times = [30, 60, 120]

print(f"\nSimulating {len(events)} events × {len(latencies)} latencies × {len(hold_times)} holds...")

results = {}  # (latency, hold) → list of r_pct
for lat in latencies:
    for hold in hold_times:
        trades = []
        for ev in events:
            # Entry: we arrive 'lat' seconds after the big-move tick
            cur.execute("""
                SELECT mid, spread FROM ticks WHERE token_id=? 
                AND timestamp BETWEEN ? AND ?
                ORDER BY ABS(timestamp-?) LIMIT 1
            """, (ev["tid"], ev["ts"]+lat-2, ev["ts"]+lat+2, ev["ts"]+lat))
            entry_r = cur.fetchone()
            if not entry_r or entry_r[0] <= 0:
                continue
            entry_mid, entry_sp = entry_r
            
            # Buy at ask (realistic slippage)
            entry_ask = entry_mid + entry_sp / 2
            
            # Direction: buy if move was UP, sell if move was DN
            # For simplicity, always buy the token that moved up
            if ev["dir"] == "DN":
                continue  # skip down moves for now (can't short on poly)
            
            # Exit: at hold time
            exit_ts = ev["ts"] + lat + hold
            cur.execute("""
                SELECT mid, spread FROM ticks WHERE token_id=?
                AND timestamp BETWEEN ? AND ?
                ORDER BY ABS(timestamp-?) LIMIT 1
            """, (ev["tid"], exit_ts-5, exit_ts+5, exit_ts))
            exit_r = cur.fetchone()
            if not exit_r or exit_r[0] <= 0:
                continue
            exit_mid, exit_sp = exit_r
            
            # Sell at bid (realistic slippage)
            exit_bid = exit_mid - exit_sp / 2
            
            pnl = exit_bid - entry_ask
            r_pct = pnl / entry_ask if entry_ask > 0 else 0
            
            trades.append({
                "r_pct": r_pct, "pnl": pnl,
                "entry": entry_ask, "exit": exit_bid,
                "ev_delta": ev["delta"], "title": ev["title"],
                "ts": ev["ts"],
            })
        
        results[(lat, hold)] = trades

# Step 3: Report
print(f"\n{'='*70}")
print(f"RESULTS BY LATENCY × HOLD TIME (after slippage)")
print(f"{'='*70}")
print(f"{'Latency':>8} {'Hold':>6} | {'N':>5} {'Avg R':>8} {'WR':>6} {'Kelly':>7} | {'$100 QK':>12} {'DD':>6}")
print("-" * 70)

best_combo = None
best_eq = 0

for lat in latencies:
    for hold in hold_times:
        trades = results[(lat, hold)]
        if not trades:
            continue
        
        all_r = [t["r_pct"] for t in trades]
        avg_r = sum(all_r) / len(all_r)
        wins = [t for t in trades if t["r_pct"] > 0.001]
        losses = [t for t in trades if t["r_pct"] < -0.001]
        
        wr = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0
        
        # Kelly
        kelly = 0
        if wins and losses:
            p = len(wins) / (len(wins) + len(losses))
            avg_w = sum(t["r_pct"] for t in wins) / len(wins)
            avg_l = abs(sum(t["r_pct"] for t in losses) / len(losses))
            b = avg_w / avg_l if avg_l > 0 else 1
            kelly = max(0, (p * b - (1 - p)) / b)
        
        # Equity (quarter kelly)
        qk = kelly / 4
        eq = 100.0
        eq_hi = 100.0
        max_dd = 0.0
        for t in trades:
            eq += qk * eq * t["r_pct"]
            if eq > eq_hi: eq_hi = eq
            dd = (eq_hi - eq) / eq_hi if eq_hi > 0 else 0
            if dd > max_dd: max_dd = dd
            if eq <= 0: eq = 0; break
        
        if eq > best_eq:
            best_eq = eq
            best_combo = (lat, hold, trades, kelly)
        
        marker = " ◄" if eq > 200 else ""
        print(f"{lat:>6}s {hold:>5}s | {len(trades):>5} {avg_r:>+7.2%} {wr:>5.0%} {kelly:>6.1%} | ${eq:>11,.2f} {max_dd:>5.1%}{marker}")

# Best combo deep dive
if best_combo:
    lat, hold, trades, kelly = best_combo
    print(f"\n{'='*70}")
    print(f"BEST: latency={lat}s hold={hold}s | Kelly={kelly:.2%}")
    print(f"{'='*70}")
    
    all_r = [t["r_pct"] for t in trades]
    wins = [t for t in trades if t["r_pct"] > 0.001]
    losses = [t for t in trades if t["r_pct"] < -0.001]
    
    if wins:
        print(f"  Avg win:  {sum(t['r_pct'] for t in wins)/len(wins):+.2%}")
    if losses:
        print(f"  Avg loss: {sum(t['r_pct'] for t in losses)/len(losses):+.2%}")
    
    # Full equity breakdown
    for frac, label in [(kelly, "FULL"), (kelly/2, "HALF"), (kelly/4, "QTR"), (kelly/8, "8TH")]:
        if frac <= 0: continue
        eq = 100.0
        eq_hi = 100.0
        max_dd = 0.0
        for t in trades:
            eq += frac * eq * t["r_pct"]
            if eq > eq_hi: eq_hi = eq
            dd = (eq_hi - eq) / eq_hi if eq_hi > 0 else 0
            if dd > max_dd: max_dd = dd
            if eq <= 0: eq = 0; break
        print(f"  {label:>4} Kelly ({frac:.2%}): ${eq:>12,.2f} | DD={max_dd:.1%}")
    
    # Monthly
    print(f"\n  Monthly (Quarter Kelly):")
    qk = kelly / 4
    eq = 100.0
    by_month = defaultdict(list)
    for t in trades:
        m = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%Y-%m")
        by_month[m].append(t)
    for m in sorted(by_month.keys()):
        ms = eq
        for t in by_month[m]:
            eq += qk * eq * t["r_pct"]
            if eq <= 0: eq = 0; break
        mr = (eq - ms) / ms if ms > 0 else 0
        print(f"    {m}: {len(by_month[m]):>3} trades | ${eq:>10,.2f} | R={mr:+.1%}")
    
    # Top trades
    print(f"\n  Top 5 winners:")
    for t in sorted(trades, key=lambda x: -x["r_pct"])[:5]:
        dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
        print(f"    {dt} | e={t['entry']:.3f} x={t['exit']:.3f} | R={t['r_pct']:+.2%} | {t['title'][:45]}")
    
    print(f"\n  Top 5 losers:")
    for t in sorted(trades, key=lambda x: x["r_pct"])[:5]:
        dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
        print(f"    {dt} | e={t['entry']:.3f} x={t['exit']:.3f} | R={t['r_pct']:+.2%} | {t['title'][:45]}")

conn.close()
print(f"\n{'='*70}")
print("DONE")
