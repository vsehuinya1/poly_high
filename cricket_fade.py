"""
Cricket Mean Reversion — FADE the Spike

The momentum backtest showed 88% of big cricket moves reverse.
This is the inverse: after a big UP move, buy the OTHER token
(which just crashed) at the panic low. Ride the mean reversion.

On Polymarket: each match has 2 tokens (Team A, Team B).
When A spikes UP, B drops. We buy B at the panic bid.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

conn = sqlite3.connect("sports_data/tick_history.db")
cur = conn.cursor()

print("=" * 70)
print("CRICKET MEAN REVERSION — FADE THE SPIKE")
print("=" * 70)

# Get all IPL match pairs (2 tokens per match)
cur.execute("""
    SELECT token_id, market_title FROM token_labels
    WHERE sport='cricket' AND (market_title LIKE '%Indian Premier%' OR market_title LIKE '%IPL%')
    ORDER BY market_title
""")
tokens = cur.fetchall()

# Group by match (same title = same match)
matches = defaultdict(list)
for tid, title in tokens:
    # Normalize title to group pairs
    base = title.split(":")[0].strip() if ":" in title else title[:40]
    matches[base].append((tid, title))

print(f"  IPL matches with tokens: {len(matches)}")
pair_count = sum(1 for m in matches.values() if len(m) == 2)
print(f"  Matches with token pairs: {pair_count}")

# For each pair, find spikes on token A and check if token B reverts
all_trades = []
HOLD_TIMES = [30, 60, 120, 300]

for match_name, toks in matches.items():
    if len(toks) != 2:
        continue
    
    # Load ticks for both tokens
    tick_data = {}
    for tid, title in toks:
        cur.execute("SELECT timestamp, mid, spread FROM ticks WHERE token_id=? ORDER BY timestamp", (tid,))
        tick_data[tid] = cur.fetchall()
    
    tid_a, title_a = toks[0]
    tid_b, title_b = toks[1]
    ticks_a = tick_data[tid_a]
    ticks_b = tick_data[tid_b]
    
    if len(ticks_a) < 20 or len(ticks_b) < 20:
        continue
    
    # Index ticks_b by time for fast lookup
    b_by_time = {}
    for ts, mid, sp in ticks_b:
        b_by_time[int(ts)] = (mid, sp)
    
    # Find spikes in token A, then trade token B (the crash side)
    i = 1
    last_trade_ts = 0
    while i < len(ticks_a) - 5:
        ts_a, mid_a, sp_a = ticks_a[i]
        prev_ts_a, prev_mid_a, _ = ticks_a[i-1]
        delta_a = mid_a - prev_mid_a
        
        # Token A spiked UP > 5c → Token B just crashed
        if delta_a < 0.05:
            i += 1; continue
        
        # Dedup: 120s between trades
        if ts_a - last_trade_ts < 120:
            i += 1; continue
        
        # Find token B's price at this moment
        best_b = None
        for offset in range(0, 15):
            key = int(ts_a) + offset
            if key in b_by_time:
                best_b = b_by_time[key]
                break
            key = int(ts_a) - offset
            if key in b_by_time:
                best_b = b_by_time[key]
                break
        
        if not best_b:
            # Linear search fallback
            for ts_b, mid_b, sp_b in ticks_b:
                if abs(ts_b - ts_a) < 30:
                    best_b = (mid_b, sp_b)
                    break
        
        if not best_b or best_b[1] <= 0 or best_b[1] >= 1:
            i += 1; continue
        
        entry_mid_b, entry_sp_b = best_b
        
        # Skip if spread too wide
        if entry_sp_b > 0.05:
            i += 1; continue
        
        # BUY token B at ask (the crashed token — mean reversion play)
        entry_ask_b = entry_mid_b + entry_sp_b / 2
        
        last_trade_ts = ts_a
        
        # Check exits at various hold times
        for hold in HOLD_TIMES:
            exit_b = None
            for ts_b, mid_b, sp_b in ticks_b:
                if abs(ts_b - (ts_a + hold)) < 15:
                    exit_b = (mid_b, sp_b)
                    break
            
            if not exit_b:
                continue
            
            exit_mid_b, exit_sp_b = exit_b
            exit_bid_b = exit_mid_b - (exit_sp_b / 2 if 0 < exit_sp_b < 1 else entry_sp_b / 2)
            
            pnl = exit_bid_b - entry_ask_b
            r_pct = pnl / entry_ask_b if entry_ask_b > 0 else 0
            
            all_trades.append({
                "ts": ts_a, "match": match_name,
                "title_spiked": title_a[:30], "title_bought": title_b[:30],
                "spike_delta": delta_a,
                "entry_mid": entry_mid_b, "entry_ask": entry_ask_b,
                "exit_mid": exit_mid_b, "exit_bid": exit_bid_b,
                "spread": entry_sp_b, "pnl": pnl, "r_pct": r_pct,
                "hold": hold,
            })
        
        i += 1
        while i < len(ticks_a) and ticks_a[i][0] < ts_a + 60:
            i += 1

print(f"  Total trade observations: {len(all_trades)}")

if not all_trades:
    print("NO TRADES"); conn.close(); exit()

# Report by hold time
for hold in HOLD_TIMES:
    trades = [t for t in all_trades if t["hold"] == hold]
    if not trades:
        continue
    
    wins = [t for t in trades if t["r_pct"] > 0.001]
    losses = [t for t in trades if t["r_pct"] < -0.001]
    all_r = [t["r_pct"] for t in trades]
    avg_r = sum(all_r) / len(all_r)
    avg_sp = sum(t["spread"] for t in trades) / len(trades)
    
    kelly = 0
    if wins and losses:
        p = len(wins) / (len(wins) + len(losses))
        avg_w = sum(t["r_pct"] for t in wins) / len(wins)
        avg_l = abs(sum(t["r_pct"] for t in losses) / len(losses))
        b = avg_w / avg_l if avg_l > 0 else 1
        kelly = max(0, (p * b - (1 - p)) / b)
    
    wr = len(wins)/(len(wins)+len(losses)) if (wins or losses) else 0
    
    print(f"\n{'='*70}")
    print(f"HOLD={hold}s | n={len(trades)} | spread={avg_sp:.3f}")
    print(f"{'='*70}")
    print(f"  Wins: {len(wins)} | Losses: {len(losses)} | WR: {wr:.0%}")
    print(f"  Avg R:    {avg_r:+.2%}")
    if wins: print(f"  Avg win:  {sum(t['r_pct'] for t in wins)/len(wins):+.2%}")
    if losses: print(f"  Avg loss: {sum(t['r_pct'] for t in losses)/len(losses):+.2%}")
    print(f"  Kelly:    {kelly:.2%}")
    
    if kelly > 0:
        for frac, label in [(kelly, "FULL"), (kelly/2, "HALF"), (kelly/4, "QTR"), (kelly/8, "8TH")]:
            eq = 100.0; eq_hi = 100.0; max_dd = 0.0
            for t in sorted(trades, key=lambda x: x["ts"]):
                eq += frac * eq * t["r_pct"]
                if eq > eq_hi: eq_hi = eq
                dd = (eq_hi - eq) / eq_hi if eq_hi > 0 else 0
                if dd > max_dd: max_dd = dd
                if eq <= 0: eq = 0; break
            print(f"  {label:>4} ({frac:.2%}): $100 → ${eq:>10,.2f} | DD={max_dd:.0%}")
        
        # Monthly
        print(f"\n  Monthly (Quarter Kelly):")
        qk = kelly / 4
        eq = 100.0
        by_month = defaultdict(list)
        for t in sorted(trades, key=lambda x: x["ts"]):
            m = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%Y-%m")
            by_month[m].append(t)
        for m in sorted(by_month.keys()):
            ms = eq
            for t in by_month[m]:
                eq += qk * eq * t["r_pct"]
                if eq <= 0: eq = 0; break
            mr = (eq-ms)/ms if ms > 0 else 0
            n = len(by_month[m])
            wr_m = sum(1 for t in by_month[m] if t["r_pct"]>0)/n
            print(f"    {m}: {n:>3} trades | ${eq:>10,.2f} | R={mr:+.1%} | wr={wr_m:.0%}")
    
    # Top trades
    print(f"\n  Top 3 winners:")
    for t in sorted(trades, key=lambda x: -x["r_pct"])[:3]:
        dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
        print(f"    {dt} | ask={t['entry_ask']:.3f} bid={t['exit_bid']:.3f} sp={t['spread']:.3f} R={t['r_pct']:+.2%} spike={t['spike_delta']:+.3f} | {t['match'][:40]}")
    print(f"  Top 3 losers:")
    for t in sorted(trades, key=lambda x: x["r_pct"])[:3]:
        dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
        print(f"    {dt} | ask={t['entry_ask']:.3f} bid={t['exit_bid']:.3f} sp={t['spread']:.3f} R={t['r_pct']:+.2%} spike={t['spike_delta']:+.3f} | {t['match'][:40]}")

conn.close()
print(f"\n{'='*70}")
print("DONE")
