"""
Cricket Mean Reversion — FADE the Spike (Fixed Pairing)

After a big UP move on token A, buy token B (the crashed side).
Ride the 88% mean reversion. With slippage.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

conn = sqlite3.connect("sports_data/tick_history.db")
cur = conn.cursor()

print("=" * 70)
print("CRICKET MEAN REVERSION — FADE THE SPIKE")
print("=" * 70)

# Get all IPL tokens, pair by exact title
cur.execute("""
    SELECT token_id, market_title FROM token_labels
    WHERE sport='cricket' AND market_title LIKE '%Indian Premier%'
    ORDER BY market_title, token_id
""")
tokens = cur.fetchall()

# Group by exact title
by_title = defaultdict(list)
for tid, title in tokens:
    by_title[title].append(tid)

# Build pairs: same title = same match, 2 tokens
pairs = []
for title, tids in by_title.items():
    if len(tids) == 2:
        pairs.append((tids[0], tids[1], title))

print(f"  Token pairs: {len(pairs)}")

HOLD_TIMES = [30, 60, 120, 300]
all_trades = []

for tid_a, tid_b, match_title in pairs:
    # Load both token tick series
    cur.execute("SELECT timestamp, mid, spread FROM ticks WHERE token_id=? ORDER BY timestamp", (tid_a,))
    ticks_a = cur.fetchall()
    cur.execute("SELECT timestamp, mid, spread FROM ticks WHERE token_id=? ORDER BY timestamp", (tid_b,))
    ticks_b = cur.fetchall()
    
    if len(ticks_a) < 20 or len(ticks_b) < 20:
        continue
    
    # We need to detect spikes on EITHER token and trade the OTHER.
    # Process both directions: A spikes → buy B, and B spikes → buy A.
    for spike_ticks, fade_ticks, spike_label, fade_label in [
        (ticks_a, ticks_b, "A", "B"),
        (ticks_b, ticks_a, "B", "A"),
    ]:
        # Index fade ticks by timestamp for lookup
        fade_idx = {}
        for ts, mid, sp in fade_ticks:
            fade_idx[int(ts)] = (mid, sp)
        
        last_trade_ts = 0
        for i in range(1, len(spike_ticks) - 1):
            ts, mid, sp = spike_ticks[i]
            prev_ts, prev_mid, _ = spike_ticks[i-1]
            delta = mid - prev_mid
            
            # Big UP move on spike token (= big DOWN on fade token)
            if delta < 0.05:
                continue
            
            # Dedup 120s
            if ts - last_trade_ts < 120:
                continue
            
            # Find fade token's current price
            fade_price = None
            for offset in range(-15, 16):
                key = int(ts) + offset
                if key in fade_idx:
                    fade_price = fade_idx[key]
                    break
            
            if not fade_price:
                continue
            
            fade_mid, fade_sp = fade_price
            
            # Skip if spread too wide or no spread
            if fade_sp <= 0 or fade_sp >= 0.50:
                continue
            
            # Skip if fade token is already near zero (match ending)
            if fade_mid < 0.05:
                continue
            
            # BUY fade token at ASK
            entry_ask = fade_mid + fade_sp / 2
            last_trade_ts = ts
            
            # Check exits at various hold times
            for hold in HOLD_TIMES:
                target_ts = ts + hold
                exit_price = None
                for offset in range(-15, 16):
                    key = int(target_ts) + offset
                    if key in fade_idx:
                        exit_price = fade_idx[key]
                        break
                
                if not exit_price:
                    continue
                
                exit_mid, exit_sp = exit_price
                exit_bid = exit_mid - (exit_sp / 2 if 0 < exit_sp < 0.50 else fade_sp / 2)
                
                pnl = exit_bid - entry_ask
                r_pct = pnl / entry_ask if entry_ask > 0 else 0
                
                all_trades.append({
                    "ts": ts, "match": match_title,
                    "spike_delta": delta,
                    "fade_entry_mid": fade_mid, "fade_entry_ask": entry_ask,
                    "fade_exit_mid": exit_mid, "fade_exit_bid": exit_bid,
                    "spread": fade_sp, "pnl": pnl, "r_pct": r_pct,
                    "hold": hold,
                })

print(f"  Total trade observations: {len(all_trades)}")

if not all_trades:
    print("NO TRADES"); conn.close(); exit()

for hold in HOLD_TIMES:
    trades = sorted([t for t in all_trades if t["hold"] == hold], key=lambda x: x["ts"])
    if not trades:
        continue
    
    wins = [t for t in trades if t["r_pct"] > 0.001]
    losses = [t for t in trades if t["r_pct"] < -0.001]
    all_r = [t["r_pct"] for t in trades]
    avg_r = sum(all_r) / len(all_r)
    avg_sp = sum(t["spread"] for t in trades) / len(trades)
    wr = len(wins) / (len(wins)+len(losses)) if (wins or losses) else 0
    
    kelly = 0
    if wins and losses:
        p = len(wins) / (len(wins) + len(losses))
        avg_w = sum(t["r_pct"] for t in wins) / len(wins)
        avg_l = abs(sum(t["r_pct"] for t in losses) / len(losses))
        b = avg_w / avg_l if avg_l > 0 else 1
        kelly = max(0, (p * b - (1-p)) / b)
    
    print(f"\n{'='*70}")
    print(f"HOLD={hold}s | n={len(trades)} | avg_spread={avg_sp:.3f}")
    print(f"{'='*70}")
    print(f"  Wins: {len(wins)} | Losses: {len(losses)} | WR: {wr:.0%}")
    print(f"  Avg R:    {avg_r:+.2%}")
    if wins: print(f"  Avg win:  {sum(t['r_pct'] for t in wins)/len(wins):+.2%}")
    if losses: print(f"  Avg loss: {sum(t['r_pct'] for t in losses)/len(losses):+.2%}")
    print(f"  Kelly:    {kelly:.2%}")
    
    if kelly > 0:
        for frac, label in [(kelly, "FULL"), (kelly/2, "HALF"), (kelly/4, "QTR"), (kelly/8, "8TH")]:
            eq = 100.0; eq_hi = 100.0; max_dd = 0.0
            for t in trades:
                eq += frac * eq * t["r_pct"]
                if eq > eq_hi: eq_hi = eq
                dd = (eq_hi - eq) / eq_hi if eq_hi > 0 else 0
                if dd > max_dd: max_dd = dd
                if eq <= 0: eq = 0; break
            print(f"  {label:>4} ({frac:.2%}): $100 → ${eq:>12,.2f} | DD={max_dd:.0%}")
        
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
            mr = (eq-ms)/ms if ms > 0 else 0
            n = len(by_month[m])
            print(f"    {m}: {n:>3} trades | ${eq:>10,.2f} | R={mr:+.1%}")

    # Top trades
    print(f"\n  Top 5 winners:")
    for t in sorted(trades, key=lambda x: -x["r_pct"])[:5]:
        dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
        print(f"    {dt} | ask={t['fade_entry_ask']:.3f} bid={t['fade_exit_bid']:.3f} sp={t['spread']:.3f} R={t['r_pct']:+.2%} spike={t['spike_delta']:+.3f} | {t['match'][:45]}")
    print(f"  Top 5 losers:")
    for t in sorted(trades, key=lambda x: x["r_pct"])[:5]:
        dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
        print(f"    {dt} | ask={t['fade_entry_ask']:.3f} bid={t['fade_exit_bid']:.3f} sp={t['spread']:.3f} R={t['r_pct']:+.2%} spike={t['spike_delta']:+.3f} | {t['match'][:45]}")

conn.close()
print(f"\n{'='*70}")
print("DONE")
