"""
IPL Cricket Corrected Backtest — Ride Big Moves with Slippage

Cricket-specific: looks for big price moves (>5c) with tight spreads (<3c),
enters at ask, exits at bid. Tests if tight cricket spreads make this profitable
where tennis failed.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

conn = sqlite3.connect("sports_data/tick_history.db")
cur = conn.cursor()

print("=" * 70)
print("IPL CRICKET — CORRECTED BACKTEST WITH SLIPPAGE")
print("=" * 70)

# Get all cricket tokens
cur.execute("""
    SELECT token_id, market_title FROM token_labels
    WHERE sport='cricket' AND (market_title LIKE '%Indian Premier%' OR market_title LIKE '%IPL%')
""")
tokens = cur.fetchall()
print(f"  IPL tokens: {len(tokens)}")

# For each token, load all ticks and find tradeable big moves
all_trades = []

for tid, title in tokens:
    cur.execute("SELECT timestamp, mid, spread FROM ticks WHERE token_id=? ORDER BY timestamp", (tid,))
    ticks = cur.fetchall()
    if len(ticks) < 20:
        continue
    
    # Find big moves with tight spreads
    i = 1
    while i < len(ticks) - 5:
        ts, mid, sp = ticks[i]
        prev_ts, prev_mid, prev_sp = ticks[i-1]
        delta = mid - prev_mid
        
        # Need >5c move AND tight spread at entry
        if abs(delta) < 0.05 or sp > 0.03 or sp <= 0:
            i += 1
            continue
        
        # Direction: buy on UP move (can't short on Poly)
        if delta < 0:
            i += 1
            continue
        
        # Entry at ASK (realistic)
        entry_ask = mid + sp / 2
        
        # Track forward for exit
        for hold in [30, 60, 120, 300]:
            # Find tick closest to hold time
            exit_tick = None
            for j in range(i+1, min(i+100, len(ticks))):
                jts = ticks[j][0]
                if jts - ts >= hold - 10:
                    exit_tick = ticks[j]
                    break
            
            if not exit_tick:
                continue
            
            exit_mid, exit_sp = exit_tick[1], exit_tick[2]
            exit_bid = exit_mid - (exit_sp / 2 if exit_sp > 0 and exit_sp < 1 else sp / 2)
            
            pnl = exit_bid - entry_ask
            r_pct = pnl / entry_ask if entry_ask > 0 else 0
            
            all_trades.append({
                "ts": ts, "tid": tid, "title": title,
                "entry_mid": mid, "entry_ask": entry_ask,
                "exit_mid": exit_mid, "exit_bid": exit_bid,
                "spread": sp, "delta": delta,
                "pnl": pnl, "r_pct": r_pct,
                "hold": hold,
            })
        
        # Skip ahead to avoid duplicate entries
        while i < len(ticks) and ticks[i][0] < ts + 60:
            i += 1
        i += 1

print(f"  Total trade observations: {len(all_trades)}")

if not all_trades:
    print("NO TRADES"); conn.close(); exit()

# Report by hold time
for hold in [30, 60, 120, 300]:
    trades = [t for t in all_trades if t["hold"] == hold]
    if not trades:
        continue
    
    wins = [t for t in trades if t["r_pct"] > 0.001]
    losses = [t for t in trades if t["r_pct"] < -0.001]
    flat = [t for t in trades if abs(t["r_pct"]) <= 0.001]
    all_r = [t["r_pct"] for t in trades]
    avg_r = sum(all_r) / len(all_r)
    avg_sp = sum(t["spread"] for t in trades) / len(trades)
    avg_sp_pct = sum(t["spread"]/t["entry_mid"] for t in trades) / len(trades)
    
    # Kelly
    kelly = 0
    if wins and losses:
        p = len(wins) / (len(wins) + len(losses))
        avg_w = sum(t["r_pct"] for t in wins) / len(wins)
        avg_l = abs(sum(t["r_pct"] for t in losses) / len(losses))
        b = avg_w / avg_l if avg_l > 0 else 1
        kelly = max(0, (p * b - (1 - p)) / b)
    
    print(f"\n{'='*70}")
    print(f"HOLD={hold}s | n={len(trades)} | spread={avg_sp:.3f} ({avg_sp_pct:.0%} of price)")
    print(f"{'='*70}")
    print(f"  Wins: {len(wins)} ({len(wins)/len(trades)*100:.0f}%)")
    print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.0f}%)")
    print(f"  Avg R: {avg_r:+.2%}")
    if wins: print(f"  Avg win: {sum(t['r_pct'] for t in wins)/len(wins):+.2%}")
    if losses: print(f"  Avg loss: {sum(t['r_pct'] for t in losses)/len(losses):+.2%}")
    print(f"  Kelly: {kelly:.2%}")
    
    if kelly > 0:
        # $100 equity
        for frac, label in [(kelly, "FULL"), (kelly/2, "HALF"), (kelly/4, "QTR")]:
            eq = 100.0
            eq_hi = 100.0
            max_dd = 0.0
            for t in trades:
                eq += frac * eq * t["r_pct"]
                if eq > eq_hi: eq_hi = eq
                dd = (eq_hi - eq) / eq_hi if eq_hi > 0 else 0
                if dd > max_dd: max_dd = dd
                if eq <= 0: eq = 0; break
            print(f"  {label} Kelly ({frac:.2%}): $100 → ${eq:,.2f} | DD={max_dd:.0%}")
        
        # Top trades
        print(f"\n  Top 3 winners:")
        for t in sorted(trades, key=lambda x: -x["r_pct"])[:3]:
            dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
            print(f"    {dt} | ask={t['entry_ask']:.3f} bid={t['exit_bid']:.3f} sp={t['spread']:.3f} R={t['r_pct']:+.2%} | {t['title'][:45]}")
        print(f"  Top 3 losers:")
        for t in sorted(trades, key=lambda x: x["r_pct"])[:3]:
            dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
            print(f"    {dt} | ask={t['entry_ask']:.3f} bid={t['exit_bid']:.3f} sp={t['spread']:.3f} R={t['r_pct']:+.2%} | {t['title'][:45]}")
        
        # By match
        print(f"\n  By match:")
        by_match = defaultdict(list)
        for t in trades:
            by_match[t["title"][:50]].append(t)
        for match, mt in sorted(by_match.items(), key=lambda x: -sum(t["r_pct"] for t in x[1])):
            avg = sum(t["r_pct"] for t in mt) / len(mt)
            total = sum(t["pnl"] for t in mt)
            print(f"    {match}: n={len(mt)} avg_R={avg:+.2%} total_pnl={total:+.3f}")

# Also test with lower threshold (>3c moves instead of >5c)
print(f"\n{'='*70}")
print("SENSITIVITY: >3c moves instead of >5c")
print(f"{'='*70}")

all_trades_3c = []
for tid, title in tokens:
    cur.execute("SELECT timestamp, mid, spread FROM ticks WHERE token_id=? ORDER BY timestamp", (tid,))
    ticks = cur.fetchall()
    if len(ticks) < 20:
        continue
    i = 1
    while i < len(ticks) - 5:
        ts, mid, sp = ticks[i]
        prev_ts, prev_mid, _ = ticks[i-1]
        delta = mid - prev_mid
        if delta < 0.03 or sp > 0.03 or sp <= 0:
            i += 1; continue
        entry_ask = mid + sp / 2
        for j in range(i+1, min(i+100, len(ticks))):
            if ticks[j][0] - ts >= 50:
                exit_mid, exit_sp = ticks[j][1], ticks[j][2]
                exit_bid = exit_mid - (exit_sp/2 if 0 < exit_sp < 1 else sp/2)
                r = (exit_bid - entry_ask) / entry_ask
                all_trades_3c.append(r)
                break
        while i < len(ticks) and ticks[i][0] < ts + 60:
            i += 1
        i += 1

if all_trades_3c:
    avg = sum(all_trades_3c)/len(all_trades_3c)
    wins3 = [r for r in all_trades_3c if r > 0.001]
    losses3 = [r for r in all_trades_3c if r < -0.001]
    wr = len(wins3)/(len(wins3)+len(losses3)) if (wins3 or losses3) else 0
    kelly3 = 0
    if wins3 and losses3:
        p = len(wins3)/(len(wins3)+len(losses3))
        aw = sum(wins3)/len(wins3)
        al = abs(sum(losses3)/len(losses3))
        b = aw/al if al > 0 else 1
        kelly3 = max(0, (p*b-(1-p))/b)
    print(f"  n={len(all_trades_3c)} avg_R={avg:+.2%} wr={wr:.0%} kelly={kelly3:.2%}")

conn.close()
print(f"\n{'='*70}")
print("DONE")
