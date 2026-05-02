"""
Limit-Order Market Making Backtest — $0.60-0.80 Tennis

Strategy: Post bid at best_bid. When filled (price dips to our level),
hold and sell at ask when price recovers. Collect the spread.

Realistic:
  - Entry at bid (limit fill)
  - Exit at ask (limit sell) or mid-spread at timeout
  - Max 5 concurrent positions
  - No overlapping same-token positions
  - 300s max hold
"""
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

conn = sqlite3.connect("sports_data/tick_history.db")
cur = conn.cursor()

LO = 0.60
HI = 0.80
MAX_HOLD = 300
MAX_CONCURRENT = 5

print("=" * 70)
print(f"LIMIT ORDER MARKET MAKING — [{LO},{HI})")
print("=" * 70)

# Get all ticks in range, ordered by time
cur.execute("""
    SELECT t.token_id, t.timestamp, t.mid, t.spread, tl.market_title
    FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id
    WHERE tl.sport='tennis' AND t.mid>=? AND t.mid<? AND t.spread>0
    ORDER BY t.timestamp ASC
""", (LO, HI))
all_ticks = cur.fetchall()
print(f"  Ticks in range: {len(all_ticks):,}")

# Group by token
by_token = defaultdict(list)
for tid, ts, mid, sp, title in all_ticks:
    by_token[tid].append((ts, mid, sp, title))
print(f"  Unique tokens: {len(by_token)}")

# Simulate: at each tick, post a bid. Check if price dips to fill us.
trades = []
active_tokens = set()

for tid, ticks in by_token.items():
    if len(ticks) < 20:
        continue
    
    i = 0
    while i < len(ticks) - 10:
        ts, mid, sp, title = ticks[i]
        
        if tid in active_tokens:
            i += 1
            continue
        
        our_bid = mid - sp / 2  # post at best bid
        
        # Look forward: does price dip to our bid?
        filled = False
        fill_ts = 0
        for j in range(i + 1, min(i + 60, len(ticks))):  # check next ~60 ticks
            jts, jmid, jsp, _ = ticks[j]
            if jts - ts > 120:  # order expires after 120s
                break
            if jmid <= our_bid:
                filled = True
                fill_ts = jts
                # Now we own at our_bid. Track forward for exit.
                entry = our_bid
                peak = jmid
                exit_price = jmid
                exit_reason = "TIMEOUT"
                exit_ts = jts + MAX_HOLD
                
                for k in range(j + 1, min(j + 120, len(ticks))):
                    kts, kmid, ksp, _ = ticks[k]
                    elapsed = kts - jts
                    if kmid > peak:
                        peak = kmid
                    
                    # Exit: price reaches our ask (entry + spread)
                    our_ask = entry + sp
                    if kmid >= our_ask:
                        exit_price = our_ask  # limit sell at ask
                        exit_reason = "SPREAD_CAPTURED"
                        exit_ts = kts
                        break
                    
                    # Exit: price drops 3c from entry (stop loss)
                    if entry - kmid >= 0.03:
                        exit_price = kmid - ksp/2  # market sell at bid
                        exit_reason = "STOP_LOSS"
                        exit_ts = kts
                        break
                    
                    # Timeout
                    if elapsed >= MAX_HOLD:
                        exit_price = kmid  # exit at mid (conservative)
                        exit_reason = "TIMEOUT"
                        exit_ts = kts
                        break
                
                pnl = exit_price - entry
                r_pct = pnl / entry if entry > 0 else 0
                hold_s = exit_ts - jts
                
                trades.append({
                    "ts": jts, "token": tid, "title": title,
                    "entry": entry, "exit": exit_price, "spread": sp,
                    "pnl": pnl, "r_pct": r_pct,
                    "reason": exit_reason, "hold_s": hold_s,
                })
                
                # Skip ahead past exit
                while i < len(ticks) and ticks[i][0] < exit_ts:
                    i += 1
                break
        
        if not filled:
            i += 3  # skip a few ticks before trying again

print(f"  Trades: {len(trades)}")

if not trades:
    print("NO TRADES"); conn.close(); exit()

# Stats
wins = [t for t in trades if t["r_pct"] > 0.001]
losses = [t for t in trades if t["r_pct"] < -0.001]
flat = [t for t in trades if abs(t["r_pct"]) <= 0.001]
all_r = [t["r_pct"] for t in trades]
avg_r = sum(all_r) / len(all_r)

print(f"\n{'='*70}")
print(f"STATS ({len(trades)} trades)")
print(f"{'='*70}")
print(f"  Wins:   {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
print(f"  Flat:   {len(flat)} ({len(flat)/len(trades)*100:.1f}%)")
print(f"  Avg R:  {avg_r:+.4%}")
print(f"  Avg spread: {sum(t['spread'] for t in trades)/len(trades):.4f}")
if wins: print(f"  Avg win R:  {sum(t['r_pct'] for t in wins)/len(wins):+.4%}")
if losses: print(f"  Avg loss R: {sum(t['r_pct'] for t in losses)/len(losses):+.4%}")

by_reason = defaultdict(list)
for t in trades:
    by_reason[t["reason"]].append(t["r_pct"])
print(f"\n  By exit:")
for reason, rs in sorted(by_reason.items()):
    avg = sum(rs)/len(rs)
    wr = sum(1 for r in rs if r > 0)/len(rs)
    print(f"    {reason:>18}: n={len(rs):>5} avg_R={avg:+.4%} wr={wr:.0%}")

# Kelly
if wins and losses:
    p = len(wins) / (len(wins) + len(losses))
    avg_w = sum(t["r_pct"] for t in wins) / len(wins)
    avg_l = abs(sum(t["r_pct"] for t in losses) / len(losses))
    b = avg_w / avg_l if avg_l > 0 else 1
    kelly = max(0, (p * b - (1 - p)) / b)
    
    print(f"\n{'='*70}")
    print(f"KELLY")
    print(f"{'='*70}")
    print(f"  Win rate:     {p:.2%}")
    print(f"  Avg win:      {avg_w:+.4%}")
    print(f"  Avg loss:     {avg_l:+.4%}")
    print(f"  W/L ratio:    {b:.2f}")
    print(f"  Full Kelly:   {kelly:.2%}")
    print(f"  Half Kelly:   {kelly/2:.2%}")
    print(f"  Quarter Kelly:{kelly/4:.2%}")

    # Equity
    print(f"\n{'='*70}")
    print(f"$100 EQUITY CURVE")
    print(f"{'='*70}")
    for frac, label in [(kelly, "FULL"), (kelly/2, "HALF"), (kelly/4, "QUARTER"), (kelly/8, "EIGHTH")]:
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
        print(f"  {label:>7} ({frac:.2%}): ${eq:>15,.2f} | DD={max_dd:.1%}")

    # Monthly
    print(f"\n{'='*70}")
    print(f"MONTHLY (Half Kelly)")
    print(f"{'='*70}")
    hk = kelly / 2
    eq = 100.0
    by_month = defaultdict(list)
    for t in trades:
        m = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%Y-%m")
        by_month[m].append(t)
    for m in sorted(by_month.keys()):
        ms = eq
        for t in by_month[m]:
            eq += hk * eq * t["r_pct"]
            if eq <= 0: eq = 0; break
        mr = (eq - ms) / ms if ms > 0 else 0
        n = len(by_month[m])
        wr = sum(1 for t in by_month[m] if t["r_pct"] > 0) / n if n else 0
        print(f"  {m}: {n:>5} | ${eq:>12,.2f} | R={mr:+.1%} | wr={wr:.0%}")

# Top trades
print(f"\n{'='*70}")
print(f"TOP 5 WINNERS")
print(f"{'='*70}")
for t in sorted(trades, key=lambda x: -x["r_pct"])[:5]:
    dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
    print(f"  {dt} | e={t['entry']:.3f} x={t['exit']:.3f} sp={t['spread']:.3f} | R={t['r_pct']:+.2%} {t['reason']:>18} | {t['title'][:40]}")

print(f"\nTOP 5 LOSERS")
for t in sorted(trades, key=lambda x: x["r_pct"])[:5]:
    dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
    print(f"  {dt} | e={t['entry']:.3f} x={t['exit']:.3f} sp={t['spread']:.3f} | R={t['r_pct']:+.2%} {t['reason']:>18} | {t['title'][:40]}")

conn.close()
print(f"\n{'='*70}")
print("DONE")
