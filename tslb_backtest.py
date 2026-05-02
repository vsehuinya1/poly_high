"""
TSLB Historical Backtest — Kelly Compounding from $100

Replays TSLB strategy (price<0.20, spread<=0.03, LONG) across all
historical tennis ticks. Computes Kelly fraction and equity curve.
"""
import sqlite3
import math
from collections import defaultdict
from datetime import datetime, timezone

DB = "sports_data/tick_history.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

# Strategy params (same as tennis/alpha.py)
MAX_PRICE = 0.20
MAX_SPREAD = 0.02
MIN_PRICE = 0.03
DEDUP_S = 90.0
HOLD_S = 300.0
TRAILING_STOP = 0.02

print("=" * 70)
print("TSLB HISTORICAL BACKTEST — Kelly Compounding")
print("=" * 70)

# Step 1: Get all tennis tokens with LB ticks
print("\nLoading lower-band tennis ticks...")
cur.execute("""
    SELECT t.token_id, t.timestamp, t.mid, t.spread, tl.market_title
    FROM ticks t
    JOIN token_labels tl ON t.token_id = tl.token_id
    WHERE tl.sport = 'tennis'
      AND t.mid > ? AND t.mid < ?
      AND t.spread > 0 AND t.spread <= ?
    ORDER BY t.timestamp ASC
""", (MIN_PRICE, MAX_PRICE, MAX_SPREAD))
candidates = cur.fetchall()
print(f"  Found {len(candidates):,} entry candidates")

# Step 2: Simulate trades
print("\nSimulating trades...")
last_entry = {}  # token_id → last entry timestamp
trades = []

for tid, ts, mid, spread, title in candidates:
    # Dedup
    if ts - last_entry.get(tid, 0) < DEDUP_S:
        continue
    last_entry[tid] = ts

    # Get forward price path
    cur.execute("""
        SELECT timestamp - ?, mid FROM ticks
        WHERE token_id = ? AND timestamp > ? AND timestamp <= ?
        ORDER BY timestamp
    """, (ts, tid, ts, ts + HOLD_S + 30))
    path = cur.fetchall()
    if len(path) < 3:
        continue

    # Simulate exit
    entry = mid
    peak = entry
    exit_price = entry
    exit_reason = "TIMEOUT"
    exit_elapsed = HOLD_S

    for dt, price in path:
        if price > peak:
            peak = price
        # Trailing stop
        if peak > entry and peak - price >= TRAILING_STOP:
            exit_price = price
            exit_reason = "TRAIL"
            exit_elapsed = dt
            break
        # 2R hit
        if price >= entry * 2:
            exit_price = price
            exit_reason = "2R"
            exit_elapsed = dt
            break
        # Timeout
        if dt >= HOLD_S:
            exit_price = price
            exit_reason = "TIMEOUT"
            exit_elapsed = dt
            break

    pnl = exit_price - entry
    r_pct = pnl / entry if entry > 0 else 0

    trades.append({
        "ts": ts, "token": tid, "title": title,
        "entry": entry, "exit": exit_price, "peak": peak,
        "pnl": pnl, "r_pct": r_pct,
        "reason": exit_reason, "hold_s": exit_elapsed,
    })

print(f"  Completed {len(trades)} trades")

if not trades:
    print("NO TRADES — cannot compute Kelly")
    conn.close()
    exit()

# Step 3: PnL distribution analysis
wins = [t for t in trades if t["pnl"] > 0.001]
losses = [t for t in trades if t["pnl"] < -0.001]
flat = [t for t in trades if abs(t["pnl"]) <= 0.001]

print(f"\n{'='*70}")
print(f"TRADE STATISTICS ({len(trades)} trades)")
print(f"{'='*70}")
print(f"  Wins:   {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
print(f"  Flat:   {len(flat)} ({len(flat)/len(trades)*100:.1f}%)")

all_pnl = [t["pnl"] for t in trades]
all_r = [t["r_pct"] for t in trades]
avg_pnl = sum(all_pnl) / len(all_pnl)
total_pnl = sum(all_pnl)
avg_r = sum(all_r) / len(all_r)

print(f"\n  Avg PnL per trade: {avg_pnl:+.4f}c")
print(f"  Total PnL (sum):  {total_pnl:+.4f}c")
print(f"  Avg R per trade:  {avg_r:+.2%}")
print(f"  Max win:  {max(all_r):+.2%}")
print(f"  Max loss: {min(all_r):+.2%}")

if wins:
    avg_win_r = sum(t["r_pct"] for t in wins) / len(wins)
    print(f"  Avg win R:  {avg_win_r:+.2%}")
if losses:
    avg_loss_r = sum(t["r_pct"] for t in losses) / len(losses)
    print(f"  Avg loss R: {avg_loss_r:+.2%}")

# By exit reason
print(f"\n  By exit reason:")
by_reason = defaultdict(list)
for t in trades:
    by_reason[t["reason"]].append(t["r_pct"])
for reason, rs in sorted(by_reason.items()):
    avg = sum(rs) / len(rs)
    wr = sum(1 for r in rs if r > 0) / len(rs)
    print(f"    {reason:>8s}: n={len(rs):>4} avg_R={avg:+.2%} wr={wr:.0%}")

# Fat tails
print(f"\n  Fat tails:")
print(f"    Hit 2R:   {sum(1 for t in trades if t['r_pct'] >= 1.0)}/{len(trades)}")
print(f"    Hit 50%+: {sum(1 for t in trades if t['r_pct'] >= 0.50)}/{len(trades)}")
print(f"    Hit 20%+: {sum(1 for t in trades if t['r_pct'] >= 0.20)}/{len(trades)}")
print(f"    Drop>20%: {sum(1 for t in trades if t['r_pct'] <= -0.20)}/{len(trades)}")

# Step 4: Kelly Criterion
# Kelly = (p * b - q) / b
# where p = win probability, b = avg_win/avg_loss ratio, q = 1-p
if wins and losses:
    p = len(wins) / (len(wins) + len(losses))
    q = 1 - p
    avg_w = sum(t["r_pct"] for t in wins) / len(wins)
    avg_l = abs(sum(t["r_pct"] for t in losses) / len(losses))
    b = avg_w / avg_l if avg_l > 0 else 1
    kelly = (p * b - q) / b if b > 0 else 0
    kelly = max(0, kelly)
    
    print(f"\n{'='*70}")
    print(f"KELLY CRITERION")
    print(f"{'='*70}")
    print(f"  Win rate (p):     {p:.2%}")
    print(f"  Avg win:          {avg_w:+.2%}")
    print(f"  Avg loss:         {avg_l:+.2%}")
    print(f"  Win/Loss ratio:   {b:.2f}")
    print(f"  Full Kelly:       {kelly:.2%}")
    print(f"  Half Kelly:       {kelly/2:.2%}")
    print(f"  Quarter Kelly:    {kelly/4:.2%}")

    # Step 5: Equity curve with Kelly compounding
    print(f"\n{'='*70}")
    print(f"EQUITY CURVE — $100 start")
    print(f"{'='*70}")
    
    for kelly_frac, label in [(kelly, "FULL KELLY"), (kelly/2, "HALF KELLY"), (kelly/4, "QUARTER KELLY")]:
        if kelly_frac <= 0:
            continue
        equity = 100.0
        max_equity = 100.0
        max_dd = 0.0
        equity_high = 100.0
        
        for t in trades:
            # Size = kelly fraction of equity
            # On this bet: risk = kelly_frac * equity
            # PnL = risk * r_pct (since r_pct is return on capital at risk)
            bet = kelly_frac * equity
            trade_pnl = bet * t["r_pct"]
            equity += trade_pnl
            
            if equity > equity_high:
                equity_high = equity
            dd = (equity_high - equity) / equity_high
            if dd > max_dd:
                max_dd = dd
            
            if equity <= 0:
                equity = 0
                break
        
        total_r = (equity - 100) / 100
        print(f"\n  {label} ({kelly_frac:.2%}):")
        print(f"    Final equity: ${equity:,.2f}")
        print(f"    Total return: {total_r:+.1%}")
        print(f"    Max drawdown: {max_dd:.1%}")
        print(f"    Peak equity:  ${equity_high:,.2f}")

    # Monthly breakdown
    print(f"\n{'='*70}")
    print(f"MONTHLY BREAKDOWN (Half Kelly)")
    print(f"{'='*70}")
    hk = kelly / 2
    equity = 100.0
    by_month = defaultdict(list)
    for t in trades:
        month = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%Y-%m")
        by_month[month].append(t)
    
    for month in sorted(by_month.keys()):
        month_start = equity
        for t in by_month[month]:
            bet = hk * equity
            equity += bet * t["r_pct"]
            if equity <= 0:
                equity = 0
                break
        month_r = (equity - month_start) / month_start if month_start > 0 else 0
        n = len(by_month[month])
        wr = sum(1 for t in by_month[month] if t["r_pct"] > 0) / n if n else 0
        print(f"  {month}: {n:>4} trades | equity=${equity:>10,.2f} | month_R={month_r:+.1%} | wr={wr:.0%}")

# Top 10 winners
print(f"\n{'='*70}")
print(f"TOP 10 WINNERS")
print(f"{'='*70}")
for t in sorted(trades, key=lambda x: -x["r_pct"])[:10]:
    dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
    print(f"  {dt} | entry={t['entry']:.3f} exit={t['exit']:.3f} | R={t['r_pct']:+.1%} | {t['reason']:>7} | {t['title'][:45]}")

# Top 10 losers
print(f"\n{'='*70}")
print(f"TOP 10 LOSERS")
print(f"{'='*70}")
for t in sorted(trades, key=lambda x: x["r_pct"])[:10]:
    dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
    print(f"  {dt} | entry={t['entry']:.3f} exit={t['exit']:.3f} | R={t['r_pct']:+.1%} | {t['reason']:>7} | {t['title'][:45]}")

conn.close()
print(f"\n{'='*70}")
print(f"BACKTEST COMPLETE")
print(f"{'='*70}")
