"""
Retroactive simulation v2: show filter layers and simulate weekend performance.
"""
import csv, os
from collections import defaultdict, Counter
from datetime import datetime

FILES = [
    'sports_data/paper_trades_20260222.csv',
    'sports_data/paper_trades_20260225.csv',
    'sports_data/paper_trades_20260226.csv',
    'sports_data/paper_trades_20260227.csv',
]

NBA_TEAMS = ['Lakers','Spurs','Heat','Raptors','Bulls','Trail Blazers','Kings',
             'Mavericks','Hornets','Pacers','Wizards','Hawks','Celtics','Nets',
             'Rockets','Magic','76ers','Timberwolves','Clippers','Grizzlies',
             'Warriors','Bucks','Cavaliers','Pistons','Thunder','Jazz','Pelicans',
             'Nuggets','Suns','Knicks']

# Parse all trades
positions = defaultdict(lambda: {'entry': None, 'exit': None})
for fp in FILES:
    if not os.path.exists(fp):
        continue
    with open(fp) as f:
        for row in csv.DictReader(f):
            pid = row['position_id']
            if row['event'] == 'ENTRY':
                positions[pid]['entry'] = row
            elif row['event'] == 'EXIT':
                positions[pid]['exit'] = row

trades = []
for pid, data in positions.items():
    if not (data['entry'] and data['exit']):
        continue
    e, x = data['entry'], data['exit']
    gs = e['game_state']
    sport = 'nba' if any(t in gs for t in NBA_TEAMS) else 'football'
    total_min = 48.0 if sport == 'nba' else 90.0
    elapsed = 0
    if "'" in gs:
        try:
            elapsed = float(gs.split("(")[-1].replace("')", "").replace("'", ""))
        except: pass
    entry_price = float(e['entry_price']) if e['entry_price'] else 0.0
    trades.append({
        'pid': pid, 'sport': sport,
        'pnl': float(x['pnl']) if x['pnl'] else 0.0,
        'entry_price': entry_price,
        'exit_price': float(x['exit_price']) if x['exit_price'] else 0.0,
        'edge': float(e['entry_edge']) if e['entry_edge'] else 0.0,
        'direction': e['direction'],
        'exit_reason': x['exit_reason'],
        'game_state': gs,
        'elapsed': elapsed, 'total_min': total_min,
        'size': float(e['size_usd']) if e['size_usd'] else 0.0,
        'timestamp': float(e['timestamp']),
        'game_id': e['game_id'],
    })
trades.sort(key=lambda t: t['timestamp'])

print("=" * 70)
print("  FILTER LAYER ANALYSIS + DEGEN SIMULATION")
print("=" * 70)

# Show what each filter removes
print(f"\nTotal trades: {len(trades)}")
total_pnl = sum(t['pnl'] for t in trades)
print(f"Total PnL (no filters): ${total_pnl:.2f}")

# Layer 1: Price band [0.20-0.65] on entry price
f1 = [t for t in trades if 0.20 <= t['entry_price'] <= 0.65]
print(f"\nAfter price band [0.20-0.65]: {len(f1)} trades")
print(f"  Removed: {len(trades)-len(f1)} | PnL of removed: ${sum(t['pnl'] for t in trades if t not in f1):.2f}")
print(f"  Remaining PnL: ${sum(t['pnl'] for t in f1):.2f}")

# Layer 2: + edge >= 0.10
f2 = [t for t in f1 if abs(t['edge']) >= 0.10]
print(f"\nAfter + edge >= 0.10: {len(f2)} trades")
print(f"  Removed: {len(f1)-len(f2)} | PnL of removed: ${sum(t['pnl'] for t in f1 if t not in f2):.2f}")
print(f"  Remaining PnL: ${sum(t['pnl'] for t in f2):.2f}")

# Layer 3: + time gate (< 75% elapsed)
f3 = [t for t in f2 if (t['elapsed'] / t['total_min']) <= 0.75]
print(f"\nAfter + time gate (<75%%): {len(f3)} trades")
print(f"  Removed: {len(f2)-len(f3)} | PnL of removed: ${sum(t['pnl'] for t in f2 if t not in f3):.2f}")
print(f"  Remaining PnL: ${sum(t['pnl'] for t in f3):.2f}")

# Layer 4: + per-game limit = 1
f4 = []
game_used = set()
for t in f3:
    if t['game_id'] not in game_used:
        game_used.add(t['game_id'])
        f4.append(t)
print(f"\nAfter + 1 per game: {len(f4)} trades")
print(f"  Removed: {len(f3)-len(f4)} | PnL of removed: ${sum(t['pnl'] for t in f3 if t not in f4):.2f}")
print(f"  Remaining PnL: ${sum(t['pnl'] for t in f4):.2f}")

# Now simulate different degen levels on the BEST filter set
# Use f3 (without per-game limit but WITH time gate) since it has more trades
print("\n" + "=" * 70)
print("  SIMULATION: f3 FILTERS (price + edge + time gate)")
print("  No per-game limit — but only 1st entry per game-direction")
print("=" * 70)

# De-duplicate: only first trade per game+direction
seen = set()
f3_dedup = []
for t in f3:
    key = (t['game_id'], t['direction'])
    if key not in seen:
        seen.add(key)
        f3_dedup.append(t)

print(f"\nTrades after dedup: {len(f3_dedup)}")
wins = [t for t in f3_dedup if t['pnl'] > 0]
losses = [t for t in f3_dedup if t['pnl'] < 0]
print(f"Win rate: {len(wins)/max(1,len(f3_dedup))*100:.1f}% ({len(wins)}W/{len(losses)}L)")
print(f"Net PnL (original sizing): ${sum(t['pnl'] for t in f3_dedup):.2f}")

# Show each trade
print(f"\n--- Trade-by-Trade ---")
for t in f3_dedup:
    dt = datetime.utcfromtimestamp(t['timestamp'])
    marker = "WIN" if t['pnl'] > 0 else "LOSS" if t['pnl'] < 0 else "FLAT"
    print(f"  {dt.strftime('%m/%d %H:%M')} [{marker:4s}] {t['direction']} @ {t['entry_price']:.3f} > "
          f"{t['exit_price']:.3f} | edge={t['edge']:.3f} | ${t['pnl']:+.2f} | "
          f"{t['exit_reason']} | {t['game_state'][:50]}")

# Degen simulations
print(f"\n--- $1000 Starting Balance Simulations ---\n")
for label, pct in [("Conservative (5%)", 0.05), ("Degen (10%)", 0.10), ("Full Degen (20%)", 0.20), ("YOLO (33%)", 0.33)]:
    balance = 1000.0
    peak = 1000.0
    max_dd = 0
    for t in f3_dedup:
        bet = balance * pct
        if bet < 5:
            break
        if t['direction'] == 'BUY':
            ret = (t['exit_price'] - t['entry_price']) / t['entry_price'] if t['entry_price'] > 0 else 0
        else:
            ret = (t['entry_price'] - t['exit_price']) / (1 - t['entry_price']) if t['entry_price'] < 1 else 0
        ret = max(ret, -1.0)
        balance += bet * ret
        if balance > peak: peak = balance
        dd = peak - balance
        if dd > max_dd: max_dd = dd
    print(f"  {label:25s}: ${balance:.2f} ({balance-1000:+.2f}) | MaxDD=${max_dd:.2f}")

# Also show what WOULD have happened with just the profitable buckets
print("\n--- Profitable Bucket Only: entry_price in [0.45-0.65] ---")
sweet = [t for t in trades if 0.45 <= t['entry_price'] <= 0.65]
sweet_dedup = []
seen2 = set()
for t in sorted(sweet, key=lambda x: x['timestamp']):
    key = (t['game_id'], t['direction'])
    if key not in seen2:
        seen2.add(key)
        sweet_dedup.append(t)

print(f"Trades: {len(sweet_dedup)}")
sw = [t for t in sweet_dedup if t['pnl'] > 0]
print(f"Win rate: {len(sw)/max(1,len(sweet_dedup))*100:.1f}%")
print(f"Net PnL (original sizing): ${sum(t['pnl'] for t in sweet_dedup):.2f}")
for label, pct in [("Degen (10%)", 0.10), ("Full Degen (20%)", 0.20), ("YOLO (33%)", 0.33)]:
    balance = 1000.0
    for t in sweet_dedup:
        bet = balance * pct
        if bet < 5: break
        if t['direction'] == 'BUY':
            ret = (t['exit_price'] - t['entry_price']) / t['entry_price'] if t['entry_price'] > 0 else 0
        else:
            ret = (t['entry_price'] - t['exit_price']) / (1 - t['entry_price']) if t['entry_price'] < 1 else 0
        ret = max(ret, -1.0)
        balance += bet * ret
    print(f"  {label:25s}: ${balance:.2f} ({balance-1000:+.2f})")
