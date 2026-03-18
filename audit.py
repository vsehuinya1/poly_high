import csv, os, math
from collections import defaultdict, Counter
from datetime import datetime

FILES = [
    'sports_data/paper_trades_20260222.csv',
    'sports_data/paper_trades_20260225.csv',
    'sports_data/paper_trades_20260226.csv',
    'sports_data/paper_trades_20260227.csv',
]

positions = defaultdict(lambda: {'entry': None, 'exit': None})
all_entries = []
all_exits = []

for fp in FILES:
    if not os.path.exists(fp):
        continue
    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['position_id']
            if row['event'] == 'ENTRY':
                positions[pid]['entry'] = row
                all_entries.append(row)
            elif row['event'] == 'EXIT':
                positions[pid]['exit'] = row
                all_exits.append(row)

trades = []
for pid, data in positions.items():
    if data['entry'] and data['exit']:
        e = data['entry']
        x = data['exit']
        pnl = float(x['pnl']) if x['pnl'] else 0.0
        entry_price = float(e['entry_price']) if e['entry_price'] else 0.0
        exit_price = float(x['exit_price']) if x['exit_price'] else 0.0
        edge = float(e['entry_edge']) if e['entry_edge'] else 0.0
        gs = e['game_state']

        nba_teams = ['Lakers','Spurs','Heat','Raptors','Bulls','Trail Blazers','Kings',
                     'Mavericks','Hornets','Pacers','Wizards','Hawks','Celtics','Nets',
                     'Rockets','Magic','76ers','Timberwolves','Clippers','Grizzlies',
                     'Warriors','Bucks','Cavaliers','Pistons','Thunder','Jazz','Pelicans',
                     'Nuggets','Suns','Knicks']
        sport = 'nba' if any(t in gs for t in nba_teams) else 'football'

        elapsed = 0
        if "'" in gs:
            try:
                mins_str = gs.split("(")[-1].replace("')", "").replace("'", "")
                elapsed = float(mins_str)
            except:
                pass

        trades.append({
            'pid': pid, 'sport': sport, 'pnl': pnl,
            'entry_price': entry_price, 'exit_price': exit_price,
            'edge': edge, 'direction': e['direction'],
            'exit_reason': x['exit_reason'], 'game_state': gs,
            'elapsed': elapsed, 'size': float(e['size_usd']),
            'timestamp': float(e['timestamp']),
        })

open_positions = sum(1 for d in positions.values() if d['entry'] and not d['exit'])

print("=" * 70)
print("  FULL SYSTEM AUDIT")
print("=" * 70)
print()
print(f"Total completed trades: {len(trades)}")
print(f"Open positions (no exit): {open_positions}")
print(f"Total entries: {len(all_entries)}")
print(f"Total exits: {len(all_exits)}")

if not trades:
    print("NO COMPLETED TRADES")
    exit()

wins = [t for t in trades if t['pnl'] > 0]
losses = [t for t in trades if t['pnl'] < 0]
flat = [t for t in trades if t['pnl'] == 0]
total_pnl = sum(t['pnl'] for t in trades)
avg_pnl = total_pnl / len(trades)
avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
win_rate = len(wins) / len(trades) * 100

print(f"\nWin rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L / {len(flat)}F)")
print(f"Total PnL: ${total_pnl:.2f}")
print(f"Avg PnL/trade: ${avg_pnl:.2f}")
print(f"Avg win: ${avg_win:.2f}")
print(f"Avg loss: ${avg_loss:.2f}")
print(f"Avg edge at entry: {sum(t['edge'] for t in trades)/len(trades):.4f}")
print(f"Avg entry price: {sum(t['entry_price'] for t in trades)/len(trades):.4f}")
print(f"Avg size: ${sum(t['size'] for t in trades)/len(trades):.2f}")

print("\n--- PnL by Sport ---")
for sport in ['nba', 'football']:
    st = [t for t in trades if t['sport'] == sport]
    if st:
        sp = sum(t['pnl'] for t in st)
        sw = sum(1 for t in st if t['pnl'] > 0)
        print(f"  {sport.upper()}: {len(st)} trades, PnL=${sp:.2f}, WR={sw/len(st)*100:.1f}%, Avg=${sp/len(st):.2f}")

print("\n--- PnL by Entry Price Bucket ---")
buckets = [(0.0,0.15),(0.15,0.30),(0.30,0.45),(0.45,0.60),(0.60,0.70),(0.70,1.0)]
for lo, hi in buckets:
    bt = [t for t in trades if lo <= t['entry_price'] < hi]
    if bt:
        bp = sum(t['pnl'] for t in bt)
        bw = sum(1 for t in bt if t['pnl'] > 0)
        print(f"  [{lo:.2f}-{hi:.2f}): {len(bt)} trades, PnL=${bp:.2f}, WR={bw/len(bt)*100:.1f}%, Avg=${bp/len(bt):.2f}")

print("\n--- PnL by Edge Bucket ---")
edge_buckets = [(0.05,0.10),(0.10,0.15),(0.15,0.25),(0.25,0.50),(0.50,1.0)]
for lo, hi in edge_buckets:
    bt = [t for t in trades if lo <= abs(t['edge']) < hi]
    if bt:
        bp = sum(t['pnl'] for t in bt)
        bw = sum(1 for t in bt if t['pnl'] > 0)
        print(f"  [{lo:.2f}-{hi:.2f}): {len(bt)} trades, PnL=${bp:.2f}, WR={bw/len(bt)*100:.1f}%, Avg=${bp/len(bt):.2f}")

print("\n--- Exit Reason Distribution ---")
reason_counts = Counter()
reason_pnl = defaultdict(float)
for t in trades:
    reason_counts[t['exit_reason']] += 1
    reason_pnl[t['exit_reason']] += t['pnl']
for reason, count in reason_counts.most_common():
    pct = count / len(trades) * 100
    print(f"  {reason}: {count} ({pct:.1f}%), PnL=${reason_pnl[reason]:.2f}")

print("\n--- Direction Distribution ---")
for direction in ['BUY', 'SELL']:
    dt = [t for t in trades if t['direction'] == direction]
    if dt:
        dp = sum(t['pnl'] for t in dt)
        dw = sum(1 for t in dt if t['pnl'] > 0)
        print(f"  {direction}: {len(dt)} trades, PnL=${dp:.2f}, WR={dw/len(dt)*100:.1f}%")

print("\n--- Elapsed Minutes at Entry ---")
min_buckets = [(0,10),(10,20),(20,30),(30,40),(40,48)]
for lo, hi in min_buckets:
    bt = [t for t in trades if lo <= t['elapsed'] < hi]
    if bt:
        bp = sum(t['pnl'] for t in bt)
        bw = sum(1 for t in bt if t['pnl'] > 0)
        print(f"  [{lo}-{hi}min): {len(bt)} trades, PnL=${bp:.2f}, WR={bw/len(bt)*100:.1f}%")

print("\n--- Top 10 Trades ---")
sorted_trades = sorted(trades, key=lambda t: t['pnl'], reverse=True)
for t in sorted_trades[:10]:
    print(f"  ${t['pnl']:+.2f} | {t['direction']} @ {t['entry_price']:.3f} > {t['exit_price']:.3f} | edge={t['edge']:.3f} | {t['exit_reason']} | {t['game_state']}")

print("\n--- Bottom 10 Trades ---")
for t in sorted_trades[-10:]:
    print(f"  ${t['pnl']:+.2f} | {t['direction']} @ {t['entry_price']:.3f} > {t['exit_price']:.3f} | edge={t['edge']:.3f} | {t['exit_reason']} | {t['game_state']}")

print("\n--- Drawdown ---")
cum = 0; peak = 0; max_dd = 0
for t in sorted(trades, key=lambda x: x['timestamp']):
    cum += t['pnl']
    if cum > peak: peak = cum
    dd = peak - cum
    if dd > max_dd: max_dd = dd
print(f"  Peak PnL: ${peak:.2f}")
print(f"  Max Drawdown: ${max_dd:.2f}")
print(f"  Final PnL: ${cum:.2f}")
if len(trades) > 1:
    pnls = [t['pnl'] for t in trades]
    mu = sum(pnls)/len(pnls)
    var = sum((p-mu)**2 for p in pnls)/(len(pnls)-1)
    std = var**0.5
    sharpe = mu/std if std > 0 else 0
    print(f"  Per-trade Sharpe: {sharpe:.3f}")

print("\n--- Trades by Date ---")
date_counts = Counter()
for t in trades:
    dt = datetime.utcfromtimestamp(t['timestamp'])
    date_counts[dt.strftime('%Y-%m-%d')] += 1
for d, c in sorted(date_counts.items()):
    dt_trades = [t for t in trades if datetime.utcfromtimestamp(t['timestamp']).strftime('%Y-%m-%d') == d]
    dp = sum(t['pnl'] for t in dt_trades)
    print(f"  {d}: {c} trades, PnL=${dp:.2f}")

print("\n--- Signal Volume ---")
signal_files = [
    'sports_data/signals_20260222.csv',
    'sports_data/signals_20260224.csv',
    'sports_data/signals_20260225.csv',
    'sports_data/signals_20260226.csv',
    'sports_data/signals_20260227.csv',
]
total_signals = 0
for sf in signal_files:
    if os.path.exists(sf):
        with open(sf) as f:
            cnt = sum(1 for _ in f) - 1
            total_signals += cnt
            print(f"  {sf}: {cnt:,} signals")
print(f"  TOTAL: {total_signals:,} signals")
print(f"  Conversion rate: {len(trades)}/{total_signals} = {len(trades)/max(1,total_signals)*100:.6f}%")

# === Spread and book_age from snapshot data ===
print("\n--- Snapshot-level Microstructure Sample ---")
snap_file = 'sports_data/snapshots_20260227.csv'
if os.path.exists(snap_file):
    with open(snap_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  Total snapshots in {snap_file}: {len(rows)}")
    if rows:
        home_mkts = [float(r['home_p_mkt']) for r in rows if r.get('home_p_mkt')]
        edges = [abs(float(r['edge'])) for r in rows if r.get('edge')]
        print(f"  Avg home_p_mkt: {sum(home_mkts)/len(home_mkts):.4f}")
        print(f"  Avg abs(edge): {sum(edges)/len(edges):.4f}")
        big_edges = [e for e in edges if e > 0.07]
        print(f"  Signals with edge > 0.07: {len(big_edges)} ({len(big_edges)/len(edges)*100:.1f}%)")
