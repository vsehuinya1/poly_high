"""
What-if analysis: show missed profitable trades from Mar 3-4 sessions.
Replays snapshot data through different filter levels.
"""
import csv, os
from collections import defaultdict, Counter
from datetime import datetime

SNAP_FILES = [
    'sports_data/snapshots_20260303.csv',
    'sports_data/snapshots_20260304.csv',
]

NBA_TEAMS = ['Lakers','Spurs','Heat','Raptors','Bulls','Trail Blazers','Kings',
             'Mavericks','Hornets','Pacers','Wizards','Hawks','Celtics','Nets',
             'Rockets','Magic','76ers','Timberwolves','Clippers','Grizzlies',
             'Warriors','Bucks','Cavaliers','Pistons','Thunder','Jazz','Pelicans',
             'Nuggets','Suns','Knicks']

all_snaps = []
for fp in SNAP_FILES:
    if not os.path.exists(fp):
        print(f"  {fp} not found, skipping")
        continue
    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                elapsed = float(row.get('elapsed', 0))
                edge = float(row.get('edge', 0))
                home_mkt = float(row.get('home_p_mkt', 0))
                adj_sec = float(row.get('adjusted_seconds', 9999))
                ts = float(row.get('timestamp', 0))
                gid = row.get('game_id', '')
                h_score = int(row.get('home_score', 0))
                a_score = int(row.get('away_score', 0))
                period = row.get('period', '')
            except:
                continue
            all_snaps.append({
                'elapsed': elapsed, 'edge': edge, 'home_mkt': home_mkt,
                'adj_sec': adj_sec, 'ts': ts, 'game_id': gid,
                'h_score': h_score, 'a_score': a_score, 'period': period,
                'file': fp,
            })

print(f"Total snapshots loaded: {len(all_snaps)}")

# Group by game
games = defaultdict(list)
for s in all_snaps:
    games[s['game_id']].append(s)

print(f"Unique games: {len(games)}")

# For each game, find the edges in sweet spot zone at different time windows
print("\n" + "=" * 70)
print("  WHAT-IF ANALYSIS: Missed opportunities at different time gates")
print("=" * 70)

for label, max_min in [("min 0-36 (current)", 36), ("min 0-40", 40), ("min 0-44", 44), ("min 0-48 (no limit)", 48)]:
    sweet = [s for s in all_snaps
             if 0.45 <= s['home_mkt'] <= 0.65
             and abs(s['edge']) >= 0.10
             and s['elapsed'] <= max_min]
    print(f"\n--- {label} ---")
    print(f"  Qualifying ticks: {len(sweet)}")
    if sweet:
        games_with = set(s['game_id'] for s in sweet)
        print(f"  Games with signals: {len(games_with)}")
        avg_edge = sum(abs(s['edge']) for s in sweet) / len(sweet)
        avg_price = sum(s['home_mkt'] for s in sweet) / len(sweet)
        print(f"  Avg edge: {avg_edge:.3f}")
        print(f"  Avg mkt price: {avg_price:.3f}")

# Show per-game detail: best edge opportunity + what happened later
print("\n" + "=" * 70)
print("  PER-GAME OPPORTUNITY ANALYSIS")
print("=" * 70)

for gid, snaps in sorted(games.items(), key=lambda x: x[1][0]['ts']):
    snaps.sort(key=lambda s: s['ts'])

    # Find sweet-spot edges
    sweet = [s for s in snaps if 0.45 <= s['home_mkt'] <= 0.65 and abs(s['edge']) >= 0.10]
    if not sweet:
        continue

    # First and last snapshot for game context
    first = snaps[0]
    last = snaps[-1]

    # Best edge in sweet spot
    best = max(sweet, key=lambda s: abs(s['edge']))

    # What happened: early price vs final price
    early_prices = [s['home_mkt'] for s in snaps if s['elapsed'] <= 20]
    late_prices = [s['home_mkt'] for s in snaps if s['elapsed'] >= 40]

    early_avg = sum(early_prices)/len(early_prices) if early_prices else 0
    late_avg = sum(late_prices)/len(late_prices) if late_prices else 0
    final_price = last['home_mkt']

    # Was the edge right? If edge was negative (SELL), did price go down?
    if best['edge'] < 0:
        direction = "SELL"
        would_profit = final_price < best['home_mkt']
    else:
        direction = "BUY"
        would_profit = final_price > best['home_mkt']

    result = "WIN" if would_profit else "LOSS"
    price_move = final_price - best['home_mkt']

    print(f"\n  Game {gid} | Score: {last['h_score']}-{last['a_score']} | Period: {last['period']}")
    print(f"    Sweet-spot ticks: {len(sweet)} | Best: min {best['elapsed']:.0f} edge={best['edge']:.3f}")
    print(f"    Best signal: {direction} @ {best['home_mkt']:.3f} (min {best['elapsed']:.0f})")
    print(f"    Final price: {final_price:.3f} | Move: {price_move:+.3f}")
    print(f"    Result: {result}")

    # Show what PnL would have been with $200 bet
    if direction == "SELL":
        pnl = (best['home_mkt'] - final_price) * 200
    else:
        pnl = (final_price - best['home_mkt']) * 200
    print(f"    Hypothetical PnL ($200): ${pnl:+.2f}")

# Show what the different time-gate versions would have earned
print("\n" + "=" * 70)
print("  HYPOTHETICAL PnL AT DIFFERENT TIME GATES ($200/trade)")
print("=" * 70)

for label, max_min in [("min 0-36", 36), ("min 0-40", 40), ("min 0-44", 44), ("min 0-48", 48)]:
    total_pnl = 0
    trade_count = 0
    wins = 0
    game_traded = set()

    for gid, snaps in games.items():
        snaps.sort(key=lambda s: s['ts'])
        sweet = [s for s in snaps if 0.45 <= s['home_mkt'] <= 0.65
                 and abs(s['edge']) >= 0.10 and s['elapsed'] <= max_min]
        if not sweet or gid in game_traded:
            continue

        # Take first qualifying signal
        sig = sweet[0]
        game_traded.add(gid)
        last = snaps[-1]

        if sig['edge'] < 0:
            pnl = (sig['home_mkt'] - last['home_mkt']) * 200
        else:
            pnl = (last['home_mkt'] - sig['home_mkt']) * 200

        total_pnl += pnl
        trade_count += 1
        if pnl > 0: wins += 1

    wr = wins/trade_count*100 if trade_count > 0 else 0
    print(f"  {label:12s}: {trade_count} trades, WR={wr:.0f}%, PnL=${total_pnl:+.2f}")
