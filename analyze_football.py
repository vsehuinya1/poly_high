import subprocess
import csv
from collections import defaultdict
import sys

# Fetch the CSV from the VPS
cmd = "sshpass -p '12345vse' ssh -o StrictHostKeyChecking=no root@161.97.185.65 'cat /root/poly_high_sports/sports_data/paper_trades_20260226.csv'"
try:
    output = subprocess.check_output(cmd, shell=True, text=True)
except subprocess.CalledProcessError as e:
    print(f"Failed to fetch CSV: {e}")
    sys.exit(1)

positions = defaultdict(lambda: {'entries': [], 'exits': []})
reader = csv.DictReader(output.strip().split('\n'))
for row in reader:
    pos_id = row.get('position_id')
    if not pos_id:
        continue
    if row['event'] == 'ENTRY':
        positions[pos_id]['entries'].append(row)
    elif row['event'] == 'EXIT':
        positions[pos_id]['exits'].append(row)

football_games = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'winners': 0, 'losers': 0, 'details': []})
total_pnl = 0.0

nba_teams = ['Lakers', 'Spurs', 'Heat', 'Raptors', 'Bulls', 'Trail Blazers', 'Kings', 'Mavericks', 'Hornets', 'Pacers', 'Wizards', 'Hawks', 'Celtics', 'Nets', 'Rockets', 'Magic', '76ers', 'Timberwolves', 'Clippers', 'Grizzlies', 'Warriors', 'Bucks', 'Cavaliers', 'Pistons', 'Thunder', 'Jazz', 'Pelicans']

for pos_id, data in positions.items():
    if not data['entries'] or not data['exits']:
        continue
    
    entry = data['entries'][0]
    exit_row = data['exits'][0]
    game_state = entry['game_state']
    
    # Filter out NBA games
    if any(nba in game_state for nba in nba_teams) or 'Quarter' in game_state or game_state.split(' ')[-1].startswith('(Q'):
        continue
        
    game = game_state.split('(')[0].strip()
    pnl = float(exit_row['pnl']) if exit_row['pnl'] else 0.0
    
    football_games[game]['trades'] += 1
    football_games[game]['pnl'] += pnl
    if pnl > 0:
        football_games[game]['winners'] += 1
    elif pnl < 0:
        football_games[game]['losers'] += 1
        
    football_games[game]['details'].append({
        'dir': entry['direction'],
        'out': entry['outcome'],
        'entry_p': float(entry['entry_price']),
        'exit_p': float(exit_row['exit_price']),
        'edge': float(entry['entry_edge']),
        'pnl': pnl,
        'reason': exit_row['exit_reason'],
        'state': game_state
    })
    total_pnl += pnl

print(f"--- FOOTBALL PERFORMANCE BREAKDOWN ---")
print(f"Total Football PnL: ${total_pnl:.2f}")
print(f"Total Trades: {sum(g['trades'] for g in football_games.values())}\n")

if not football_games:
    print("No football trades found for 2026-02-26.")
else:
    for game, stats in football_games.items():
        print(f"Game: {game}")
        print(f"  Trades: {stats['trades']} (W:{stats['winners']} L:{stats['losers']})")
        print(f"  PnL: ${stats['pnl']:.2f}")
        for d in stats['details']:
            print(f"    - {d['dir']} {d['out']} @ {d['entry_p']:.3f} -> {d['exit_p']:.3f} | Edge: {d['edge']:+.3f} | PnL: ${d['pnl']:+.2f} ({d['reason']}) [{d['state']}]")
        print("")
