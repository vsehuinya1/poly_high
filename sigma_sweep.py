"""
NBA Model Recalibration Study — Parts 1-4
Replays Mar 3-4 snapshots with different sigma values, collapse filters,
regime splits, and entry band tightening.
"""
import csv, os, math
from collections import defaultdict

# ── Model replica (inline for sweep) ─────────────────────────────────
def _phi(x):
    """Standard normal CDF."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-x*x/2.0)
    return 0.5 * (1.0 + sign * y)

def _inv_phi(p):
    """Inverse normal CDF approximation."""
    if p <= 0.0: return -8.0
    if p >= 1.0: return 8.0
    if p == 0.5: return 0.0
    if p < 0.5:
        return -_inv_phi(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t)

def nba_win_prob_custom(home_score, away_score, adj_seconds, pregame_prob, sigma_base):
    """Recompute NBA P(home_win) with custom sigma_base."""
    score_diff = home_score - away_score
    if adj_seconds <= 0:
        return 0.999 if score_diff > 0 else (0.001 if score_diff < 0 else 0.5)
    t_min = adj_seconds / 60.0
    # Dynamic sigma: scale the endgame regimes proportionally
    ratio = sigma_base / 1.70
    if adj_seconds <= 60:
        sigma = 2.10 * ratio
    elif adj_seconds <= 180:
        sigma = 1.95 * ratio
    else:
        sigma = sigma_base
    # Anchor uses FIXED 1.70 for S0 (pre-game calibration doesn't change)
    s0 = 1.70 * math.sqrt(48.0) * _inv_phi(pregame_prob)
    strength_adj = s0 * (t_min / 48.0)
    s_eff = score_diff + strength_adj
    z = s_eff / (sigma * math.sqrt(t_min))
    return max(0.001, min(0.999, _phi(z)))

# ── Load snapshot data ───────────────────────────────────────────────
SNAP_FILES = [
    'sports_data/snapshots_20260303.csv',
    'sports_data/snapshots_20260304.csv',
]

NBA_TEAMS_IN_ID = set()  # We'll detect NBA by game_id format (starts with 002)

all_snaps = []
for fp in SNAP_FILES:
    if not os.path.exists(fp):
        continue
    with open(fp) as f:
        for row in csv.DictReader(f):
            try:
                gid = row.get('game_id', '')
                if not gid.startswith('002'):  # NBA game IDs start with 002
                    continue
                all_snaps.append({
                    'game_id': gid,
                    'h_score': int(row.get('home_score', 0)),
                    'a_score': int(row.get('away_score', 0)),
                    'elapsed': float(row.get('elapsed', 0)),
                    'adj_sec': float(row.get('adjusted_seconds', 9999)),
                    'home_mkt': float(row.get('home_p_mkt', 0)),
                    'away_mkt': float(row.get('away_p_mkt', 0)),
                    'edge': float(row.get('edge', 0)),
                    'sigma': float(row.get('sigma', 1.70)),
                    'pregame': float(row.get('pregame_probability', 0.5)),
                    'ts': float(row.get('timestamp', 0)),
                    'period': row.get('period', ''),
                })
            except:
                continue

# Sort by time
all_snaps.sort(key=lambda s: s['ts'])

# Group by game, get final resolution price
games = defaultdict(list)
for s in all_snaps:
    games[s['game_id']].append(s)

# Determine final outcome for each game
game_final = {}
for gid, snaps in games.items():
    last = snaps[-1]
    # Final price: use final market price as proxy for resolution
    # If h_score > a_score at end, home wins -> final = 0.999
    if last['adj_sec'] <= 0 or last['elapsed'] >= 47:
        if last['h_score'] > last['a_score']:
            game_final[gid] = 0.999
        elif last['h_score'] < last['a_score']:
            game_final[gid] = 0.001
        else:
            game_final[gid] = last['home_mkt']  # OT - use market
    else:
        game_final[gid] = last['home_mkt']  # use last observed market price

print(f"NBA snapshots loaded: {len(all_snaps)}")
print(f"NBA games: {len(games)}")
for gid in sorted(games.keys()):
    snaps = games[gid]
    last = snaps[-1]
    print(f"  {gid}: {last['h_score']}-{last['a_score']} (period={last['period']}, "
          f"elapsed={last['elapsed']:.0f}min, final_p={game_final.get(gid, '?'):.3f})")

# ═════════════════════════════════════════════════════════════════════
#  Helper: run a replay with given parameters
# ═════════════════════════════════════════════════════════════════════
def run_replay(sigma_base, price_lo=0.45, price_hi=0.65, edge_min=0.10,
               max_minute=48, collapse_filter=False, one_per_game=True):
    """Replay signals through filters, return list of trade dicts."""
    trades = []
    game_traded = {}  # game_id -> set of directions traded

    for gid, snaps in games.items():
        if gid not in game_final:
            continue
        traded_dirs = set()

        for s in snaps:
            # Recompute model prob with custom sigma
            model_p = nba_win_prob_custom(
                s['h_score'], s['a_score'], s['adj_sec'],
                s['pregame'], sigma_base
            )
            mkt_p = s['home_mkt']
            edge = model_p - mkt_p

            # Time gate
            if s['elapsed'] > max_minute:
                continue

            # Edge threshold
            if abs(edge) < edge_min:
                continue

            # Price band on market mid
            if not (price_lo <= mkt_p <= price_hi):
                continue

            # Direction
            if edge > 0:
                direction = "BUY"
                entry_price = mkt_p  # simplified: use mid
            else:
                direction = "SELL"
                entry_price = mkt_p

            # One per game + direction
            if one_per_game:
                if direction in traded_dirs:
                    continue

            # Collapse filter
            if collapse_filter:
                time_remaining_min = 48 - s['elapsed']
                lead = abs(s['h_score'] - s['a_score'])
                if time_remaining_min < 6 and lead <= 5:
                    # Check if we're backing the trailing team
                    if direction == "BUY" and s['h_score'] < s['a_score']:
                        continue
                    if direction == "SELL" and s['h_score'] > s['a_score']:
                        continue

            traded_dirs.add(direction)
            final_p = game_final[gid]

            if direction == "BUY":
                pnl = (final_p - entry_price) * 200
            else:
                pnl = (entry_price - final_p) * 200

            trades.append({
                'game_id': gid, 'direction': direction,
                'entry_price': entry_price, 'final_price': final_p,
                'edge': edge, 'model_p': model_p, 'mkt_p': mkt_p,
                'pnl': pnl, 'elapsed': s['elapsed'],
                'h_score': s['h_score'], 'a_score': s['a_score'],
                'sigma_used': sigma_base,
            })

    return trades

def summarize(trades, label=""):
    """Print summary stats."""
    if not trades:
        print(f"  {label}: 0 trades")
        return {'count': 0, 'wr': 0, 'pnl': 0, 'avg_pnl': 0, 'sharpe': 0, 'avg_edge': 0, 'dd': 0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    wr = wins / len(trades) * 100
    total_pnl = sum(t['pnl'] for t in trades)
    avg_pnl = total_pnl / len(trades)
    avg_edge = sum(abs(t['edge']) for t in trades) / len(trades)
    avg_misprice = sum(abs(t['model_p'] - t['mkt_p']) for t in trades) / len(trades)
    # Sharpe approx
    pnls = [t['pnl'] for t in trades]
    mu = sum(pnls) / len(pnls)
    if len(pnls) > 1:
        var = sum((p-mu)**2 for p in pnls) / (len(pnls)-1)
        std = var**0.5
        sharpe = mu / std if std > 0 else 0
    else:
        sharpe = 0
    # Max drawdown
    cum = 0; peak = 0; dd = 0
    for t in trades:
        cum += t['pnl']
        if cum > peak: peak = cum
        d = peak - cum
        if d > dd: dd = d
    return {'count': len(trades), 'wr': wr, 'pnl': total_pnl, 'avg_pnl': avg_pnl,
            'sharpe': sharpe, 'avg_edge': avg_edge, 'dd': dd, 'avg_misprice': avg_misprice}

# ═════════════════════════════════════════════════════════════════════
#  PART 1 — Sigma Sweep
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("  PART 1 — SIGMA SWEEP")
print("=" * 75)
print(f"\n{'σ':>5} | {'Trades':>6} | {'WR%':>5} | {'Avg Edge':>8} | {'Avg PnL':>8} | {'Total PnL':>9} | {'MaxDD':>7} | {'Sharpe':>6} | {'Avg |Δ|':>7}")
print("-" * 75)

sigma_results = {}
for sig in [1.70, 2.00, 2.20, 2.40, 2.60]:
    trades = run_replay(sig)
    r = summarize(trades)
    sigma_results[sig] = (trades, r)
    print(f"{sig:5.2f} | {r['count']:6d} | {r['wr']:5.1f} | {r['avg_edge']:8.3f} | "
          f"${r['avg_pnl']:7.2f} | ${r['pnl']:8.2f} | ${r['dd']:6.2f} | {r['sharpe']:6.3f} | {r.get('avg_misprice',0):7.3f}")

# Find best sigma
best_sigma = max(sigma_results.keys(), key=lambda s: sigma_results[s][1]['pnl'])
print(f"\nBest σ: {best_sigma:.2f} (PnL=${sigma_results[best_sigma][1]['pnl']:.2f})")

# Show per-trade detail for best sigma
print(f"\n--- Per-trade detail (σ={best_sigma:.2f}) ---")
best_trades, _ = sigma_results[best_sigma]
for t in best_trades:
    marker = "WIN" if t['pnl'] > 0 else "LOSS"
    print(f"  [{marker:4s}] {t['direction']} @ {t['entry_price']:.3f} -> {t['final_price']:.3f} "
          f"| edge={t['edge']:+.3f} model={t['model_p']:.3f} mkt={t['mkt_p']:.3f} "
          f"| min {t['elapsed']:.0f} | {t['h_score']}-{t['a_score']} | ${t['pnl']:+.2f}")

# ═════════════════════════════════════════════════════════════════════
#  PART 2 — Collapse Filter
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print(f"  PART 2 — COLLAPSE FILTER (σ={best_sigma:.2f})")
print("=" * 75)

trades_no_collapse = run_replay(best_sigma, collapse_filter=False)
trades_with_collapse = run_replay(best_sigma, collapse_filter=True)

r_nc = summarize(trades_no_collapse)
r_wc = summarize(trades_with_collapse)

removed = r_nc['count'] - r_wc['count']
delta_pnl = r_wc['pnl'] - r_nc['pnl']

print(f"\n{'':15s} | {'Trades':>6} | {'WR%':>5} | {'Total PnL':>9} | {'Avg PnL':>8}")
print("-" * 55)
print(f"{'No filter':15s} | {r_nc['count']:6d} | {r_nc['wr']:5.1f} | ${r_nc['pnl']:8.2f} | ${r_nc['avg_pnl']:7.2f}")
print(f"{'With collapse':15s} | {r_wc['count']:6d} | {r_wc['wr']:5.1f} | ${r_wc['pnl']:8.2f} | ${r_wc['avg_pnl']:7.2f}")
print(f"\nTrades removed: {removed}")
print(f"PnL delta: ${delta_pnl:+.2f}")

# ═════════════════════════════════════════════════════════════════════
#  PART 3 — Early vs Late Regime
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print(f"  PART 3 — EARLY vs LATE REGIME (σ={best_sigma:.2f})")
print("=" * 75)

# Early: min 0-36
trades_early = run_replay(best_sigma, max_minute=36)
r_early = summarize(trades_early)

# Late: min 36-48 (take all, then filter)
trades_all = run_replay(best_sigma, max_minute=48)
trades_late = [t for t in trades_all if t['elapsed'] > 36]
r_late_count = len(trades_late)
r_late_wins = sum(1 for t in trades_late if t['pnl'] > 0)
r_late_pnl = sum(t['pnl'] for t in trades_late)
r_late_wr = r_late_wins / r_late_count * 100 if r_late_count > 0 else 0
r_late_avg = r_late_pnl / r_late_count if r_late_count > 0 else 0

# Actually we need to handle this differently - run with no game dedup, just time windows
# Let me use a different approach: run full, split by elapsed at entry time
trades_full = run_replay(best_sigma, max_minute=48, one_per_game=False)
early_t = [t for t in trades_full if t['elapsed'] <= 36]
late_t = [t for t in trades_full if t['elapsed'] > 36]

def quick_stats(tlist, label):
    if not tlist:
        return {'count': 0, 'wr': 0, 'pnl': 0, 'avg_pnl': 0}
    wins = sum(1 for t in tlist if t['pnl'] > 0)
    wr = wins/len(tlist)*100
    pnl = sum(t['pnl'] for t in tlist)
    avg = pnl/len(tlist)
    return {'count': len(tlist), 'wr': wr, 'pnl': pnl, 'avg_pnl': avg}

re = quick_stats(early_t, "Early")
rl = quick_stats(late_t, "Late")

print(f"\n{'Regime':12s} | {'Trades':>6} | {'WR%':>5} | {'Total PnL':>9} | {'Avg PnL':>8} | {'Expectancy':>10}")
print("-" * 65)
print(f"{'Early(0-36)':12s} | {re['count']:6d} | {re['wr']:5.1f} | ${re['pnl']:8.2f} | ${re['avg_pnl']:7.2f} | ${re['avg_pnl']:9.2f}")
print(f"{'Late(36-48)':12s} | {rl['count']:6d} | {rl['wr']:5.1f} | ${rl['pnl']:8.2f} | ${rl['avg_pnl']:7.2f} | ${rl['avg_pnl']:9.2f}")

# Also with dedup
trades_early_dedup = run_replay(best_sigma, max_minute=36, one_per_game=True)
red = summarize(trades_early_dedup)
print(f"\n{'Early(dedup)':12s} | {red['count']:6d} | {red['wr']:5.1f} | ${red['pnl']:8.2f} | ${red['avg_pnl']:7.2f} | ${red['avg_pnl']:9.2f}")

# ═════════════════════════════════════════════════════════════════════
#  PART 4 — Entry Band Tightening
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print(f"  PART 4 — ENTRY BAND SENSITIVITY (σ={best_sigma:.2f}, collapse={r_wc['count'] != r_nc['count']})")
print("=" * 75)

use_collapse = r_wc['pnl'] > r_nc['pnl']  # use collapse filter if it helps

print(f"\n{'Band':14s} | {'Trades':>6} | {'WR%':>5} | {'Total PnL':>9} | {'Avg PnL':>8} | {'Expectancy':>10}")
print("-" * 65)

for lo, hi in [(0.45, 0.65), (0.48, 0.62), (0.50, 0.60), (0.40, 0.70), (0.35, 0.70)]:
    t = run_replay(best_sigma, price_lo=lo, price_hi=hi, collapse_filter=use_collapse)
    r = summarize(t)
    print(f"[{lo:.2f}-{hi:.2f}]   | {r['count']:6d} | {r['wr']:5.1f} | ${r['pnl']:8.2f} | ${r['avg_pnl']:7.2f} | ${r['avg_pnl']:9.2f}")

# ═════════════════════════════════════════════════════════════════════
#  FINAL RECOMMENDATION
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("  FINAL RECOMMENDATION")
print("=" * 75)
print(f"\n  Optimal σ: {best_sigma:.2f}")
print(f"  Collapse filter: {'YES' if use_collapse else 'NO'}")

# Find best band
best_band_pnl = -9999
best_band = (0.45, 0.65)
for lo, hi in [(0.45, 0.65), (0.48, 0.62), (0.50, 0.60), (0.40, 0.70), (0.35, 0.70)]:
    t = run_replay(best_sigma, price_lo=lo, price_hi=hi, collapse_filter=use_collapse)
    r = summarize(t)
    if r['pnl'] > best_band_pnl:
        best_band_pnl = r['pnl']
        best_band = (lo, hi)
print(f"  Optimal entry band: [{best_band[0]:.2f}, {best_band[1]:.2f}]")
print(f"  Late-game trading: {'KEEP' if rl['avg_pnl'] > 0 else 'DISABLE (negative expectancy)'}")
