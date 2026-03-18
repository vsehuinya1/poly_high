"""
Time-Weighted Lead Discount Sweep (NBA Model v3.5 Evaluation)
─────────────────────────────────────────────────────────────
EffectiveLead = RawLead × (1 − exp(−k × elapsed/48))
k ∈ {1.5, 2.0, 2.5, 3.0}  +  baseline (k=∞ i.e. no discount)
SELL-only, entry band [0.45-0.65], edge ≥ 0.10, first per game, max min 36
"""
import csv, os, math
from collections import defaultdict

# ── Model functions ──────────────────────────────────────────────────
def _phi(x):
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-x*x/2.0)
    return 0.5 * (1.0 + sign * y)

def _inv_phi(p):
    if p <= 0.0: return -8.0
    if p >= 1.0: return 8.0
    if p == 0.5: return 0.0
    if p < 0.5: return -_inv_phi(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t)

def nba_prob_discount(h_score, a_score, adj_sec, pregame_prob, k_discount=None):
    """
    NBA P(home_win) with optional time-weighted lead discount.
    k_discount=None → baseline (no discount, original model)
    k_discount=float → EffectiveLead = RawLead × (1 − exp(−k × elapsed_frac))
    """
    raw_diff = h_score - a_score

    if adj_sec <= 0:
        return 0.999 if raw_diff > 0 else (0.001 if raw_diff < 0 else 0.5)

    t_min = adj_sec / 60.0
    elapsed_min = 48.0 - t_min
    elapsed_frac = max(0, elapsed_min) / 48.0

    # Apply lead discount
    if k_discount is not None:
        discount = 1.0 - math.exp(-k_discount * elapsed_frac)
        score_diff = raw_diff * discount
    else:
        score_diff = raw_diff

    # Dynamic sigma (same as production)
    if adj_sec <= 60:
        sigma = 2.10
    elif adj_sec <= 180:
        sigma = 1.95
    else:
        sigma = 1.70

    # Strength anchor (unchanged)
    s0 = 1.70 * math.sqrt(48.0) * _inv_phi(pregame_prob)
    strength_adj = s0 * (t_min / 48.0)

    s_eff = score_diff + strength_adj
    z = s_eff / (sigma * math.sqrt(t_min))
    return max(0.001, min(0.999, _phi(z)))

# ── Load data ────────────────────────────────────────────────────────
SNAP_FILES = [
    'sports_data/snapshots_20260303.csv',
    'sports_data/snapshots_20260304.csv',
]

all_snaps = []
for fp in SNAP_FILES:
    if not os.path.exists(fp):
        continue
    with open(fp) as f:
        for row in csv.DictReader(f):
            try:
                gid = row.get('game_id', '')
                if not gid.startswith('002'):
                    continue
                all_snaps.append({
                    'game_id': gid,
                    'h_score': int(row.get('home_score', 0)),
                    'a_score': int(row.get('away_score', 0)),
                    'elapsed': float(row.get('elapsed', 0)),
                    'adj_sec': float(row.get('adjusted_seconds', 9999)),
                    'home_mkt': float(row.get('home_p_mkt', 0)),
                    'pregame': float(row.get('pregame_probability', 0.5)),
                    'ts': float(row.get('timestamp', 0)),
                })
            except:
                continue

all_snaps.sort(key=lambda s: s['ts'])
games = defaultdict(list)
for s in all_snaps:
    games[s['game_id']].append(s)

# Final outcomes
game_final = {}
for gid, snaps in games.items():
    last = snaps[-1]
    if last['h_score'] > last['a_score']:
        game_final[gid] = 0.999
    elif last['h_score'] < last['a_score']:
        game_final[gid] = 0.001
    else:
        game_final[gid] = last['home_mkt']

print(f"Loaded {len(all_snaps)} NBA snapshots, {len(games)} games\n")

# ── Replay function ──────────────────────────────────────────────────
def replay(k_discount=None, sell_only=False, buy_only=False,
           price_lo=0.45, price_hi=0.65, edge_min=0.10, max_min=36):
    trades = []
    for gid, snaps in games.items():
        if gid not in game_final:
            continue
        traded_dirs = set()
        for s in snaps:
            if s['elapsed'] > max_min:
                continue
            model_p = nba_prob_discount(
                s['h_score'], s['a_score'], s['adj_sec'],
                s['pregame'], k_discount
            )
            mkt_p = s['home_mkt']
            edge = model_p - mkt_p
            if abs(edge) < edge_min:
                continue
            if not (price_lo <= mkt_p <= price_hi):
                continue
            direction = "BUY" if edge > 0 else "SELL"
            if sell_only and direction != "SELL":
                continue
            if buy_only and direction != "BUY":
                continue
            if direction in traded_dirs:
                continue
            traded_dirs.add(direction)
            final_p = game_final[gid]
            if direction == "BUY":
                pnl = (final_p - mkt_p) * 200
            else:
                pnl = (mkt_p - final_p) * 200
            trades.append({
                'game_id': gid, 'direction': direction,
                'entry': mkt_p, 'final': final_p,
                'edge': edge, 'model_p': model_p, 'mkt_p': mkt_p,
                'pnl': pnl, 'elapsed': s['elapsed'],
                'h_score': s['h_score'], 'a_score': s['a_score'],
                'raw_lead': s['h_score'] - s['a_score'],
            })
    return trades

def stats(trades):
    if not trades:
        return {'n': 0, 'wr': 0, 'pnl': 0, 'avg': 0, 'edge': 0, 'dd': 0, 'sharpe': 0}
    w = sum(1 for t in trades if t['pnl'] > 0)
    pnls = [t['pnl'] for t in trades]
    tot = sum(pnls)
    mu = tot / len(pnls)
    var = sum((p-mu)**2 for p in pnls) / max(1, len(pnls)-1)
    std = var**0.5
    cum = peak = dd = 0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        d = peak - cum
        if d > dd: dd = d
    return {
        'n': len(trades), 'wr': w/len(trades)*100, 'pnl': tot,
        'avg': mu, 'edge': sum(abs(t['edge']) for t in trades)/len(trades),
        'dd': dd, 'sharpe': mu/std if std > 0 else 0,
    }

# ═════════════════════════════════════════════════════════════════════
#  PART 1 — k SWEEP (SELL-only, max min 36)
# ═════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  k-SWEEP — SELL-ONLY, [0.45-0.65], edge≥0.10, max min 36")
print("=" * 80)
header = f"{'k':>8} | {'Trades':>6} | {'WR%':>5} | {'Avg Edge':>8} | {'Avg PnL':>8} | {'Total PnL':>9} | {'MaxDD':>7} | {'Sharpe':>6}"
print(f"\n{header}")
print("-" * 80)

configs = [
    ("baseline", None),
    ("k=1.50", 1.50),
    ("k=2.00", 2.00),
    ("k=2.50", 2.50),
    ("k=3.00", 3.00),
]

best_k = None
best_pnl = -99999
k_results = {}

for label, k in configs:
    t = replay(k_discount=k, sell_only=True)
    r = stats(t)
    k_results[label] = (t, r)
    print(f"{label:>8} | {r['n']:6d} | {r['wr']:5.1f} | {r['edge']:8.3f} | "
          f"${r['avg']:7.2f} | ${r['pnl']:8.2f} | ${r['dd']:6.2f} | {r['sharpe']:6.3f}")
    if r['pnl'] > best_pnl:
        best_pnl = r['pnl']
        best_k = label

print(f"\nBest: {best_k} (PnL=${best_pnl:.2f})")

# Per-trade detail for best
print(f"\n--- Per-trade detail ({best_k}) ---")
best_trades, _ = k_results[best_k]
for t in best_trades:
    m = "WIN" if t['pnl'] > 0 else "LOSS"
    print(f"  [{m:4s}] SELL @ {t['entry']:.3f} → {t['final']:.3f} | "
          f"edge={t['edge']:+.3f} model={t['model_p']:.3f} mkt={t['mkt_p']:.3f} | "
          f"min {t['elapsed']:.0f} | {t['h_score']}-{t['a_score']} (lead={t['raw_lead']:+d}) | ${t['pnl']:+.2f}")

# ═════════════════════════════════════════════════════════════════════
#  PART 2 — SELL-only vs ALL (with best k)
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"  DIRECTION COMPARISON (best k)")
print("=" * 80)

# Use the actual best k value
best_k_val = None
for label, k in configs:
    if label == best_k:
        best_k_val = k
        break

for mode_label, sell_o, buy_o in [("SELL-only", True, False), ("BUY-only", False, True), ("ALL", False, False)]:
    t = replay(k_discount=best_k_val, sell_only=sell_o, buy_only=buy_o)
    r = stats(t)
    print(f"  {mode_label:12s}: {r['n']:2d} trades, WR={r['wr']:5.1f}%, PnL=${r['pnl']:+8.2f}, "
          f"Avg=${r['avg']:+7.2f}, DD=${r['dd']:.2f}")

# ═════════════════════════════════════════════════════════════════════
#  PART 3 — EARLY-GAME DIAGNOSTIC (min 0–6)
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  EARLY-GAME DIAGNOSTIC (min 0–6)")
print("=" * 80)

early = [s for s in all_snaps if s['elapsed'] <= 6]
print(f"\nEarly snapshots: {len(early)}")

print(f"\n{'k':>8} | {'Avg Model P':>11} | {'Avg Mkt P':>9} | {'Avg |Δ|':>8} | {'Avg Misprice':>12} | {'BUY signals':>11}")
print("-" * 75)

for label, k in configs:
    model_ps = []
    mkt_ps = []
    deltas = []
    buy_sigs = 0
    for s in early:
        mp = nba_prob_discount(s['h_score'], s['a_score'], s['adj_sec'], s['pregame'], k)
        mkp = s['home_mkt']
        model_ps.append(mp)
        mkt_ps.append(mkp)
        deltas.append(mp - mkp)
        if mp - mkp > 0.10:  # Would generate BUY signal
            buy_sigs += 1
    avg_model = sum(model_ps) / len(model_ps)
    avg_mkt = sum(mkt_ps) / len(mkt_ps)
    avg_abs_delta = sum(abs(d) for d in deltas) / len(deltas)
    avg_misprice = sum(deltas) / len(deltas)
    print(f"{label:>8} | {avg_model:11.4f} | {avg_mkt:9.4f} | {avg_abs_delta:8.4f} | "
          f"{avg_misprice:+12.4f} | {buy_sigs:11d}")

# Show how the discount changes specific early-game snapshots
print("\n--- Sample early-game snapshots (min 1-3, lead ≥ 2) ---")
print(f"{'Score':>8} | {'Min':>4} | {'Mkt':>5} | {'Base':>5} | {'k=1.5':>5} | {'k=2.0':>5} | {'k=2.5':>5} | {'k=3.0':>5}")
print("-" * 60)

shown = set()
for s in early:
    lead = abs(s['h_score'] - s['a_score'])
    if lead < 2 or s['elapsed'] > 3:
        continue
    key = (s['h_score'], s['a_score'], int(s['elapsed']))
    if key in shown:
        continue
    shown.add(key)
    if len(shown) > 10:
        break
    probs = []
    for _, k in configs:
        p = nba_prob_discount(s['h_score'], s['a_score'], s['adj_sec'], s['pregame'], k)
        probs.append(p)
    print(f"{s['h_score']:3d}-{s['a_score']:<3d} | {s['elapsed']:4.0f} | {s['home_mkt']:5.3f} | "
          + " | ".join(f"{p:5.3f}" for p in probs))

# ═════════════════════════════════════════════════════════════════════
#  PART 4 — BUY RE-ENABLEMENT TEST (with best k)
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"  BUY RE-ENABLEMENT TEST ({best_k}, different edge thresholds)")
print("=" * 80)

print(f"\n{'Edge ≥':>8} | {'ALL Trades':>10} | {'ALL WR%':>7} | {'ALL PnL':>9} | {'BUY-only':>9} | {'BUY WR%':>7} | {'BUY PnL':>8}")
print("-" * 75)

for edge_th in [0.10, 0.15, 0.20, 0.25]:
    t_all = replay(k_discount=best_k_val, sell_only=False, edge_min=edge_th)
    t_buy = replay(k_discount=best_k_val, buy_only=True, edge_min=edge_th)
    r_all = stats(t_all)
    r_buy = stats(t_buy)
    print(f"  {edge_th:6.2f} | {r_all['n']:10d} | {r_all['wr']:6.1f}% | ${r_all['pnl']:8.2f} | "
          f"{r_buy['n']:9d} | {r_buy['wr']:6.1f}% | ${r_buy['pnl']:7.2f}")

# ═════════════════════════════════════════════════════════════════════
#  CONCLUSION
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CONCLUSION")
print("=" * 80)

best_sell, best_sell_r = k_results[best_k]
baseline_sell, baseline_r = k_results["baseline"]

print(f"\n  Best k: {best_k}")
print(f"  SELL-only baseline:  {baseline_r['n']} trades, WR={baseline_r['wr']:.0f}%, PnL=${baseline_r['pnl']:+.2f}")
print(f"  SELL-only best k:    {best_sell_r['n']} trades, WR={best_sell_r['wr']:.0f}%, PnL=${best_sell_r['pnl']:+.2f}")
print(f"  Delta:               PnL ${best_sell_r['pnl'] - baseline_r['pnl']:+.2f}")

# Check BUY with best k at edge ≥ 0.20
t_buy_strict = replay(k_discount=best_k_val, buy_only=True, edge_min=0.20)
r_buy_strict = stats(t_buy_strict)
buy_safe = r_buy_strict['pnl'] > 0 and r_buy_strict['wr'] >= 50

print(f"\n  BUY re-enable (edge≥0.20): {r_buy_strict['n']} trades, WR={r_buy_strict['wr']:.0f}%, PnL=${r_buy_strict['pnl']:+.2f}")
print(f"  BUY re-enable verdict: {'SAFE' if buy_safe else 'NOT YET — keep SELL-only'}")
print(f"  Model salvageable: {'YES — with lead discount' if best_sell_r['pnl'] > baseline_r['pnl'] else 'MARGINAL — discount helps but edge is thin'}")
