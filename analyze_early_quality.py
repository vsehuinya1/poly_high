#!/usr/bin/env python3
"""
Can we predict tradeable games from the first 5 minutes of book data?

For each game, compute early-window metrics:
  1. Average spread (first 5 min of SNAP data)
  2. Avg bid/ask depth
  3. Update frequency (price_changes per minute from WS_DIAG)
  4. Book age stability (avg book_age in first 5 min)

Then compare against which games were actually tradeable later.
"""
import re
from collections import defaultdict

LOG_FILE = "/root/poly_high_sports/sports_system.log"

# ── Parse SNAP lines ──────────────────────────────────────────────
SNAP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*SNAP (\w+) (\d+)-(\d+) \| "
    r"adj=\s*([\d.]+) σ=([\d.]+) seff=([-\d.]+) z=([-\d.]+) \| "
    r"model=([\d.]+) \| "
    r"bid=([\d.]+) ask=([\d.]+) mid=([\d.]+) \| "
    r"bsz=([\d.]+) asz=([\d.]+) sprd=([\d.]+) age=([-\d.]+)s \| "
    r"edge=([-+\d.]+)"
)

# Collect all SNAPs per team
team_snaps = defaultdict(list)
with open(LOG_FILE) as f:
    for line in f:
        m = SNAP_RE.search(line)
        if m:
            s = {
                "time": m.group(1),
                "team": m.group(2),
                "hs": int(m.group(3)),
                "as_": int(m.group(4)),
                "adj_sec": float(m.group(5)),
                "model": float(m.group(9)),
                "bid": float(m.group(10)),
                "ask": float(m.group(11)),
                "mid": float(m.group(12)),
                "bsz": float(m.group(13)),
                "asz": float(m.group(14)),
                "sprd": float(m.group(15)),
                "age": float(m.group(16)),
                "edge": float(m.group(17)),
            }
            team_snaps[s["team"]].append(s)

# ── Compute metrics ───────────────────────────────────────────────
# "First 5 min" = first 60 SNAP lines (~5 min at 5s intervals)
EARLY_WINDOW = 60

results = []
for team, snaps in sorted(team_snaps.items()):
    total = len(snaps)
    early = snaps[:EARLY_WINDOW]
    late = snaps[EARLY_WINDOW:]
    
    if not early:
        continue
    
    # Early window metrics
    early_avg_sprd = sum(s["sprd"] for s in early) / len(early)
    early_avg_age = sum(s["age"] for s in early) / len(early)
    early_avg_bsz = sum(s["bsz"] for s in early) / len(early)
    early_avg_asz = sum(s["asz"] for s in early) / len(early)
    early_avg_depth = early_avg_bsz + early_avg_asz
    
    # Count how many early ticks had fresh data (age < 20s)
    early_fresh = sum(1 for s in early if s["age"] <= 20)
    early_fresh_pct = 100 * early_fresh / len(early)
    
    # Count how many early ticks had tight spread
    early_tight = sum(1 for s in early if s["sprd"] <= 0.08)
    early_tight_pct = 100 * early_tight / len(early)
    
    # Two-sided book: both bid>0 and ask>0
    early_twosided = sum(1 for s in early if s["bid"] > 0 and s["ask"] > 0 and s["ask"] < 1.0)
    early_twosided_pct = 100 * early_twosided / len(early)
    
    # Full game metrics (for comparison)
    all_tradeable = sum(1 for s in snaps if s["sprd"] <= 0.08 and s["age"] <= 20 and 0.08 <= s["mid"] <= 0.92)
    all_tradeable_pct = 100 * all_tradeable / total if total > 0 else 0
    
    all_edge = sum(1 for s in snaps if s["sprd"] <= 0.08 and s["age"] <= 20 and 0.08 <= s["mid"] <= 0.92 and abs(s["edge"]) >= 0.07)
    
    # Composite early score (0-100)
    # Weight: 40% fresh books, 30% tight spread, 20% two-sided, 10% depth
    depth_score = min(100, early_avg_depth / 5)  # 500+ total depth = 100
    composite = 0.40 * early_fresh_pct + 0.30 * early_tight_pct + 0.20 * early_twosided_pct + 0.10 * depth_score
    
    results.append({
        "team": team,
        "total_ticks": total,
        "early_avg_sprd": early_avg_sprd,
        "early_avg_age": early_avg_age,
        "early_avg_depth": early_avg_depth,
        "early_fresh_pct": early_fresh_pct,
        "early_tight_pct": early_tight_pct,
        "early_twosided_pct": early_twosided_pct,
        "composite": composite,
        "tradeable_ticks": all_tradeable,
        "tradeable_pct": all_tradeable_pct,
        "edge_ticks": all_edge,
    })

# ── Sort by composite score ──────────────────────────────────────
results.sort(key=lambda x: -x["composite"])

print("=" * 110)
print("EARLY-WINDOW BOOK QUALITY vs ACTUAL TRADEABILITY")
print("Early window = first 60 ticks (~5 minutes)")
print("=" * 110)
print()
print(f"{'Team':<6} {'Comp':>5} | {'Sprd':>6} {'Age':>6} {'Depth':>6} {'Fresh%':>6} {'Tight%':>6} {'2Side%':>6} | {'Tradeable':>9} {'%':>6} {'Edge':>5} | {'Verdict'}")
print("-" * 110)

for r in results:
    # Verdict
    if r["tradeable_pct"] >= 10:
        verdict = "✅ TRADEABLE"
    elif r["tradeable_pct"] >= 1:
        verdict = "⚠️  MARGINAL"
    else:
        verdict = "❌ GARBAGE"
    
    print(f"{r['team']:<6} {r['composite']:>5.1f} | "
          f"{r['early_avg_sprd']:>6.3f} {r['early_avg_age']:>6.0f} {r['early_avg_depth']:>6.0f} "
          f"{r['early_fresh_pct']:>5.1f}% {r['early_tight_pct']:>5.1f}% {r['early_twosided_pct']:>5.1f}% | "
          f"{r['tradeable_ticks']:>9} {r['tradeable_pct']:>5.1f}% {r['edge_ticks']:>5} | "
          f"{verdict}")

# ── Would a composite threshold eliminate garbage? ────────────────
print()
print("=" * 110)
print("THRESHOLD ANALYSIS")
print("=" * 110)

for threshold in [10, 20, 30, 40, 50]:
    kept = [r for r in results if r["composite"] >= threshold]
    removed = [r for r in results if r["composite"] < threshold]
    
    kept_tradeable = sum(r["tradeable_ticks"] for r in kept)
    removed_tradeable = sum(r["tradeable_ticks"] for r in removed)
    total_tradeable = kept_tradeable + removed_tradeable
    
    kept_garbage = sum(r["total_ticks"] - r["tradeable_ticks"] for r in kept)
    removed_garbage = sum(r["total_ticks"] - r["tradeable_ticks"] for r in removed)
    total_garbage = kept_garbage + removed_garbage
    
    # What % of garbage was eliminated?
    garbage_eliminated_pct = 100 * removed_garbage / total_garbage if total_garbage > 0 else 0
    # What % of tradeable ticks were kept?
    tradeable_kept_pct = 100 * kept_tradeable / total_tradeable if total_tradeable > 0 else 0
    
    print(f"  Threshold ≥ {threshold:>2}: "
          f"keep {len(kept)}/{len(results)} games | "
          f"garbage eliminated: {garbage_eliminated_pct:.0f}% | "
          f"tradeable preserved: {tradeable_kept_pct:.0f}% | "
          f"tradeable ticks: {kept_tradeable}/{total_tradeable}")

# ── Per-game early vs late correlation ────────────────────────────
print()
print("=" * 110)
print("EARLY SCORE vs LATE OUTCOME — CORRELATION")
print("=" * 110)
print()
print("If composite ≥ 30 = 'trade this game', how would we have done?")
print()
good_games = [r for r in results if r["composite"] >= 30]
bad_games = [r for r in results if r["composite"] < 30]
print(f"  PASSED (composite ≥ 30): {[r['team'] for r in good_games]}")
print(f"  FAILED (composite < 30): {[r['team'] for r in bad_games]}")
print()
print(f"  Tradeable ticks in PASSED games: {sum(r['tradeable_ticks'] for r in good_games)}")
print(f"  Tradeable ticks in FAILED games: {sum(r['tradeable_ticks'] for r in bad_games)}")
print(f"  Edge opportunities in PASSED:    {sum(r['edge_ticks'] for r in good_games)}")
print(f"  Edge opportunities in FAILED:    {sum(r['edge_ticks'] for r in bad_games)}")
print()

# How many ticks would we process?
total_ticks = sum(r["total_ticks"] for r in results)
kept_ticks = sum(r["total_ticks"] for r in good_games)
print(f"  Total ticks reduced: {total_ticks} → {kept_ticks} ({100*kept_ticks/total_ticks:.0f}%)")
print(f"  Garbage ticks eliminated: {total_ticks - kept_ticks - sum(r['tradeable_ticks'] for r in bad_games)}")
