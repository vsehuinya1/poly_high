#!/usr/bin/env python3
"""
Segmented PnL analysis for last night's run.
Parses SNAP log lines + paper_trades CSV.
Filters, segments by probability bucket and liquidity quality.
"""
import csv
import re
import sys
from collections import defaultdict

LOG_FILE = "/root/poly_high_sports/sports_system.log"
TRADES_FILE = "/root/poly_high_sports/sports_data/paper_trades_20260225.csv"

# ── Parse all SNAP lines from log ─────────────────────────────────
# Format: SNAP Pac 38-30 | adj=2182 σ=1.70 seff=8.0 z=0.78 | model=0.782 | bid=0.550 ask=0.620 mid=0.585 | bsz=7 asz=17 sprd=0.070 age=277s | edge=+0.1974
SNAP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*SNAP (\w+) (\d+)-(\d+) \| "
    r"adj=\s*([\d.]+) σ=([\d.]+) seff=([-\d.]+) z=([-\d.]+) \| "
    r"model=([\d.]+) \| "
    r"bid=([\d.]+) ask=([\d.]+) mid=([\d.]+) \| "
    r"bsz=([\d.]+) asz=([\d.]+) sprd=([\d.]+) age=([-\d.]+)s \| "
    r"edge=([-+\d.]+)"
)

snaps = []
with open(LOG_FILE) as f:
    for line in f:
        m = SNAP_RE.search(line)
        if m:
            snaps.append({
                "time": m.group(1),
                "team": m.group(2),
                "hs": int(m.group(3)),
                "as_": int(m.group(4)),
                "adj_sec": float(m.group(5)),
                "sigma": float(m.group(6)),
                "seff": float(m.group(7)),
                "z": float(m.group(8)),
                "model": float(m.group(9)),
                "bid": float(m.group(10)),
                "ask": float(m.group(11)),
                "mid": float(m.group(12)),
                "bsz": float(m.group(13)),
                "asz": float(m.group(14)),
                "sprd": float(m.group(15)),
                "age": float(m.group(16)),
                "edge": float(m.group(17)),
            })

print(f"Parsed {len(snaps)} SNAP lines from log")

# ── Parse paper trades ────────────────────────────────────────────
TRADE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*PAPER (ENTRY|EXIT): (P\d+) (BUY|SELL) (\w+) @ ([\d.]+)"
)
ENTRY_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*PAPER ENTRY: (P\d+) (BUY|SELL) (\w+) @ ([\d.]+) \(\$(\d+)\) edge=([-\d.]+) \| (.+)"
)
EXIT_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*PAPER EXIT: (P\d+) (\w+) @ ([\d.]+) → ([\d.]+) PnL=\$([-\d.]+) \((\w+)\)"
)

entries = {}
trades = []

with open(LOG_FILE) as f:
    for line in f:
        em = ENTRY_RE.search(line)
        if em:
            pid = em.group(2)
            entries[pid] = {
                "time": em.group(1),
                "pid": pid,
                "direction": em.group(3),
                "outcome": em.group(4),
                "entry_price": float(em.group(5)),
                "size": float(em.group(6)),
                "edge": float(em.group(7)),
                "game_state": em.group(8),
            }
            continue
        
        xm = EXIT_RE.search(line)
        if xm:
            pid = xm.group(2)
            entry = entries.get(pid, {})
            trade = {
                **entry,
                "exit_time": xm.group(1),
                "exit_price": float(xm.group(5)),
                "pnl": float(xm.group(6)),
                "exit_reason": xm.group(7),
            }
            # Fix PnL sign from log (log shows absolute, need to check direction)
            if "PnL=$-" in line:
                trade["pnl"] = -abs(trade["pnl"])
            trades.append(trade)

print(f"Parsed {len(trades)} completed trades")

# ── Find closest SNAP for each trade entry ────────────────────────
def find_snap_at_trade(trade, snaps):
    """Find the SNAP line closest to trade entry time."""
    t_time = trade.get("time", "")
    team_prefix = ""
    gs = trade.get("game_state", "")
    
    # Try to match game from game_state
    best = None
    best_dist = 999999
    for s in snaps:
        if s["time"] == t_time:
            return s
        # Simple time distance (just compare HH:MM:SS strings)
        if s["time"] <= t_time:
            # Use last SNAP before trade
            best = s
    return best

# ── ANALYSIS ──────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SECTION 1: ALL TRADES — RAW")
print("=" * 80)
print(f"{'PID':<6} {'Dir':<5} {'Out':<5} {'Entry':>6} {'Exit':>6} {'PnL':>9} {'Edge':>7} {'Size':>5} {'Reason':<12} {'Game'}")
print("-" * 95)
total_pnl = 0
for t in trades:
    total_pnl += t["pnl"]
    print(f"{t.get('pid','?'):<6} {t.get('direction','?'):<5} {t.get('outcome','?'):<5} "
          f"{t.get('entry_price',0):>6.3f} {t.get('exit_price',0):>6.3f} "
          f"{t['pnl']:>+9.2f} {t.get('edge',0):>+7.3f} {t.get('size',0):>5.0f} "
          f"{t.get('exit_reason','?'):<12} {t.get('game_state','?')[:40]}")
print(f"\nTotal PnL: ${total_pnl:+.2f} ({len(trades)} trades, {sum(1 for t in trades if t['pnl']>0)} wins)")

# ── FILTER: Remove extreme prices, bad spread, stale books ───────
print("\n" + "=" * 80)
print("SECTION 2: FILTERED TRADES")
print("Filters: entry_price in [0.08, 0.92], spread ≤ 0.08, book_age ≤ 20s")
print("=" * 80)

# For each trade, find matching SNAP to get spread/age
# We'll use the SNAP lines to build a per-team time-indexed lookup
team_snaps = defaultdict(list)
for s in snaps:
    team_snaps[s["team"]].append(s)

def get_book_quality(trade):
    """Get spread and book_age at trade entry time from SNAP data."""
    gs = trade.get("game_state", "")
    entry_time = trade.get("time", "")
    
    # Try to extract team abbreviation from game state
    # E.g. "Nets 93-101 Mavericks (38')" → try "Net"
    parts = gs.split()
    if parts:
        team3 = parts[0][:3]
    else:
        team3 = ""
    
    # Find closest SNAP for this team before entry time
    candidates = team_snaps.get(team3, [])
    best = None
    for s in candidates:
        if s["time"] <= entry_time:
            best = s
        elif best:
            break
    
    if best:
        return best["sprd"], best["age"], best["mid"], best["bid"], best["ask"], best["bsz"], best["asz"]
    return None, None, None, None, None, None, None

filtered = []
removed = []
for t in trades:
    ep = t.get("entry_price", 0)
    sprd, age, mid, bid, ask, bsz, asz = get_book_quality(t)
    
    reasons = []
    if ep > 0.92 or ep < 0.08:
        reasons.append(f"EXTREME_PRICE({ep:.3f})")
    if sprd is not None and sprd > 0.08:
        reasons.append(f"WIDE_SPREAD({sprd:.3f})")
    if age is not None and age > 20:
        reasons.append(f"STALE_BOOK({age:.0f}s)")
    if sprd is None:
        reasons.append("NO_SNAP_DATA")
    
    t["_sprd"] = sprd
    t["_age"] = age
    t["_mid"] = mid
    t["_bid"] = bid
    t["_ask"] = ask
    
    if reasons:
        t["_filter_reason"] = " | ".join(reasons)
        removed.append(t)
    else:
        filtered.append(t)

print(f"\nKept: {len(filtered)} trades")
print(f"Removed: {len(removed)} trades")

if removed:
    print(f"\nRemoved trades:")
    print(f"{'PID':<6} {'Entry':>6} {'PnL':>9} {'Spread':>7} {'Age':>5} {'Reason'}")
    print("-" * 80)
    for t in removed:
        print(f"{t.get('pid','?'):<6} {t.get('entry_price',0):>6.3f} {t['pnl']:>+9.2f} "
              f"{t.get('_sprd','?'):>7} {t.get('_age','?'):>5} "
              f"{t.get('_filter_reason','')}")

removed_pnl = sum(t["pnl"] for t in removed)
filtered_pnl = sum(t["pnl"] for t in filtered)
print(f"\nRemoved trades PnL: ${removed_pnl:+.2f}")
print(f"Filtered trades PnL: ${filtered_pnl:+.2f}")

# ── SEGMENT BY PROBABILITY BUCKET ────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 3: PnL BY PROBABILITY BAND (filtered trades only)")
print("=" * 80)

buckets = {
    "0.08-0.20": (0.08, 0.20),
    "0.20-0.35": (0.20, 0.35),
    "0.35-0.50": (0.35, 0.50),
    "0.50-0.65": (0.50, 0.65),
    "0.65-0.80": (0.65, 0.80),
    "0.80-0.92": (0.80, 0.92),
}

bucket_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0, "entries": []})

for t in filtered:
    ep = t.get("entry_price", 0)
    placed = False
    for label, (lo, hi) in buckets.items():
        if lo <= ep <= hi:
            bucket_stats[label]["trades"] += 1
            bucket_stats[label]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                bucket_stats[label]["wins"] += 1
            bucket_stats[label]["entries"].append(t)
            placed = True
            break
    if not placed:
        bucket_stats["other"]["trades"] += 1
        bucket_stats["other"]["pnl"] += t["pnl"]

print(f"\n{'Bucket':<12} {'Trades':>6} {'Wins':>5} {'WinRate':>8} {'PnL':>10} {'Avg PnL':>9}")
print("-" * 55)
for label in sorted(bucket_stats.keys()):
    b = bucket_stats[label]
    wr = b["wins"] / b["trades"] * 100 if b["trades"] > 0 else 0
    avg = b["pnl"] / b["trades"] if b["trades"] > 0 else 0
    print(f"{label:<12} {b['trades']:>6} {b['wins']:>5} {wr:>7.1f}% {b['pnl']:>+10.2f} {avg:>+9.2f}")

# ── SECTION 4: SNAP-level analysis — how many ticks had tradeable books?
print("\n" + "=" * 80)
print("SECTION 4: BOOK QUALITY DISTRIBUTION (all SNAP ticks)")
print("=" * 80)

total_snaps = len(snaps)
tradeable = [s for s in snaps if s["sprd"] <= 0.08 and s["age"] <= 20 and 0.08 <= s["mid"] <= 0.92]
edge_and_tradeable = [s for s in tradeable if abs(s["edge"]) >= 0.05]
strong_edge = [s for s in tradeable if abs(s["edge"]) >= 0.07]

print(f"  Total SNAP ticks:              {total_snaps}")
print(f"  Tradeable (sprd≤0.08, age≤20s, mid 0.08-0.92): {len(tradeable)} ({100*len(tradeable)/total_snaps:.1f}%)")
print(f"  + Edge ≥ 5%:                   {len(edge_and_tradeable)} ({100*len(edge_and_tradeable)/total_snaps:.1f}%)")
print(f"  + Edge ≥ 7%:                   {len(strong_edge)} ({100*len(strong_edge)/total_snaps:.1f}%)")

# By team
print(f"\n  By game:")
team_stats = defaultdict(lambda: {"total": 0, "tradeable": 0, "edge": 0})
for s in snaps:
    team_stats[s["team"]]["total"] += 1
    if s["sprd"] <= 0.08 and s["age"] <= 20 and 0.08 <= s["mid"] <= 0.92:
        team_stats[s["team"]]["tradeable"] += 1
        if abs(s["edge"]) >= 0.05:
            team_stats[s["team"]]["edge"] += 1

print(f"  {'Team':<6} {'Ticks':>6} {'Tradeable':>10} {'%':>6} {'w/Edge':>7}")
print(f"  {'-'*40}")
for team in sorted(team_stats.keys(), key=lambda x: -team_stats[x]["tradeable"]):
    ts = team_stats[team]
    pct = 100 * ts["tradeable"] / ts["total"] if ts["total"] > 0 else 0
    print(f"  {team:<6} {ts['total']:>6} {ts['tradeable']:>10} {pct:>5.1f}% {ts['edge']:>7}")

# ── VERDICT ───────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
print(f"  Raw PnL (all trades):      ${total_pnl:+.2f} ({len(trades)} trades)")
print(f"  Filtered PnL:              ${filtered_pnl:+.2f} ({len(filtered)} trades)")
print(f"  Removed by filters:        ${removed_pnl:+.2f} ({len(removed)} trades)")
if filtered:
    print(f"  Filtered win rate:         {100*sum(1 for t in filtered if t['pnl']>0)/len(filtered):.0f}%")
