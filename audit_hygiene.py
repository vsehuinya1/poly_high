#!/usr/bin/env python3
"""
NBA v1 Execution Hygiene Audit — Pure validation, no code changes.
Loads most recent data from sports_data/ and system log.
"""
import csv, re, os, glob, time
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("/root/poly_high_sports/sports_data")
LOG_FILE = "/root/poly_high_sports/sports_system.log"

# Find most recent date
csv_files = sorted(glob.glob(str(DATA_DIR / "snapshots_*.csv")))
if not csv_files:
    print("ERROR: No snapshot files found")
    exit(1)
latest_date = csv_files[-1].split("_")[-1].replace(".csv", "")
print(f"Auditing date: {latest_date}")

SNAP_FILE = DATA_DIR / f"snapshots_{latest_date}.csv"
SIG_FILE = DATA_DIR / f"signals_{latest_date}.csv"
TRADE_FILE = DATA_DIR / f"paper_trades_{latest_date}.csv"

# ── Load snapshots ────────────────────────────────────────────────
snapshots_by_game = defaultdict(list)
all_snapshots = []
with open(SNAP_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["_ts"] = float(row["timestamp"])
        row["_home_score"] = int(row.get("home_score", 0))
        row["_away_score"] = int(row.get("away_score", 0))
        row["_home_p_model"] = float(row.get("home_p_model", 0))
        row["_away_p_model"] = float(row.get("away_p_model", 0))
        row["_home_p_mkt"] = float(row.get("home_p_mkt", 0))
        row["_away_p_mkt"] = float(row.get("away_p_mkt", 0))
        row["_edge"] = float(row.get("edge", 0))
        row["_adj_sec"] = float(row.get("adjusted_seconds", 0))
        snapshots_by_game[row["game_id"]].append(row)
        all_snapshots.append(row)

print(f"Loaded {len(all_snapshots)} snapshots across {len(snapshots_by_game)} games")

# ── Load trades ───────────────────────────────────────────────────
trades_raw = []
with open(TRADE_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        trades_raw.append(row)

print(f"Loaded {len(trades_raw)} trade events")

# Pair entries and exits
entries = {}
completed_trades = []
for row in trades_raw:
    if row["event"] == "ENTRY":
        entries[row["position_id"]] = row
    elif row["event"] == "EXIT":
        entry = entries.get(row["position_id"], {})
        completed_trades.append({
            "pid": row["position_id"],
            "game_id": row.get("game_id", entry.get("game_id", "")),
            "token_id": row.get("token_id", entry.get("token_id", "")),
            "outcome": row.get("outcome", entry.get("outcome", "")),
            "direction": entry.get("direction", ""),
            "entry_price": float(entry.get("entry_price", 0)),
            "exit_price": float(row.get("exit_price", 0)),
            "entry_edge": float(entry.get("entry_edge", 0)),
            "size": float(entry.get("size_usd", 0)),
            "pnl": float(row.get("pnl", 0)),
            "exit_reason": row.get("exit_reason", ""),
            "game_state_entry": entry.get("game_state", ""),
            "game_state_exit": row.get("game_state", ""),
            "entry_ts": float(entry.get("timestamp", 0)),
            "exit_ts": float(row.get("timestamp", 0)),
        })

# If only entries (no exits), still list them
open_positions = []
for pid, entry in entries.items():
    if not any(t["pid"] == pid for t in completed_trades):
        open_positions.append({
            "pid": pid,
            "game_id": entry.get("game_id", ""),
            "direction": entry.get("direction", ""),
            "outcome": entry.get("outcome", ""),
            "entry_price": float(entry.get("entry_price", 0)),
            "entry_edge": float(entry.get("entry_edge", 0)),
            "size": float(entry.get("size_usd", 0)),
            "game_state_entry": entry.get("game_state", ""),
            "entry_ts": float(entry.get("timestamp", 0)),
        })

print(f"Completed trades: {len(completed_trades)}, Open positions: {len(open_positions)}")

# ── Parse SNAP log lines for book quality at trade time ───────────
SNAP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*SNAP (\w+) (\d+)-(\d+) \| "
    r"adj=\s*([\d.]+) .*?"
    r"bid=([\d.]+) ask=([\d.]+) mid=([\d.]+) \| "
    r"bsz=([\d.]+) asz=([\d.]+) sprd=([\d.]+) age=([-\d.]+)s \| "
    r"edge=([-+\d.]+) \| "
    r"gs=(\w+) cd=([\d.]+) gpnl=\$([-\d.]+)"
)

# Also parse old format without gs= (in case log was started pre-v1.4)
SNAP_RE_OLD = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*SNAP (\w+) (\d+)-(\d+) \| "
    r"adj=\s*([\d.]+) .*?"
    r"bid=([\d.]+) ask=([\d.]+) mid=([\d.]+) \| "
    r"bsz=([\d.]+) asz=([\d.]+) sprd=([\d.]+) age=([-\d.]+)s \| "
    r"edge=([-+\d.]+)"
)

ENTRY_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*PAPER ENTRY: (P\d+) (BUY|SELL) (\w+) @ ([\d.]+) \(\$(\d+)\) edge=([-\d.]+) \| (.+)"
)
EXIT_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*PAPER EXIT: (P\d+) (\w+) @ ([\d.]+) . ([\d.]+) PnL=\$([-\d.]+) \((\w+)\)"
)

GATE_RE = re.compile(r"(\d{2}:\d{2}:\d{2}).*?(GAME_ACTIVATED|GAME_DEACTIVATED|GAME_FROZEN|GAME_UNFROZEN|GAME_STOPPED)")

snap_log = []  # all SNAP lines from log
trade_entries_log = {}
trade_exits_log = {}
gate_transitions = []

with open(LOG_FILE) as f:
    for line in f:
        m = SNAP_RE.search(line)
        if m:
            snap_log.append({
                "time": m.group(1), "team": m.group(2),
                "hs": int(m.group(3)), "as_": int(m.group(4)),
                "adj_sec": float(m.group(5)),
                "bid": float(m.group(6)), "ask": float(m.group(7)),
                "mid": float(m.group(8)),
                "bsz": float(m.group(9)), "asz": float(m.group(10)),
                "sprd": float(m.group(11)), "age": float(m.group(12)),
                "edge": float(m.group(13)),
                "gs": m.group(14), "cd": float(m.group(15)),
                "gpnl": float(m.group(16)),
            })
            continue

        m = SNAP_RE_OLD.search(line)
        if m:
            snap_log.append({
                "time": m.group(1), "team": m.group(2),
                "hs": int(m.group(3)), "as_": int(m.group(4)),
                "adj_sec": float(m.group(5)),
                "bid": float(m.group(6)), "ask": float(m.group(7)),
                "mid": float(m.group(8)),
                "bsz": float(m.group(9)), "asz": float(m.group(10)),
                "sprd": float(m.group(11)), "age": float(m.group(12)),
                "edge": float(m.group(13)),
                "gs": "UNKNOWN", "cd": 0, "gpnl": 0,
            })
            continue

        em = ENTRY_RE.search(line)
        if em:
            trade_entries_log[em.group(2)] = {
                "time": em.group(1), "direction": em.group(3),
                "outcome": em.group(4), "entry_price": float(em.group(5)),
                "size": float(em.group(6)), "edge": float(em.group(7)),
                "game_state": em.group(8),
            }
            continue

        xm = EXIT_RE.search(line)
        if xm:
            pnl = float(xm.group(6))
            if "PnL=$-" in line:
                pnl = -abs(pnl)
            trade_exits_log[xm.group(2)] = {
                "time": xm.group(1), "exit_price": float(xm.group(5)),
                "pnl": pnl, "exit_reason": xm.group(7),
            }
            continue

        gm = GATE_RE.search(line)
        if gm:
            gate_transitions.append({"time": gm.group(1), "event": gm.group(2), "line": line.strip()})

print(f"Parsed {len(snap_log)} SNAP lines from log, {len(gate_transitions)} gate transitions")

# Build team-indexed SNAP lookup
team_snaps = defaultdict(list)
for s in snap_log:
    team_snaps[s["team"]].append(s)

# ── Helper: find SNAP at trade entry time ─────────────────────────
def find_snap_at_time(team3, entry_time_str):
    candidates = team_snaps.get(team3, [])
    best = None
    for s in candidates:
        if s["time"] <= entry_time_str:
            best = s
        elif best:
            break
    return best

# ══════════════════════════════════════════════════════════════════
# TASK 1: TRADE-LEVEL VALIDATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TASK 1: TRADE-LEVEL VALIDATION")
print("=" * 100)

all_trades_for_audit = completed_trades + open_positions

if not all_trades_for_audit:
    print("  No trades to audit.")
else:
    print(f"\n  {'PID':<7} {'Game State':<40} {'Dir':<5} {'Entry':>6} {'Exit':>6} {'PnL':>9} {'Edge':>7} {'Size':>5} {'Reason':<12}")
    print(f"  {'-'*100}")

    validation_results = []
    for t in all_trades_for_audit:
        pid = t.get("pid", "?")
        gs = t.get("game_state_entry", "")
        parts = gs.split()
        team3 = parts[0][:3] if parts else ""
        entry_log = trade_entries_log.get(pid, {})
        exit_log = trade_exits_log.get(pid, {})
        entry_time = entry_log.get("time", "")
        snap = find_snap_at_time(team3, entry_time)

        ep = t.get("entry_price", 0)
        xp = t.get("exit_price", 0)
        pnl = t.get("pnl", 0)
        edge = t.get("entry_edge", 0)
        sz = t.get("size", 0)
        reason = t.get("exit_reason", "OPEN")

        sprd = snap["sprd"] if snap else -1
        age = snap["age"] if snap else -1
        game_status = snap["gs"] if snap else "?"
        score_diff = abs(snap["hs"] - snap["as_"]) if snap else -1
        adj_sec = snap["adj_sec"] if snap else -1

        print(f"  {pid:<7} {gs[:40]:<40} {t.get('direction','?'):<5} {ep:>6.3f} {xp:>6.3f} {pnl:>+9.2f} {edge:>+7.3f} {sz:>5.0f} {reason:<12}")
        print(f"          snap: sprd={sprd:.3f} age={age:.0f}s score_diff={score_diff} adj_sec={adj_sec:.0f} gs={game_status}")

        # Validate gates
        violations = []
        if not (0.08 <= ep <= 0.92):
            violations.append(f"PRICE_BAND({ep:.3f})")
        if snap and sprd > 0.08:
            violations.append(f"SPREAD({sprd:.3f})")
        if snap and age > 20:
            violations.append(f"STALE_BOOK({age:.0f}s)")
        if snap and adj_sec > 600:
            violations.append(f"ADJ_SEC({adj_sec:.0f})")
        if snap and score_diff > 15:
            violations.append(f"SCORE_DIFF({score_diff})")
        if snap and game_status not in ("ACTIVE", "UNKNOWN"):
            violations.append(f"GAME_STATUS({game_status})")

        passed = "Y" if not violations else "N"
        validation_results.append({"pid": pid, "passed": passed, "violations": violations})

        viol_str = ", ".join(violations) if violations else "—"
        print(f"          GATES: {passed} | {viol_str}")
        print()

    # Summary table
    print(f"\n  GATE VALIDATION SUMMARY")
    print(f"  {'TradeID':<8} {'Passed':>7} {'Violations'}")
    print(f"  {'-'*60}")
    for r in validation_results:
        viol = ", ".join(r["violations"]) if r["violations"] else "—"
        print(f"  {r['pid']:<8} {r['passed']:>7} {viol}")

# ══════════════════════════════════════════════════════════════════
# TASK 2: GAME-LEVEL GATE VALIDATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TASK 2: GAME-LEVEL GATE VALIDATION")
print("=" * 100)

# Compute per-game freshness from SNAP log
game_stats = {}
for team, ss in sorted(team_snaps.items()):
    n = len(ss)
    if n == 0: continue
    fresh20 = sum(1 for s in ss if s["age"] <= 20)
    fresh20_pct = 100 * fresh20 / n

    # Longest consecutive fresh streak (age <= 30s)
    best_streak = streak = 0
    for s in ss:
        if s["age"] <= 30:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0
    streak_sec = best_streak * 5  # ~5s per tick

    # Was gate ever ACTIVE?
    was_active = any(s.get("gs") == "ACTIVE" for s in ss)
    final_gs = ss[-1].get("gs", "?") if ss else "?"

    # Count trades for this game
    game_trades = [t for t in all_trades_for_audit if t.get("game_state_entry", "").startswith(team[:3]) or
                   t.get("game_state_entry", "").split()[0][:3] == team[:3] if t.get("game_state_entry")]

    game_pnl = sum(t.get("pnl", 0) for t in game_trades if "pnl" in t)

    game_stats[team] = {
        "n": n, "fresh20_pct": fresh20_pct,
        "streak_sec": streak_sec, "was_active": was_active,
        "final_gs": final_gs, "trades": len(game_trades), "pnl": game_pnl,
    }

print(f"\n  {'Team':<6} {'Ticks':>5} {'Fresh20%':>8} {'Streak':>7} {'Gate':>7} {'Final_GS':>10} {'Trades':>6} {'PnL':>9}")
print(f"  {'-'*70}")
for team in sorted(game_stats.keys(), key=lambda x: -game_stats[x]["fresh20_pct"]):
    g = game_stats[team]
    gate = "PASS" if g["was_active"] else "FAIL"
    print(f"  {team:<6} {g['n']:>5} {g['fresh20_pct']:>7.1f}% {g['streak_sec']:>5}s {gate:>7} {g['final_gs']:>10} {g['trades']:>6} {g['pnl']:>+9.2f}")

# Check for trades in FAIL games
fail_games = [t for t in game_stats if not game_stats[t]["was_active"]]
trades_in_fail = [t for t in all_trades_for_audit
                  for fg in fail_games
                  if fg[:3] in t.get("game_state_entry", "")[:3]]
if trades_in_fail:
    print(f"\n  ⚠️  VIOLATION: {len(trades_in_fail)} trades in games that FAILED the gate!")
    for t in trades_in_fail:
        print(f"    {t.get('pid')}: {t.get('game_state_entry')}")
else:
    print(f"\n  ✅ No trades in games that failed the freshness gate.")

# Gate transitions log
if gate_transitions:
    print(f"\n  Gate transitions:")
    for gt in gate_transitions:
        print(f"    {gt['time']} {gt['event']}")

# ══════════════════════════════════════════════════════════════════
# TASK 3: STATISTICAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TASK 3: STATISTICAL SUMMARY")
print("=" * 100)

if completed_trades:
    wins = [t for t in completed_trades if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in completed_trades)
    avg_edge = sum(abs(t["entry_edge"]) for t in completed_trades) / len(completed_trades)
    avg_hold = sum(t["exit_ts"] - t["entry_ts"] for t in completed_trades) / len(completed_trades)

    # Avg spread and book_age at entry
    spreads = []
    ages = []
    for t in completed_trades:
        pid = t["pid"]
        gs = t.get("game_state_entry", "")
        parts = gs.split()
        team3 = parts[0][:3] if parts else ""
        entry_log = trade_entries_log.get(pid, {})
        snap = find_snap_at_time(team3, entry_log.get("time", ""))
        if snap:
            spreads.append(snap["sprd"])
            ages.append(snap["age"])

    print(f"\n  Total trades:       {len(completed_trades)}")
    print(f"  Open positions:     {len(open_positions)}")
    print(f"  Win rate:           {100*len(wins)/len(completed_trades):.0f}% ({len(wins)}/{len(completed_trades)})")
    print(f"  Avg edge at entry:  {avg_edge:.4f}")
    print(f"  Avg spread:         {sum(spreads)/len(spreads):.4f}" if spreads else "  Avg spread: N/A")
    print(f"  Avg book_age:       {sum(ages)/len(ages):.1f}s" if ages else "  Avg book_age: N/A")
    print(f"  Avg holding time:   {avg_hold:.0f}s ({avg_hold/60:.1f} min)")
    print(f"  Total PnL:          ${total_pnl:+.2f}")

    # PnL by game
    print(f"\n  PnL by game:")
    game_pnls = defaultdict(lambda: {"trades": 0, "pnl": 0})
    for t in completed_trades:
        gid = t["game_id"]
        game_pnls[gid]["trades"] += 1
        game_pnls[gid]["pnl"] += t["pnl"]
        game_pnls[gid]["state"] = t.get("game_state_entry", "")
    for gid, gp in game_pnls.items():
        print(f"    {gp['state'][:40]:<40} trades={gp['trades']} pnl=${gp['pnl']:+.2f}")

    # PnL by probability bucket
    print(f"\n  PnL by probability bucket:")
    buckets = {"0.08-0.25": (0.08,0.25), "0.25-0.50": (0.25,0.50), "0.50-0.75": (0.50,0.75), "0.75-0.92": (0.75,0.92)}
    print(f"  {'Bucket':<12} {'Trades':>6} {'PnL':>10} {'Avg PnL':>9}")
    print(f"  {'-'*42}")
    for label, (lo, hi) in buckets.items():
        bt = [t for t in completed_trades if lo <= t["entry_price"] <= hi]
        bpnl = sum(t["pnl"] for t in bt)
        avg = bpnl / len(bt) if bt else 0
        print(f"  {label:<12} {len(bt):>6} {bpnl:>+10.2f} {avg:>+9.2f}")

    # PnL by quarter (use adj_sec from snap at entry)
    print(f"\n  PnL by quarter (adj_seconds at entry):")
    q_buckets = {"Q4 last 2m": (0,120), "Q4 2-5m": (120,300), "Q4 5-10m": (300,600)}
    print(f"  {'Period':<15} {'Trades':>6} {'PnL':>10}")
    print(f"  {'-'*35}")
    for label, (lo, hi) in q_buckets.items():
        qt = []
        for t in completed_trades:
            pid = t["pid"]
            gs = t.get("game_state_entry", "")
            parts = gs.split()
            team3 = parts[0][:3] if parts else ""
            snap = find_snap_at_time(team3, trade_entries_log.get(pid, {}).get("time", ""))
            if snap and lo <= snap["adj_sec"] < hi:
                qt.append(t)
        qpnl = sum(t["pnl"] for t in qt)
        print(f"  {label:<15} {len(qt):>6} {qpnl:>+10.2f}")
else:
    print(f"\n  No completed trades to summarize.")
    if open_positions:
        print(f"  Open positions: {len(open_positions)}")
        for op in open_positions:
            print(f"    {op['pid']}: {op.get('direction','')} {op.get('outcome','')} @ {op.get('entry_price',0):.3f} | edge={op.get('entry_edge',0):+.3f} | ${op.get('size',0):.0f} | {op.get('game_state_entry','')}")

# ══════════════════════════════════════════════════════════════════
# TASK 4: STRUCTURAL CHECKS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TASK 4: STRUCTURAL CHECKS")
print("=" * 100)

issues = []

for t in all_trades_for_audit:
    pid = t.get("pid", "?")
    ep = t.get("entry_price", 0)
    gs = t.get("game_state_entry", "")
    parts = gs.split()
    team3 = parts[0][:3] if parts else ""
    entry_log = trade_entries_log.get(pid, {})
    snap = find_snap_at_time(team3, entry_log.get("time", ""))

    if ep >= 0.95 or ep <= 0.05:
        issues.append(f"  {pid}: EXTREME_PRICE entry at {ep:.3f}")
    if snap and snap["sprd"] > 0.08:
        issues.append(f"  {pid}: WIDE_SPREAD at entry ({snap['sprd']:.3f})")
    if snap and snap["age"] > 20:
        issues.append(f"  {pid}: STALE_BOOK at entry ({snap['age']:.0f}s)")

# Cooldown check
game_last_entry = {}
for t in sorted(all_trades_for_audit, key=lambda x: x.get("entry_ts", 0)):
    gid = t.get("game_id", "")
    ts = t.get("entry_ts", 0)
    if gid in game_last_entry:
        diff = ts - game_last_entry[gid]
        if diff < 300 and diff > 0:
            issues.append(f"  {t.get('pid','?')}: COOLDOWN_VIOLATION {diff:.0f}s since previous in game {gid}")
    game_last_entry[gid] = ts

# Per-game stop check
game_cumulative = defaultdict(float)
for t in sorted(completed_trades, key=lambda x: x.get("exit_ts", 0)):
    gid = t.get("game_id", "")
    game_cumulative[gid] += t.get("pnl", 0)
    if game_cumulative[gid] < -200:
        # Check if any more trades happened after this
        later_trades = [t2 for t2 in completed_trades if t2["game_id"] == gid and t2.get("entry_ts", 0) > t.get("exit_ts", 0)]
        if later_trades:
            issues.append(f"  GAME_STOP_VIOLATION: game {gid} exceeded -$200 ({game_cumulative[gid]:.2f}) but {len(later_trades)} more trades followed")

if issues:
    print(f"\n  ⚠️  {len(issues)} structural issue(s) found:")
    for issue in issues:
        print(f"  {issue}")
else:
    print(f"\n  ✅ No structural issues found. All checks passed.")

# ══════════════════════════════════════════════════════════════════
# TASK 5: FINAL VERDICT
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TASK 5: FINAL VERDICT")
print("=" * 100)

all_passed = all(r["passed"] == "Y" for r in validation_results) if 'validation_results' in dir() and validation_results else True
no_structural = len(issues) == 0

total_trade_count = len(completed_trades) + len(open_positions)

print(f"\n  1. Did execution hygiene function correctly?")
if all_passed and no_structural:
    print(f"     → YES — all gates respected, no structural violations")
elif total_trade_count == 0:
    print(f"     → N/A — no trades to validate (system may be correctly filtering)")
else:
    print(f"     → NO — violations found (see above)")

print(f"\n  2. Was any loss variance or structural?")
if completed_trades:
    total_pnl = sum(t["pnl"] for t in completed_trades)
    if total_pnl >= 0:
        print(f"     → No loss. Total PnL: ${total_pnl:+.2f}")
    else:
        # Check if losses came from valid trades
        print(f"     → Total PnL: ${total_pnl:+.2f}")
        if all_passed:
            print(f"     → Loss is VARIANCE (all gates passed)")
        else:
            print(f"     → Loss may be STRUCTURAL (gate violations found)")
elif open_positions:
    print(f"     → Positions still open, cannot determine yet")
else:
    print(f"     → No trades taken")

print(f"\n  3. Are trades in 0.20-0.65 probability band?")
outside_band = [t for t in all_trades_for_audit if not (0.20 <= t.get("entry_price", 0) <= 0.65)]
if outside_band:
    print(f"     → NO — {len(outside_band)} trade(s) outside [0.20, 0.65]:")
    for t in outside_band:
        print(f"       {t.get('pid')}: entry_price={t.get('entry_price',0):.3f}")
else:
    if all_trades_for_audit:
        print(f"     → YES — all {len(all_trades_for_audit)} trades within [0.20, 0.65]")
    else:
        print(f"     → N/A — no trades taken")

print(f"\n  4. Is the system safe to continue running unchanged?")
if no_structural:
    print(f"     → YES — execution hygiene is functioning as designed")
else:
    print(f"     → REVIEW NEEDED — structural issues detected")
