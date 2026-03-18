#!/usr/bin/env python3
"""
NBA v1 Execution Hygiene — Full Analysis
Tasks 1-4 + PnL projection from Feb 24-25 data.
"""
import re, math
from collections import defaultdict, Counter

LOG = "/root/poly_high_sports/sports_system.log"

# ── Parse SNAP lines ──────────────────────────────────────────────
SNAP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*SNAP (\w+) (\d+)-(\d+) \| "
    r"adj=\s*([\d.]+) σ=([\d.]+) seff=([-\d.]+) z=([-\d.]+) \| "
    r"model=([\d.]+) \| "
    r"bid=([\d.]+) ask=([\d.]+) mid=([\d.]+) \| "
    r"bsz=([\d.]+) asz=([\d.]+) sprd=([\d.]+) age=([-\d.]+)s \| "
    r"edge=([-+\d.]+)"
)

ENTRY_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*PAPER ENTRY: (P\d+) (BUY|SELL) (\w+) @ ([\d.]+) \(\$(\d+)\) edge=([-\d.]+) \| (.+)"
)
EXIT_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*PAPER EXIT: (P\d+) (\w+) @ ([\d.]+) → ([\d.]+) PnL=\$([-\d.]+) \((\w+)\)"
)

team_snaps = defaultdict(list)
entries = {}
trades = []

with open(LOG) as f:
    for line in f:
        m = SNAP_RE.search(line)
        if m:
            s = {
                "time": m.group(1), "team": m.group(2),
                "hs": int(m.group(3)), "as_": int(m.group(4)),
                "adj_sec": float(m.group(5)), "sigma": float(m.group(6)),
                "seff": float(m.group(7)), "z": float(m.group(8)),
                "model": float(m.group(9)),
                "bid": float(m.group(10)), "ask": float(m.group(11)),
                "mid": float(m.group(12)),
                "bsz": float(m.group(13)), "asz": float(m.group(14)),
                "sprd": float(m.group(15)), "age": float(m.group(16)),
                "edge": float(m.group(17)),
            }
            team_snaps[s["team"]].append(s)
            continue

        em = ENTRY_RE.search(line)
        if em:
            pid = em.group(2)
            entries[pid] = {
                "time": em.group(1), "pid": pid,
                "direction": em.group(3), "outcome": em.group(4),
                "entry_price": float(em.group(5)), "size": float(em.group(6)),
                "edge": float(em.group(7)), "game_state": em.group(8),
            }
            continue

        xm = EXIT_RE.search(line)
        if xm:
            pid = xm.group(2)
            entry = entries.get(pid, {})
            pnl = float(xm.group(6))
            if "PnL=$-" in line:
                pnl = -abs(pnl)
            trades.append({
                **entry, "exit_time": xm.group(1),
                "exit_price": float(xm.group(5)),
                "pnl": pnl, "exit_reason": xm.group(7),
            })

all_snaps = []
for team, ss in team_snaps.items():
    for s in ss:
        all_snaps.append(s)

print(f"Parsed {len(all_snaps)} SNAP lines, {len(trades)} trades, {len(team_snaps)} games")

# ══════════════════════════════════════════════════════════════════
# TASK 1: VALIDATE FRESHNESS AS PRIMARY PREDICTOR
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TASK 1: PREDICTOR CORRELATION — What drives tradeability?")
print("=" * 90)

game_metrics = {}
for team, ss in sorted(team_snaps.items()):
    n = len(ss)
    fresh20 = sum(1 for s in ss if s["age"] <= 20) / n * 100
    fresh30 = sum(1 for s in ss if s["age"] <= 30) / n * 100
    avg_age = sum(s["age"] for s in ss) / n
    avg_sprd = sum(s["sprd"] for s in ss) / n
    avg_depth = sum(s["bsz"] + s["asz"] for s in ss) / n
    tradeable = sum(1 for s in ss if s["sprd"] <= 0.08 and s["age"] <= 20 and 0.08 <= s["mid"] <= 0.92) / n * 100

    # Longest consecutive streak with age <= 30
    best_streak = streak = 0
    for s in ss:
        if s["age"] <= 30:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0
    streak_min = best_streak * 5 / 60  # ~5s per tick

    # Price change frequency estimate: count ticks where age < 10s (implies recent update)
    recent_updates = sum(1 for s in ss if s["age"] <= 10)
    updates_per_min = recent_updates / (n * 5 / 60)  # n ticks * 5s / 60 = minutes

    game_metrics[team] = {
        "n": n, "fresh20": fresh20, "fresh30": fresh30,
        "avg_age": avg_age, "avg_sprd": avg_sprd, "avg_depth": avg_depth,
        "tradeable": tradeable, "streak_min": streak_min,
        "updates_per_min": updates_per_min,
    }

# Correlation (Pearson)
def pearson(xs, ys):
    n = len(xs)
    if n < 3: return 0
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs, ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0

teams = sorted(game_metrics.keys())
tradeable_pcts = [game_metrics[t]["tradeable"] for t in teams]

predictors = {
    "avg_spread": [game_metrics[t]["avg_sprd"] for t in teams],
    "avg_depth": [game_metrics[t]["avg_depth"] for t in teams],
    "avg_book_age": [game_metrics[t]["avg_age"] for t in teams],
    "%_fresh_20s": [game_metrics[t]["fresh20"] for t in teams],
    "%_fresh_30s": [game_metrics[t]["fresh30"] for t in teams],
    "updates/min": [game_metrics[t]["updates_per_min"] for t in teams],
    "streak_min": [game_metrics[t]["streak_min"] for t in teams],
}

print(f"\n{'Predictor':<16} {'r (Pearson)':>12} {'|r|':>6}  Interpretation")
print("-" * 65)
ranked = sorted(predictors.items(), key=lambda x: -abs(pearson(x[1], tradeable_pcts)))
for name, vals in ranked:
    r = pearson(vals, tradeable_pcts)
    strength = "DOMINANT" if abs(r) > 0.85 else ("STRONG" if abs(r) > 0.7 else ("MODERATE" if abs(r) > 0.5 else "WEAK"))
    print(f"{name:<16} {r:>+12.4f} {abs(r):>6.3f}  {strength}")

print(f"\n  Per-game detail:")
print(f"  {'Team':<6} {'Ticks':>5} {'Fresh20%':>8} {'Fresh30%':>8} {'AvgAge':>7} {'AvgSprd':>7} {'Depth':>7} {'Upd/min':>7} {'Streak':>7} {'Trade%':>7}")
print(f"  {'-'*80}")
for t in sorted(teams, key=lambda x: -game_metrics[x]["tradeable"]):
    g = game_metrics[t]
    print(f"  {t:<6} {g['n']:>5} {g['fresh20']:>7.1f}% {g['fresh30']:>7.1f}% {g['avg_age']:>7.0f} {g['avg_sprd']:>7.3f} {g['avg_depth']:>7.0f} {g['updates_per_min']:>7.1f} {g['streak_min']:>6.1f}m {g['tradeable']:>6.1f}%")

# ══════════════════════════════════════════════════════════════════
# TASK 2: GAME-LEVEL GATE — Threshold sweep
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TASK 2: GAME-LEVEL FRESHNESS GATE — Threshold Sweep")
print("=" * 90)

# Gate: book_age ≤ 30s for at least 3 consecutive min (36 ticks) AND updates/min ≥ X
CONSEC_TICKS = 36  # 3 min at 5s intervals

def game_passes_gate(team, min_upd_per_min):
    ss = team_snaps[team]
    # Check consecutive fresh streak
    streak = 0
    has_streak = False
    for s in ss:
        if s["age"] <= 30:
            streak += 1
            if streak >= CONSEC_TICKS:
                has_streak = True
                break
        else:
            streak = 0

    g = game_metrics[team]
    return has_streak and g["updates_per_min"] >= min_upd_per_min

print(f"\n  Gate: book_age ≤ 30s for 3+ consecutive min AND updates/min ≥ X")
print(f"\n  {'X':>5} {'Games':>6} {'Allowed':>30} {'Trade_ticks':>12} {'%_retained':>10} {'Garbage_elim':>13}")
print(f"  {'-'*85}")

total_tradeable = sum(sum(1 for s in ss if s["sprd"]<=0.08 and s["age"]<=20 and 0.08<=s["mid"]<=0.92) for ss in team_snaps.values())
total_garbage = len(all_snaps) - total_tradeable

for x in [0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
    allowed = [t for t in teams if game_passes_gate(t, x)]
    blocked = [t for t in teams if not game_passes_gate(t, x)]

    kept_trade = sum(sum(1 for s in team_snaps[t] if s["sprd"]<=0.08 and s["age"]<=20 and 0.08<=s["mid"]<=0.92) for t in allowed)
    kept_garbage = sum(len(team_snaps[t]) for t in allowed) - kept_trade
    elim_garbage = total_garbage - kept_garbage

    retained_pct = 100 * kept_trade / total_tradeable if total_tradeable > 0 else 0
    elim_pct = 100 * elim_garbage / total_garbage if total_garbage > 0 else 0

    print(f"  {x:>5.1f} {len(allowed):>6} {str(allowed):>30} {kept_trade:>12} {retained_pct:>9.0f}% {elim_pct:>12.0f}%")

# ══════════════════════════════════════════════════════════════════
# TASK 3: INTRA-GAME FREEZE — Backtest
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TASK 3: INTRA-GAME FREEZE — book_age > 45s for 60s → freeze")
print("=" * 90)

# Simulate freeze logic per game
FREEZE_THRESHOLD = 45   # seconds
FREEZE_DURATION = 12    # 12 ticks @ 5s = 60s
UNFREEZE_STREAK = 36    # 3 min fresh to resume

for team in sorted(teams, key=lambda x: -game_metrics[x]["tradeable"]):
    ss = team_snaps[team]
    frozen = False
    freeze_count = 0
    stale_streak = 0
    fresh_streak = 0
    active_ticks = 0
    frozen_ticks = 0
    active_tradeable = 0
    frozen_tradeable = 0

    for s in ss:
        is_tradeable = s["sprd"] <= 0.08 and s["age"] <= 20 and 0.08 <= s["mid"] <= 0.92

        if s["age"] > FREEZE_THRESHOLD:
            stale_streak += 1
            fresh_streak = 0
        else:
            stale_streak = 0
            fresh_streak += 1

        if not frozen:
            if stale_streak >= FREEZE_DURATION:
                frozen = True
                freeze_count += 1
            else:
                active_ticks += 1
                if is_tradeable:
                    active_tradeable += 1
        else:
            if fresh_streak >= UNFREEZE_STREAK:
                frozen = False
                active_ticks += 1
                if is_tradeable:
                    active_tradeable += 1
            else:
                frozen_ticks += 1
                if is_tradeable:
                    frozen_tradeable += 1

    total = len(ss)
    g = game_metrics[team]
    total_t = sum(1 for s in ss if s["sprd"]<=0.08 and s["age"]<=20 and 0.08<=s["mid"]<=0.92)
    print(f"  {team:<6} | active={active_ticks:>5} frozen={frozen_ticks:>5} freezes={freeze_count} | "
          f"trade_kept={active_tradeable:>4}/{total_t:<4} trade_lost={frozen_tradeable}")

# ══════════════════════════════════════════════════════════════════
# TASK 4: STRUCTURAL ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TASK 4: STRUCTURAL ANALYSIS — Where do tradeable ticks live?")
print("=" * 90)

# 4a. By adj_seconds bucket (proxy for quarter/time)
print("\n  4a. Tradeable ticks by time remaining (adj_seconds)")
time_buckets = {"Q4 last 2 min (0-120)": (0,120), "Q4 2-5 min (120-300)": (120,300),
                "Q4 5-10 min (300-600)": (300,600), "Q3 (600-1440)": (600,1440),
                "Q2 (1440-2160)": (1440,2160), "Q1 (2160-2880)": (2160,2880)}

print(f"  {'Bucket':<25} {'Total':>6} {'Tradeable':>10} {'%':>6} {'Avg Edge':>9}")
print(f"  {'-'*60}")
for label, (lo, hi) in time_buckets.items():
    bucket = [s for s in all_snaps if lo <= s["adj_sec"] < hi]
    t = [s for s in bucket if s["sprd"]<=0.08 and s["age"]<=20 and 0.08<=s["mid"]<=0.92]
    avg_edge = sum(abs(s["edge"]) for s in t) / len(t) if t else 0
    pct = 100 * len(t) / len(bucket) if bucket else 0
    print(f"  {label:<25} {len(bucket):>6} {len(t):>10} {pct:>5.1f}% {avg_edge:>9.4f}")

# 4b. By score diff bucket
print(f"\n  4b. Tradeable ticks by score differential")
diff_buckets = {"0-5 (close)": (0,5), "6-10": (6,10), "11-15": (11,15),
                "16-20": (16,20), "21+ (blowout)": (21,100)}
print(f"  {'Score Diff':<18} {'Total':>6} {'Tradeable':>10} {'%':>6}")
print(f"  {'-'*45}")
for label, (lo, hi) in diff_buckets.items():
    bucket = [s for s in all_snaps if lo <= abs(s["hs"]-s["as_"]) <= hi]
    t = [s for s in bucket if s["sprd"]<=0.08 and s["age"]<=20 and 0.08<=s["mid"]<=0.92]
    pct = 100 * len(t) / len(bucket) if bucket else 0
    print(f"  {label:<18} {len(bucket):>6} {len(t):>10} {pct:>5.1f}%")

# 4c. Final 2 minutes collapse?
print(f"\n  4c. Do tradeable books collapse in final 2 minutes?")
for team in ["Tra", "Net"]:  # Only check the tradeable games
    ss = team_snaps.get(team, [])
    if not ss: continue
    last2 = [s for s in ss if s["adj_sec"] <= 120]
    pre2 = [s for s in ss if 120 < s["adj_sec"] <= 600]
    last2_t = sum(1 for s in last2 if s["sprd"]<=0.08 and s["age"]<=20 and 0.08<=s["mid"]<=0.92)
    pre2_t = sum(1 for s in pre2 if s["sprd"]<=0.08 and s["age"]<=20 and 0.08<=s["mid"]<=0.92)
    last2_pct = 100*last2_t/len(last2) if last2 else 0
    pre2_pct = 100*pre2_t/len(pre2) if pre2 else 0
    print(f"  {team}: last 2 min: {last2_t}/{len(last2)} ({last2_pct:.0f}%) tradeable | Q4 2-10 min: {pre2_t}/{len(pre2)} ({pre2_pct:.0f}%) tradeable")

# 4d. Freshness stability — do games start fresh and decay?
print(f"\n  4d. Freshness trajectory (first quarter vs later)")
for team in sorted(teams, key=lambda x: -game_metrics[x]["tradeable"]):
    ss = team_snaps[team]
    n = len(ss)
    q1 = ss[:n//4]
    q2 = ss[n//4:n//2]
    q3 = ss[n//2:3*n//4]
    q4 = ss[3*n//4:]
    ages = [sum(s["age"] for s in q)/len(q) if q else 0 for q in [q1,q2,q3,q4]]
    fresh = [100*sum(1 for s in q if s["age"]<=20)/len(q) if q else 0 for q in [q1,q2,q3,q4]]
    trend = "STABLE" if max(fresh)-min(fresh) < 20 else ("DECAYS" if fresh[0] > fresh[-1]+20 else "IMPROVES")
    print(f"  {team:<6} fresh%: Q1={fresh[0]:>4.0f}% Q2={fresh[1]:>4.0f}% Q3={fresh[2]:>4.0f}% Q4={fresh[3]:>4.0f}%  avg_age: {ages[0]:>5.0f} {ages[1]:>5.0f} {ages[2]:>5.0f} {ages[3]:>5.0f}  → {trend}")

# ══════════════════════════════════════════════════════════════════
# TASK 5: PROJECTED PnL WITH ALL FILTERS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PROJECTED PnL: Feb 24-25 re-run with full execution hygiene")
print("=" * 90)

# Filters:
# 1. Game-level gate: 3 min fresh streak (age≤30s) + updates/min ≥ 2
# 2. Price band: entry_price in [0.08, 0.92]
# 3. Spread: ≤ 0.08
# 4. Staleness: book_age ≤ 20s
# 5. Freeze: stale >45s for 60s → freeze until 3 min fresh
# 6. Cooldown: 5 min between trades on same game
# 7. Per-game stop: max $200 loss per game

GATE_STREAK = 36
GATE_UPD_MIN = 2.0
COOLDOWN_TICKS = 60  # 5 min
MAX_GAME_LOSS = 200

# Determine which games pass the gate
gated_games = set()
for team in teams:
    if game_passes_gate(team, GATE_UPD_MIN):
        gated_games.add(team)

print(f"\n  Games passing gate (3min fresh + ≥{GATE_UPD_MIN} upd/min): {sorted(gated_games)}")
print(f"  Games blocked: {sorted(set(teams) - gated_games)}")

# Replay trades with all filters
print(f"\n  Replaying {len(trades)} trades with filters:")
print(f"  {'PID':<6} {'Game':<35} {'Entry':>6} {'PnL':>9} {'Filter Result'}")
print(f"  {'-'*90}")

filtered_pnl = 0
filtered_trades = 0
game_pnl = defaultdict(float)
game_last_trade_time = defaultdict(lambda: "00:00:00")
game_frozen = defaultdict(bool)

for t in trades:
    gs = t.get("game_state", "")
    ep = t.get("entry_price", 0)
    pid = t.get("pid", "?")
    pnl = t["pnl"]

    # Extract team from game state
    parts = gs.split()
    team3 = parts[0][:3] if parts else ""

    # Find snap at trade time
    entry_time = t.get("time", "")
    snap = None
    for s in team_snaps.get(team3, []):
        if s["time"] <= entry_time:
            snap = s
        else:
            break

    reasons = []

    # 1. Game-level gate
    if team3 not in gated_games:
        reasons.append(f"GATE_BLOCKED({team3})")

    # 2. Price band
    if ep > 0.92 or ep < 0.08:
        reasons.append(f"EXTREME_PRICE({ep:.3f})")

    # 3. Spread
    if snap and snap["sprd"] > 0.08:
        reasons.append(f"WIDE_SPREAD({snap['sprd']:.3f})")

    # 4. Staleness
    if snap and snap["age"] > 20:
        reasons.append(f"STALE({snap['age']:.0f}s)")

    # 6. Cooldown
    last_t = game_last_trade_time[team3]
    # Simple time comparison (HH:MM:SS)
    if entry_time and last_t:
        def to_sec(t):
            h,m,s = t.split(":")
            return int(h)*3600+int(m)*60+int(s)
        if entry_time > "00:00:00" and last_t > "00:00:00":
            diff = to_sec(entry_time) - to_sec(last_t)
            if diff < 300 and diff > 0:
                reasons.append(f"COOLDOWN({diff}s)")

    # 7. Per-game stop
    if game_pnl[team3] <= -MAX_GAME_LOSS:
        reasons.append(f"GAME_STOP(${game_pnl[team3]:.0f})")

    if reasons:
        status = "BLOCKED: " + " | ".join(reasons)
    else:
        status = f"ALLOWED → PnL ${pnl:+.2f}"
        filtered_pnl += pnl
        filtered_trades += 1
        game_pnl[team3] += pnl
        game_last_trade_time[team3] = entry_time

    print(f"  {pid:<6} {gs[:35]:<35} {ep:>6.3f} {pnl:>+9.2f} {status}")

print(f"\n  {'='*60}")
print(f"  RAW PnL (all trades):     ${sum(t['pnl'] for t in trades):>+10.2f} ({len(trades)} trades)")
print(f"  FILTERED PnL:             ${filtered_pnl:>+10.2f} ({filtered_trades} trades)")
print(f"  Improvement:              ${filtered_pnl - sum(t['pnl'] for t in trades):>+10.2f}")

# Expected trade frequency
total_game_minutes = sum(g["n"] * 5 / 60 for g in game_metrics.values())
gated_minutes = sum(game_metrics[t]["n"] * 5 / 60 for t in gated_games)
print(f"\n  Expected trade frequency:")
print(f"    Total game-minutes tonight: {total_game_minutes:.0f}")
print(f"    Gated game-minutes:         {gated_minutes:.0f}")
print(f"    Trades per gated game-minute: {filtered_trades/gated_minutes:.4f}" if gated_minutes > 0 else "    N/A")
print(f"    ≈ trades per 11-game slate:   {filtered_trades:.0f}")
