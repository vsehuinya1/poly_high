#!/usr/bin/env python3
"""
Tick-by-tick replay of yesterday's snapshot data.
Applies CURRENT v1.1.1 model + trade filters to every tick.
Reports all signals and which were blocked by execution filters.
"""
import csv
import math
import sys
from datetime import datetime, timezone

SNAP_FILE = "/root/poly_high_sports/sports_data/snapshots_20260222.csv"

# ── Current model parameters (v1.1.1) ───────────────────────────
PREGAME_PROB = 0.5  # no anchor was set (system started mid-game)

# ── Current filter thresholds ────────────────────────────────────
SIGNAL_THRESHOLD = 0.05    # fires a signal log
TRADE_THRESHOLD = 0.07     # required for actual trade entry
MAX_ADJ_SECONDS = 600      # last 10 minutes only
MAX_SCORE_DIFF = 15        # close games only

# ── Model functions (exact copy from v1.1.1 models.py) ──────────
def phi(x):
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    t2, t3, t4, t5 = t*t, t**3, t**4, t**5
    cdf = 1.0 - (1.0/math.sqrt(2*math.pi)) * math.exp(-x*x/2) * (
        0.319381530*t - 0.356563782*t2 + 1.781477937*t3
        - 1.821255978*t4 + 1.330274429*t5)
    return 0.5 + sign * (cdf - 0.5)

def inv_phi(p):
    p = max(0.0001, min(0.9999, p))
    if abs(p - 0.5) < 0.001:
        return 0.0
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
        num = 2.515517 + 0.802853*t + 0.010328*t*t
        den = 1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t
        return -(t - (num / den))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        num = 2.515517 + 0.802853*t + 0.010328*t*t
        den = 1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t
        return t - (num / den)

def nba_model(home_score, away_score, adj_seconds, period="", pregame_prob=0.5):
    """Returns (p_home, sigma, strength_adj, s_eff, z)"""
    score_diff = home_score - away_score

    if adj_seconds <= 0:
        if score_diff > 0:
            return (1.0, 0, 0, score_diff, 99)
        elif score_diff < 0:
            return (0.0, 0, 0, score_diff, -99)
        else:
            return (0.5, 0, 0, 0, 0)

    is_ot = "OT" in str(period).upper()
    if is_ot:
        adj_seconds = min(adj_seconds, 300.0)

    t_min = adj_seconds / 60.0

    if adj_seconds <= 60:
        sigma = 2.10
    elif adj_seconds <= 180:
        sigma = 1.95
    else:
        sigma = 1.70

    s0 = 1.70 * math.sqrt(48.0) * inv_phi(pregame_prob)
    strength_adj = s0 * (t_min / 48.0)
    s_eff = score_diff + strength_adj
    z = s_eff / (sigma * math.sqrt(t_min))
    p_home = max(0.001, min(0.999, phi(z)))

    return (p_home, sigma, strength_adj, s_eff, z)


# ── Load snapshot data ───────────────────────────────────────────
rows = []
with open(SNAP_FILE) as f:
    for r in csv.DictReader(f):
        rows.append(r)

print("=" * 80)
print("TICK-BY-TICK REPLAY — Feb 22 2026")
print("Snapshot ticks: %d" % len(rows))
print("Model: v1.1.1 (anchor + dynamic sigma)")
print("Pregame anchor: %.2f (neutral — system discovered mid-game)" % PREGAME_PROB)
print("=" * 80)

# ── Replay ───────────────────────────────────────────────────────
signals_before_filters = []  # (tick, ts, edge, direction, adj_sec, score_diff, sigma)
signals_after_filters = []
blocked_signals = []

prev_score = None

for i, r in enumerate(rows):
    try:
        ts = float(r["timestamp"])
        hs = int(r.get("home_score", 0) or 0)
        aws = int(r.get("away_score", 0) or 0)
        elapsed = float(r.get("elapsed_min", 0) or 0)
        period = r.get("period", "")
        mid = float(r.get("mid", 0) or 0)
        # Market probabilities from the snapshot
        mkt_home = mid  # mid IS the home probability for home token
        mkt_away = 1.0 - mid if mid > 0 else 0

        # Compute adjusted seconds (48 min total for NBA)
        adj_sec = max(0, (48.0 - elapsed) * 60.0)

        # Run v1.1.1 model
        p_home, sigma, str_adj, s_eff, z = nba_model(hs, aws, adj_sec, period, PREGAME_PROB)
        p_away = 1.0 - p_home

        # Track score changes
        curr_score = (hs, aws)
        score_changed = (prev_score is not None and curr_score != prev_score)
        prev_score = curr_score

        if mid <= 0:
            continue  # no market data

        # Compute edges
        edge_home = p_home - mkt_home
        edge_away = p_away - mkt_away
        abs_edge = max(abs(edge_home), abs(edge_away))
        score_diff = abs(hs - aws)

        # Best edge direction
        if abs(edge_home) >= abs(edge_away):
            edge = edge_home
            direction = "BUY_HOME" if edge > 0 else "SELL_HOME"
        else:
            edge = edge_away
            direction = "BUY_AWAY" if edge_away > 0 else "SELL_AWAY"

        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")

        # ── Signal check (pre-filter) ────────────────────────
        if abs_edge >= SIGNAL_THRESHOLD:
            sig = {
                "tick": i,
                "time": dt,
                "ts": ts,
                "score": "%d-%d" % (hs, aws),
                "elapsed": "%.1f" % elapsed,
                "adj_sec": adj_sec,
                "score_diff": score_diff,
                "p_home_model": p_home,
                "p_home_mkt": mkt_home,
                "edge": edge,
                "abs_edge": abs_edge,
                "direction": direction,
                "sigma": sigma,
                "s_eff": s_eff,
                "z": z,
                "score_changed": score_changed,
            }
            signals_before_filters.append(sig)

            # ── Execution filter check ───────────────────────
            blocked_reasons = []
            if adj_sec > MAX_ADJ_SECONDS:
                blocked_reasons.append("TIME: adj_sec=%.0f > 600 (not in last 10 min)" % adj_sec)
            if score_diff > MAX_SCORE_DIFF:
                blocked_reasons.append("SPREAD: diff=%d > 15 (blowout)" % score_diff)
            if abs_edge < TRADE_THRESHOLD:
                blocked_reasons.append("EDGE: abs=%.4f < 0.07 (below trade threshold)" % abs_edge)

            if blocked_reasons:
                sig["blocked"] = blocked_reasons
                blocked_signals.append(sig)
            else:
                signals_after_filters.append(sig)

    except Exception as e:
        pass

# ── Report ───────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 1: ALL SIGNALS (abs_edge >= 0.05) — BEFORE EXECUTION FILTERS")
print("=" * 80)
print("Total: %d signals out of %d ticks" % (len(signals_before_filters), len(rows)))
print("")
print("%-8s %-7s %-6s %-5s %-7s %-7s %-8s %-8s %-12s %-5s" % (
    "Time", "Score", "Elap", "Adj_s", "Model", "Market", "Edge", "AbsEdge", "Direction", "Sigma"))
print("-" * 85)
for s in signals_before_filters:
    print("%-8s %-7s %-6s %-5.0f %-7.3f %-7.3f %-+8.4f %-8.4f %-12s %-5.2f" % (
        s["time"], s["score"], s["elapsed"], s["adj_sec"],
        s["p_home_model"], s["p_home_mkt"], s["edge"], s["abs_edge"],
        s["direction"], s["sigma"]))

print("\n" + "=" * 80)
print("SECTION 2: SIGNALS THAT PASSED ALL FILTERS (would have traded)")
print("Filters: adj_sec<=600 AND score_diff<=15 AND abs_edge>=0.07")
print("=" * 80)
print("Total: %d / %d passed" % (len(signals_after_filters), len(signals_before_filters)))
print("")
if signals_after_filters:
    print("%-8s %-7s %-6s %-5s %-7s %-7s %-+8s %-8s %-12s %-5s" % (
        "Time", "Score", "Elap", "Adj_s", "Model", "Market", "Edge", "AbsEdge", "Direction", "Sigma"))
    print("-" * 85)
    for s in signals_after_filters:
        print("%-8s %-7s %-6s %-5.0f %-7.3f %-7.3f %-+8.4f %-8.4f %-12s %-5.2f" % (
            s["time"], s["score"], s["elapsed"], s["adj_sec"],
            s["p_home_model"], s["p_home_mkt"], s["edge"], s["abs_edge"],
            s["direction"], s["sigma"]))
else:
    print("  NONE — all signals were blocked by filters")

print("\n" + "=" * 80)
print("SECTION 3: BLOCKED SIGNALS — WHY EACH WAS REJECTED")
print("=" * 80)
print("Total blocked: %d" % len(blocked_signals))
print("")

# Count blocks by reason
reason_counts = {}
for s in blocked_signals:
    for r in s["blocked"]:
        tag = r.split(":")[0]
        reason_counts[tag] = reason_counts.get(tag, 0) + 1

print("Block reasons (signals can have multiple):")
for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print("  %s: %d" % (reason, count))

print("\nFirst 20 blocked signals:")
print("%-8s %-7s %-5s %-+8s %-12s  %s" % ("Time", "Score", "Adj_s", "Edge", "Direction", "Reason(s)"))
print("-" * 90)
for s in blocked_signals[:20]:
    reasons = " | ".join(s["blocked"])
    print("%-8s %-7s %-5.0f %-+8.4f %-12s  %s" % (
        s["time"], s["score"], s["adj_sec"], s["edge"], s["direction"], reasons))

# ── Sizing simulation for passed signals ─────────────────────────
print("\n" + "=" * 80)
print("SECTION 4: SIMULATED TRADE LOG (if trades had executed)")
print("Size formula: min(300, max(50, abs(edge) * 1000))")
print("=" * 80)
if signals_after_filters:
    total_exposure = 0
    for s in signals_after_filters:
        size = min(300, max(50, abs(s["edge"]) * 1000))
        total_exposure += size
        print("  %s | %s | edge=%+.4f | size=$%.0f | %s" % (
            s["time"], s["score"], s["edge"], size, s["direction"]))
    print("\n  Total exposure across all signals: $%.0f" % total_exposure)
    print("  Unique signal count: %d" % len(signals_after_filters))
else:
    print("  No trades to simulate.")

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
if len(signals_after_filters) > 0:
    print("  %d signals would have generated trades." % len(signals_after_filters))
    print("  The WS pipeline failure was the ONLY reason for zero trades.")
elif len(signals_before_filters) > 0 and len(signals_after_filters) == 0:
    print("  %d raw signals existed but ALL were blocked by filters." % len(signals_before_filters))
    print("  Filters are too strict for this game.")
    time_blocked = sum(1 for s in blocked_signals if any("TIME" in r for r in s["blocked"]))
    edge_blocked = sum(1 for s in blocked_signals if any("EDGE" in r for r in s["blocked"]))
    spread_blocked = sum(1 for s in blocked_signals if any("SPREAD" in r for r in s["blocked"]))
    print("  TIME filter blocked: %d" % time_blocked)
    print("  EDGE filter blocked: %d" % edge_blocked)
    print("  SPREAD filter blocked: %d" % spread_blocked)
else:
    print("  No signals at all — model never found edge > 0.05.")
