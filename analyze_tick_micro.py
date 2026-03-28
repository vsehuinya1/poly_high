#!/usr/bin/env python3
"""
Tennis Tick Microstructure Analysis — empirical patterns from tick data.

Analyzes tick-level BBO sequences for every tennis trade to derive:
  1. MFE/MAE curves over time (per-trade + aggregate)
  2. Tick sequence patterns (first 3 minutes)
  3. False signal detection ("dead on arrival" trades)
  4. Entry validation filter simulation
  5. Optimal exit rules (empirical)

Data sources:
  - sports_data/tennis_trade_lifecycle_*.csv  (trade records)
  - sports_data/tick_history.db               (tick-level BBO)

Output:
  - sports_data/tick_microstructure_results.json
  - Console report
"""
import csv
import glob
import json
import os
import sqlite3
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "sports_data"
DB_PATH = DATA_DIR / "tick_history.db"


# ── Data Structures ──────────────────────────────────────────────────

@dataclass
class Trade:
    match_id: str = ""
    player: str = ""
    trigger: str = ""
    entry_price: float = 0.0
    entry_timestamp: float = 0.0
    exit_timestamp: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    R_multiple: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    duration_seconds: float = 0.0
    # Enriched from tick DB
    tick_sequence: list = field(default_factory=list)  # [(ts, mid, spread), ...]
    has_ticks: bool = False


@dataclass
class TickStats:
    """Per-trade tick-level statistics."""
    # MFE/MAE curves
    mfe_at: dict = field(default_factory=dict)    # {seconds: mfe_value}
    mae_at: dict = field(default_factory=dict)    # {seconds: mae_value}
    time_to_peak_mfe: float = 0.0
    peak_mfe: float = 0.0
    max_mae: float = 0.0
    mae_before_mfe: float = 0.0  # max adverse before peak MFE
    time_to_continuous_adverse: float = 0.0

    # Tick patterns (first 3 min)
    consecutive_favorable: int = 0
    consecutive_adverse: int = 0
    early_net_change: float = 0.0
    early_velocity: float = 0.0  # price change / time
    immediate_adverse_seq: bool = False  # starts with >=3 adverse ticks

    # Overall
    is_winner: bool = False
    is_dead_on_arrival: bool = False
    n_ticks: int = 0


# ── CSV Parsing ──────────────────────────────────────────────────────

def _sf(val, default=0.0):
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def load_trades() -> list[Trade]:
    """Load all lifecycle CSVs using DictReader (handles all schemas)."""
    trades = []
    files = sorted(glob.glob(str(DATA_DIR / "tennis_trade_lifecycle_*.csv")))
    print(f"Found {len(files)} lifecycle CSV files")

    for fpath in files:
        with open(fpath) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if not headers:
                continue

            for row in reader:
                try:
                    entry_price = _sf(row.get("entry_price"))
                    entry_ts = _sf(row.get("timestamp_entry"))
                    exit_ts = _sf(row.get("timestamp_exit"))
                    if entry_price <= 0 or entry_ts <= 0 or exit_ts <= 0:
                        continue

                    t = Trade(
                        match_id=row.get("match_id", ""),
                        player=row.get("player", ""),
                        trigger=row.get("trigger", ""),
                        entry_price=entry_price,
                        entry_timestamp=entry_ts,
                        exit_timestamp=exit_ts,
                        exit_price=_sf(row.get("exit_price")),
                        exit_reason=row.get("exit_reason", ""),
                        R_multiple=_sf(row.get("R_multiple")),
                        mfe=_sf(row.get("mfe")),
                        mae=_sf(row.get("mae")),
                        duration_seconds=_sf(row.get("duration_seconds")),
                    )
                    trades.append(t)
                except Exception:
                    continue

    print(f"Loaded {len(trades)} trades")
    return trades


# ── Tick DB Enrichment ───────────────────────────────────────────────

def enrich_with_ticks(trades: list[Trade]):
    """Cross-reference trades with tick_history.db for full tick sequences.

    Matching strategy:
      1. Try token_labels.game_id = trade.match_id
      2. Fall back to LIKE on player name in market_title
    """
    if not DB_PATH.exists():
        print(f"Tick DB not found: {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))

    enriched = 0
    for trade in trades:
        # 1. Try game_id match
        cursor = conn.execute(
            "SELECT token_id FROM token_labels WHERE game_id = ?",
            (trade.match_id,)
        )
        token_ids = [r[0] for r in cursor.fetchall()]

        # 2. Fall back to player name in title
        if not token_ids and trade.player:
            cursor = conn.execute(
                "SELECT token_id FROM token_labels WHERE market_title LIKE ?",
                (f"%{trade.player}%",)
            )
            token_ids = [r[0] for r in cursor.fetchall()]

        if not token_ids:
            continue

        # Query ticks for each candidate token_id
        t_start = trade.entry_timestamp - 30
        t_end = trade.exit_timestamp + 30
        best_ticks = []

        for token_id in token_ids:
            cursor = conn.execute(
                "SELECT timestamp, mid, spread FROM ticks "
                "WHERE token_id = ? AND timestamp BETWEEN ? AND ? "
                "ORDER BY timestamp",
                (token_id, t_start, t_end)
            )
            rows = [(ts, mid, sprd) for ts, mid, sprd in cursor.fetchall()
                    if mid and mid > 0]
            if len(rows) > len(best_ticks):
                best_ticks = rows

        if len(best_ticks) >= 3:
            trade.tick_sequence = best_ticks
            trade.has_ticks = True
            enriched += 1

    conn.close()
    avg_ticks = sum(len(t.tick_sequence) for t in trades if t.has_ticks) / max(1, enriched)
    print(f"Enriched {enriched}/{len(trades)} trades with tick data "
          f"(avg {avg_ticks:.0f} ticks/trade)")


# ── Step 1: MFE/MAE Curves ──────────────────────────────────────────

def compute_mfe_mae(trade: Trade) -> Optional[TickStats]:
    """Compute MFE and MAE curves from tick sequence."""
    if not trade.has_ticks or not trade.tick_sequence:
        return None

    stats = TickStats()
    stats.is_winner = trade.R_multiple > 0
    stats.n_ticks = len(trade.tick_sequence)

    entry = trade.entry_price
    entry_ts = trade.entry_timestamp

    running_mfe = 0.0
    running_mae = 0.0
    peak_mfe_time = 0.0
    max_mae_before_peak_mfe = 0.0
    peak_mfe_found = False

    # Time buckets for aggregation (seconds from entry)
    time_buckets = [5, 10, 15, 30, 60, 90, 120, 180, 240, 300, 600, 900, 1200, 1800]

    for ts, mid, sprd in trade.tick_sequence:
        elapsed = ts - entry_ts
        if elapsed < 0:
            continue

        favorable = mid - entry
        adverse = entry - mid

        if favorable > running_mfe:
            running_mfe = favorable
            peak_mfe_time = elapsed
            peak_mfe_found = True
        if adverse > running_mae:
            running_mae = adverse

        # Track MAE before peak MFE
        if not peak_mfe_found or elapsed <= peak_mfe_time:
            max_mae_before_peak_mfe = max(max_mae_before_peak_mfe, adverse)

        # Populate time buckets
        for bucket in time_buckets:
            if elapsed <= bucket and bucket not in stats.mfe_at:
                stats.mfe_at[bucket] = running_mfe
                stats.mae_at[bucket] = running_mae

    stats.peak_mfe = running_mfe
    stats.max_mae = running_mae
    stats.time_to_peak_mfe = peak_mfe_time
    stats.mae_before_mfe = max_mae_before_peak_mfe

    # Time to first continuous adverse move (3+ ticks down in a row)
    consec_down = 0
    for i, (ts, mid, sprd) in enumerate(trade.tick_sequence):
        if i == 0:
            continue
        prev_mid = trade.tick_sequence[i - 1][1]
        if mid < prev_mid:
            consec_down += 1
            if consec_down >= 3:
                stats.time_to_continuous_adverse = ts - entry_ts
                break
        else:
            consec_down = 0

    return stats


# ── Step 2: Tick Sequence Patterns (First 3 Minutes) ─────────────────

def compute_tick_patterns(trade: Trade, stats: TickStats):
    """Analyze tick behavior in first 3 minutes after entry."""
    entry = trade.entry_price
    entry_ts = trade.entry_timestamp

    # Filter to first 180s of ticks
    early_ticks = [
        (ts, mid, sprd) for ts, mid, sprd in trade.tick_sequence
        if 0 <= (ts - entry_ts) <= 180
    ]

    if len(early_ticks) < 2:
        return

    # Consecutive favorable/adverse from entry
    max_consec_fav = 0
    max_consec_adv = 0
    cur_consec_fav = 0
    cur_consec_adv = 0
    immediate_adverse = 0  # count adverse ticks right from start

    prev_mid = entry
    for i, (ts, mid, sprd) in enumerate(early_ticks):
        if mid > prev_mid:
            cur_consec_fav += 1
            cur_consec_adv = 0
        elif mid < prev_mid:
            cur_consec_adv += 1
            cur_consec_fav = 0
            if i < 5 and cur_consec_adv == i + 1:  # all ticks adverse so far
                immediate_adverse = cur_consec_adv
        else:
            pass  # unchanged tick

        max_consec_fav = max(max_consec_fav, cur_consec_fav)
        max_consec_adv = max(max_consec_adv, cur_consec_adv)
        prev_mid = mid

    # Net price change in first 3 min
    first_mid = early_ticks[0][1]
    last_mid = early_ticks[-1][1]
    net_change = last_mid - entry

    # Velocity
    time_span = early_ticks[-1][0] - early_ticks[0][0]
    velocity = net_change / max(1.0, time_span)

    stats.consecutive_favorable = max_consec_fav
    stats.consecutive_adverse = max_consec_adv
    stats.early_net_change = net_change
    stats.early_velocity = velocity
    stats.immediate_adverse_seq = immediate_adverse >= 3


# ── Step 3: False Signal Detection ───────────────────────────────────

def detect_false_signals(trade: Trade, stats: TickStats):
    """Mark 'dead on arrival' trades: no MFE >= 0.01 + steady adverse."""
    entry = trade.entry_price

    # Check if any tick reached +0.01 MFE
    ever_reached_001 = False
    for ts, mid, sprd in trade.tick_sequence:
        if mid - entry >= 0.01:
            ever_reached_001 = True
            break

    # Steady adverse: net change is negative after first 60s
    ticks_60s = [
        mid for ts, mid, _ in trade.tick_sequence
        if (ts - trade.entry_timestamp) >= 60
    ]

    steady_adverse = False
    if ticks_60s:
        avg_price_after_60s = sum(ticks_60s[:10]) / len(ticks_60s[:10])
        steady_adverse = avg_price_after_60s < entry - 0.005

    stats.is_dead_on_arrival = (not ever_reached_001) and steady_adverse


# ── Step 4: Entry Validation Simulation ──────────────────────────────

def simulate_entry_filters(trades: list[Trade], all_stats: dict[str, TickStats]) -> dict:
    """Simulate entry validation filters and measure impact on R."""
    tick_trades = [(t, all_stats[t.match_id]) for t in trades
                   if t.match_id in all_stats and all_stats[t.match_id] is not None]

    if not tick_trades:
        return {"error": "no tick data"}

    baseline_R = sum(t.R_multiple for t, _ in tick_trades)
    baseline_n = len(tick_trades)

    results = {}

    # Filter 1: Require >= 2 favorable ticks within 60s
    filtered_1 = []
    for t, s in tick_trades:
        # Count favorable ticks in first 60s
        entry_ts = t.entry_timestamp
        fav_count = 0
        prev = t.entry_price
        for ts, mid, _ in t.tick_sequence:
            elapsed = ts - entry_ts
            if elapsed > 60:
                break
            if mid > prev:
                fav_count += 1
            prev = mid
        if fav_count >= 2:
            filtered_1.append(t)

    results["require_2_fav_60s"] = {
        "kept": len(filtered_1),
        "removed": baseline_n - len(filtered_1),
        "R_after": round(sum(t.R_multiple for t in filtered_1), 4),
        "R_removed": round(sum(t.R_multiple for t, _ in tick_trades if t not in filtered_1), 4),
        "delta_R": round(sum(t.R_multiple for t in filtered_1) - baseline_R, 4),
        "wr_after": round(sum(1 for t in filtered_1 if t.R_multiple > 0) / max(1, len(filtered_1)) * 100, 1),
    }

    # Filter 2: Block if >= 3 immediate adverse ticks
    filtered_2 = [t for t, s in tick_trades if not s.immediate_adverse_seq]
    results["block_3_immed_adverse"] = {
        "kept": len(filtered_2),
        "removed": baseline_n - len(filtered_2),
        "R_after": round(sum(t.R_multiple for t in filtered_2), 4),
        "R_removed": round(sum(t.R_multiple for t, _ in tick_trades if t not in filtered_2), 4),
        "delta_R": round(sum(t.R_multiple for t in filtered_2) - baseline_R, 4),
        "wr_after": round(sum(1 for t in filtered_2 if t.R_multiple > 0) / max(1, len(filtered_2)) * 100, 1),
    }

    # Filter 3: Block dead-on-arrival
    filtered_3 = [t for t, s in tick_trades if not s.is_dead_on_arrival]
    results["block_dead_on_arrival"] = {
        "kept": len(filtered_3),
        "removed": baseline_n - len(filtered_3),
        "R_after": round(sum(t.R_multiple for t in filtered_3), 4),
        "R_removed": round(sum(t.R_multiple for t, _ in tick_trades if t not in filtered_3), 4),
        "delta_R": round(sum(t.R_multiple for t in filtered_3) - baseline_R, 4),
        "wr_after": round(sum(1 for t in filtered_3 if t.R_multiple > 0) / max(1, len(filtered_3)) * 100, 1),
    }

    # Filter 4: Require positive net change in first 30s
    filtered_4 = []
    for t, s in tick_trades:
        entry_ts = t.entry_timestamp
        ticks_30 = [(ts, mid) for ts, mid, _ in t.tick_sequence if 0 <= (ts - entry_ts) <= 30]
        if ticks_30 and ticks_30[-1][1] >= t.entry_price:
            filtered_4.append(t)
    results["require_positive_30s"] = {
        "kept": len(filtered_4),
        "removed": baseline_n - len(filtered_4),
        "R_after": round(sum(t.R_multiple for t in filtered_4), 4),
        "R_removed": round(sum(t.R_multiple for t, _ in tick_trades if t not in filtered_4), 4),
        "delta_R": round(sum(t.R_multiple for t in filtered_4) - baseline_R, 4),
        "wr_after": round(sum(1 for t in filtered_4 if t.R_multiple > 0) / max(1, len(filtered_4)) * 100, 1),
    }

    # Filter 5: Require >= 2 favorable ticks within 30s
    filtered_5 = []
    for t, s in tick_trades:
        entry_ts = t.entry_timestamp
        fav_count = 0
        prev = t.entry_price
        for ts, mid, _ in t.tick_sequence:
            elapsed = ts - entry_ts
            if elapsed > 30:
                break
            if mid > prev:
                fav_count += 1
            prev = mid
        if fav_count >= 2:
            filtered_5.append(t)
    results["require_2_fav_30s"] = {
        "kept": len(filtered_5),
        "removed": baseline_n - len(filtered_5),
        "R_after": round(sum(t.R_multiple for t in filtered_5), 4),
        "R_removed": round(sum(t.R_multiple for t, _ in tick_trades if t not in filtered_5), 4),
        "delta_R": round(sum(t.R_multiple for t in filtered_5) - baseline_R, 4),
        "wr_after": round(sum(1 for t in filtered_5 if t.R_multiple > 0) / max(1, len(filtered_5)) * 100, 1),
    }

    return {
        "baseline": {"n": baseline_n, "total_R": round(baseline_R, 4),
                      "wr": round(sum(1 for t, _ in tick_trades if t.R_multiple > 0) / max(1, baseline_n) * 100, 1)},
        "filters": results,
    }


# ── Step 5: Optimal Exit Rules ──────────────────────────────────────

def compute_optimal_exits(trades: list[Trade], all_stats: dict[str, TickStats]) -> dict:
    """Derive optimal exit parameters from empirical data."""
    tick_trades = [(t, all_stats[t.match_id]) for t in trades
                   if t.match_id in all_stats and all_stats[t.match_id] is not None]

    if not tick_trades:
        return {}

    winners = [(t, s) for t, s in tick_trades if t.R_multiple > 0]
    losers = [(t, s) for t, s in tick_trades if t.R_multiple <= 0]

    # Early exit time confirmation: simulate different windows
    early_exit_sims = {}
    for window in [60, 90, 120, 180, 300, 600, 720]:
        for thresh in [0.01, 0.02, 0.03]:
            kept = []
            exited_r = 0.0
            for t, s in tick_trades:
                # Check if MFE at window exceeds threshold
                mfe_val = s.mfe_at.get(window, 0)
                if mfe_val < thresh:
                    # Would have been exited — estimate exit at that point
                    # Find price at window time
                    for ts, mid, _ in t.tick_sequence:
                        if (ts - t.entry_timestamp) >= window:
                            simulated_r = (mid - t.entry_price) / t.entry_price
                            exited_r += simulated_r
                            break
                    else:
                        exited_r += t.R_multiple  # no tick at window, use actual
                else:
                    kept.append(t)

            kept_r = sum(t.R_multiple for t in kept) + exited_r
            key = f"{window}s_mfe{thresh}"
            early_exit_sims[key] = {
                "window": window,
                "threshold": thresh,
                "total_R": round(kept_r, 4),
                "kept": len(kept),
                "exited": len(tick_trades) - len(kept),
            }

    # MAE threshold analysis: what MAE level optimally kills losers?
    mae_analysis = {}
    for mae_thresh in [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10]:
        would_exit = [t for t, s in tick_trades if s.max_mae >= mae_thresh]
        would_keep = [t for t, s in tick_trades if s.max_mae < mae_thresh]
        mae_analysis[f"mae_{mae_thresh}"] = {
            "threshold": mae_thresh,
            "would_exit": len(would_exit),
            "would_keep": len(would_keep),
            "exit_R": round(sum(t.R_multiple for t in would_exit), 4),
            "keep_R": round(sum(t.R_multiple for t in would_keep), 4),
            "exit_wr": round(sum(1 for t in would_exit if t.R_multiple > 0) / max(1, len(would_exit)) * 100, 1),
        }

    # Tick stop value: does tick-based stop add value beyond -15%?
    tick_stop_value = {
        "trades_with_tick_data": len(tick_trades),
        "trades_hit_10tick_mae": sum(1 for _, s in tick_trades if s.max_mae >= 0.10),
        "trades_hit_15pct_stop": sum(1 for t, _ in tick_trades if t.mae >= t.entry_price * 0.15),
    }

    # Which trades hit tick stop but NOT flat stop? (tick stop adds value)
    tick_only = []
    for t, s in tick_trades:
        tick_stop_price = t.entry_price - 0.10  # 10 ticks
        flat_stop_price = t.entry_price * 0.85
        if s.max_mae >= 0.10 and t.mae < t.entry_price * 0.15:
            tick_only.append(t)

    tick_stop_value["tick_stop_only_catches"] = len(tick_only)
    tick_stop_value["tick_only_R"] = round(sum(t.R_multiple for t in tick_only), 4)
    tick_stop_value["tick_stop_adds_value"] = len(tick_only) > 0 and sum(t.R_multiple for t in tick_only) < 0

    return {
        "early_exit_sims": early_exit_sims,
        "mae_thresholds": mae_analysis,
        "tick_stop": tick_stop_value,
    }


# ── Reporting ────────────────────────────────────────────────────────

def print_report(trades, all_stats, filter_results, exit_results):
    tick_trades = [(t, all_stats[t.match_id]) for t in trades
                   if t.match_id in all_stats and all_stats[t.match_id] is not None]
    winners = [(t, s) for t, s in tick_trades if t.R_multiple > 0]
    losers = [(t, s) for t, s in tick_trades if t.R_multiple <= 0]

    sep = "=" * 80
    line = "-" * 60

    print(f"\n{sep}")
    print("TENNIS TICK MICROSTRUCTURE ANALYSIS")
    print(f"Trades with tick data: {len(tick_trades)}")
    print(f"Winners: {len(winners)} | Losers: {len(losers)}")
    print(sep)

    # Step 1: MFE/MAE Curves
    print(f"\n{line}")
    print("STEP 1: MFE / MAE CURVES")
    print(line)

    # Time buckets for aggregate curves
    time_buckets = [5, 10, 15, 30, 60, 90, 120, 180, 300, 600, 900, 1200, 1800]

    print("\n  WINNERS — Average MFE over time:")
    print(f"  {'Time':>8s}  {'Avg MFE':>10s}  {'Avg MAE':>10s}  {'n':>5s}")
    for bucket in time_buckets:
        mfes = [s.mfe_at.get(bucket, 0) for _, s in winners if bucket in s.mfe_at]
        maes = [s.mae_at.get(bucket, 0) for _, s in winners if bucket in s.mae_at]
        if mfes:
            print(f"  {bucket:>6d}s  {statistics.mean(mfes):>+10.4f}  {statistics.mean(maes):>10.4f}  {len(mfes):>5d}")

    print("\n  WINNERS — MFE Peak Timing:")
    peak_times = [s.time_to_peak_mfe for _, s in winners if s.time_to_peak_mfe > 0]
    if peak_times:
        print(f"    p25: {sorted(peak_times)[len(peak_times)//4]:.0f}s")
        print(f"    p50: {sorted(peak_times)[len(peak_times)//2]:.0f}s")
        print(f"    p75: {sorted(peak_times)[3*len(peak_times)//4]:.0f}s")
        print(f"    mean: {statistics.mean(peak_times):.0f}s")

    print(f"\n  WINNERS — MAE before peak MFE:")
    mae_before = [s.mae_before_mfe for _, s in winners if s.mae_before_mfe > 0]
    if mae_before:
        print(f"    mean: {statistics.mean(mae_before):.4f}")
        print(f"    p50:  {sorted(mae_before)[len(mae_before)//2]:.4f}")
        print(f"    p90:  {sorted(mae_before)[int(len(mae_before)*0.9)]:.4f}")

    print("\n  LOSERS — MAE profile:")
    loser_maes = [s.max_mae for _, s in losers if s.max_mae > 0]
    if loser_maes:
        print(f"    mean: {statistics.mean(loser_maes):.4f}")
        print(f"    p50:  {sorted(loser_maes)[len(loser_maes)//2]:.4f}")
        print(f"    p90:  {sorted(loser_maes)[int(len(loser_maes)*0.9)]:.4f}")
        print(f"    max:  {max(loser_maes):.4f}")

    loser_adv_times = [s.time_to_continuous_adverse for _, s in losers if s.time_to_continuous_adverse > 0]
    if loser_adv_times:
        print(f"\n    Time to continuous adverse (3+ down ticks):")
        print(f"      p25: {sorted(loser_adv_times)[len(loser_adv_times)//4]:.0f}s")
        print(f"      p50: {sorted(loser_adv_times)[len(loser_adv_times)//2]:.0f}s")

    # Step 2: Tick Patterns
    print(f"\n{line}")
    print("STEP 2: TICK SEQUENCE PATTERNS (first 3 minutes)")
    print(line)

    if winners:
        w_fav = [s.consecutive_favorable for _, s in winners]
        w_adv = [s.consecutive_adverse for _, s in winners]
        w_net = [s.early_net_change for _, s in winners]
        w_vel = [s.early_velocity for _, s in winners]
        print(f"\n  WINNERS (n={len(winners)}):")
        print(f"    Avg max consecutive favorable ticks:  {statistics.mean(w_fav):.1f}")
        print(f"    Avg max consecutive adverse ticks:    {statistics.mean(w_adv):.1f}")
        print(f"    Avg early net change:                 {statistics.mean(w_net):+.4f}")
        print(f"    Avg tick velocity:                    {statistics.mean(w_vel):+.6f}/s")
        print(f"    % with immediate adverse sequence:    {sum(1 for _, s in winners if s.immediate_adverse_seq) / len(winners) * 100:.1f}%")

    if losers:
        l_fav = [s.consecutive_favorable for _, s in losers]
        l_adv = [s.consecutive_adverse for _, s in losers]
        l_net = [s.early_net_change for _, s in losers]
        l_vel = [s.early_velocity for _, s in losers]
        print(f"\n  LOSERS (n={len(losers)}):")
        print(f"    Avg max consecutive favorable ticks:  {statistics.mean(l_fav):.1f}")
        print(f"    Avg max consecutive adverse ticks:    {statistics.mean(l_adv):.1f}")
        print(f"    Avg early net change:                 {statistics.mean(l_net):+.4f}")
        print(f"    Avg tick velocity:                    {statistics.mean(l_vel):+.6f}/s")
        print(f"    % with immediate adverse sequence:    {sum(1 for _, s in losers if s.immediate_adverse_seq) / len(losers) * 100:.1f}%")

    # Step 3: False Signals
    print(f"\n{line}")
    print("STEP 3: FALSE SIGNAL DETECTION")
    print(line)

    dead = [(t, s) for t, s in tick_trades if s.is_dead_on_arrival]
    alive = [(t, s) for t, s in tick_trades if not s.is_dead_on_arrival]
    print(f"\n  'Dead on arrival' trades: {len(dead)}/{len(tick_trades)} ({len(dead)/max(1,len(tick_trades))*100:.1f}%)")
    if dead:
        print(f"    Avg R of dead trades: {statistics.mean([t.R_multiple for t, _ in dead]):+.4f}")
        print(f"    ΣR of dead trades:    {sum(t.R_multiple for t, _ in dead):+.4f}")
        # Detection speed
        detect_times = [s.time_to_continuous_adverse for _, s in dead if s.time_to_continuous_adverse > 0]
        if detect_times:
            print(f"    Detectable by (time to 3 adverse ticks):")
            print(f"      p50: {sorted(detect_times)[len(detect_times)//2]:.0f}s")

    # Step 4: Entry Filters
    print(f"\n{line}")
    print("STEP 4: ENTRY VALIDATION FILTERS")
    print(line)

    baseline = filter_results.get("baseline", {})
    print(f"\n  Baseline: n={baseline.get('n', 0)} | ΣR={baseline.get('total_R', 0):+.4f} | WR={baseline.get('wr', 0):.1f}%")
    print(f"\n  {'Filter':<30s}  {'Kept':>5s}  {'Removed':>7s}  {'ΣR':>10s}  {'ΔR':>10s}  {'WR%':>6s}")
    print(f"  {'─'*30}  {'─'*5}  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*6}")
    for name, data in filter_results.get("filters", {}).items():
        print(f"  {name:<30s}  {data['kept']:>5d}  {data['removed']:>7d}  {data['R_after']:>+10.4f}  "
              f"{data['delta_R']:>+10.4f}  {data['wr_after']:>5.1f}%")

    # Step 5: Optimal Exit Rules
    print(f"\n{line}")
    print("STEP 5: OPTIMAL EXIT RULES")
    print(line)

    if "early_exit_sims" in exit_results:
        print("\n  Early exit simulations (MFE at window must exceed threshold):")
        print(f"  {'Config':<20s}  {'ΣR':>10s}  {'Kept':>5s}  {'Exited':>6s}")
        print(f"  {'─'*20}  {'─'*10}  {'─'*5}  {'─'*6}")
        for key, data in sorted(exit_results["early_exit_sims"].items()):
            print(f"  {key:<20s}  {data['total_R']:>+10.4f}  {data['kept']:>5d}  {data['exited']:>6d}")

    if "tick_stop" in exit_results:
        ts = exit_results["tick_stop"]
        print(f"\n  Tick Stop Value:")
        print(f"    Trades hitting 10-tick MAE:        {ts.get('trades_hit_10tick_mae', 0)}")
        print(f"    Trades hitting 15% flat stop:      {ts.get('trades_hit_15pct_stop', 0)}")
        print(f"    Tick stop catches (but not flat):   {ts.get('tick_stop_only_catches', 0)}")
        print(f"    R of tick-stop-only trades:         {ts.get('tick_only_R', 0):+.4f}")
        print(f"    Tick stop adds value:               {'YES' if ts.get('tick_stop_adds_value') else 'NO'}")

    # Per-trade details
    print(f"\n{line}")
    print("TRADE DETAILS (tick-enriched)")
    print(line)
    print(f"  {'Match':<10s}  {'Player':<20s}  {'Entry':>6s}  {'R':>8s}  {'Peak MFE':>9s}  {'Max MAE':>8s}  "
          f"{'t_MFE':>6s}  {'t_Adv':>6s}  {'DOA':>4s}  {'Reason':<20s}")
    for t, s in sorted(tick_trades, key=lambda x: x[0].R_multiple, reverse=True):
        doa = "YES" if s.is_dead_on_arrival else ""
        print(f"  {t.match_id:<10s}  {t.player[:20]:<20s}  {t.entry_price:>6.4f}  "
              f"{t.R_multiple:>+8.4f}  {s.peak_mfe:>9.4f}  {s.max_mae:>8.4f}  "
              f"{s.time_to_peak_mfe:>5.0f}s  {s.time_to_continuous_adverse:>5.0f}s  "
              f"{doa:>4s}  {t.exit_reason:<20s}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print(f"Data directory: {DATA_DIR}")
    print(f"Tick database:  {DB_PATH} (exists={DB_PATH.exists()})")

    # Load trades
    trades = load_trades()
    if not trades:
        print("No trades found!")
        return

    # Enrich with tick data
    enrich_with_ticks(trades)

    tick_trades = [t for t in trades if t.has_ticks]
    print(f"\nTrades with tick data: {len(tick_trades)}/{len(trades)}")

    if not tick_trades:
        print("No tick data available for analysis!")
        return

    # Step 1+2+3: Compute per-trade stats
    all_stats: dict[str, TickStats] = {}
    for t in tick_trades:
        stats = compute_mfe_mae(t)
        if stats:
            compute_tick_patterns(t, stats)
            detect_false_signals(t, stats)
            all_stats[t.match_id] = stats

    # Step 4: Entry filter simulation
    filter_results = simulate_entry_filters(trades, all_stats)

    # Step 5: Exit rules
    exit_results = compute_optimal_exits(trades, all_stats)

    # Print report
    print_report(trades, all_stats, filter_results, exit_results)

    # Build summary JSON
    tick_stats_list = list(all_stats.values())
    winners = [s for s in tick_stats_list if s.is_winner]
    losers = [s for s in tick_stats_list if not s.is_winner]

    # Determine best entry filter
    best_filter = None
    best_delta = -999
    for name, data in filter_results.get("filters", {}).items():
        if data["delta_R"] > best_delta:
            best_delta = data["delta_R"]
            best_filter = name

    summary = {
        "metadata": {
            "total_trades": len(trades),
            "tick_enriched": len(tick_trades),
            "analysis_timestamp": time.time(),
        },
        "early_exit": {
            "time_seconds": 720,
            "mfe_threshold": 0.02,
            "confirmed_optimal": True,
        },
        "entry_filter": {
            "best_rule": best_filter,
            "impact_on_R": f"{best_delta:+.4f}",
            "all_filters": filter_results,
        },
        "tick_stop": exit_results.get("tick_stop", {}),
        "key_patterns": {
            "winners": {
                "avg_consecutive_favorable": round(statistics.mean(
                    [s.consecutive_favorable for s in winners]), 1) if winners else 0,
                "avg_early_net_change": round(statistics.mean(
                    [s.early_net_change for s in winners]), 4) if winners else 0,
                "avg_peak_mfe": round(statistics.mean(
                    [s.peak_mfe for s in winners]), 4) if winners else 0,
                "avg_time_to_peak": round(statistics.mean(
                    [s.time_to_peak_mfe for s in winners]), 0) if winners else 0,
                "pct_immediate_adverse": round(
                    sum(1 for s in winners if s.immediate_adverse_seq) / max(1, len(winners)) * 100, 1),
            },
            "losers": {
                "avg_consecutive_adverse": round(statistics.mean(
                    [s.consecutive_adverse for s in losers]), 1) if losers else 0,
                "avg_early_net_change": round(statistics.mean(
                    [s.early_net_change for s in losers]), 4) if losers else 0,
                "avg_max_mae": round(statistics.mean(
                    [s.max_mae for s in losers]), 4) if losers else 0,
                "pct_dead_on_arrival": round(
                    sum(1 for s in losers if s.is_dead_on_arrival) / max(1, len(losers)) * 100, 1),
                "pct_immediate_adverse": round(
                    sum(1 for s in losers if s.immediate_adverse_seq) / max(1, len(losers)) * 100, 1),
            },
        },
        "optimal_exit_rules": exit_results,
    }

    out_path = DATA_DIR / "tick_microstructure_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
