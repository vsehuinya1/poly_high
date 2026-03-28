#!/usr/bin/env python3
"""
Tennis Early MFE Window Optimization

Analyses lifecycle CSVs + tick-level data to determine optimal
early-exit window based on MFE timing distribution.

Run on VPS: python3 analyze_tennis_mfe.py
"""
import csv
import glob
import json
import os
import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "sports_data"
TICK_DB = DATA_DIR / "tick_history.db"


# ═══════════════════════════════════════════════════════════════
#  Trade Data Loading
# ═══════════════════════════════════════════════════════════════

@dataclass
class Trade:
    match_id: str
    player: str
    trigger: str
    entry_price: float
    fair_entry: float
    edge_entry: float
    entry_timestamp: float
    exit_timestamp: float
    exit_price: float
    exit_reason: str
    R_multiple: float
    duration_seconds: float
    entry_score: str = ""
    exit_score: str = ""
    # v3 fields
    mfe: float = 0.0
    mae: float = 0.0
    mae_ticks: int = 0
    peak_price: float = 0.0
    min_price_seen: float = 0.0
    capture_ratio: float = 0.0
    time_to_mfe: float = 0.0
    runner_v2_active: bool = False
    spread: float = 0.0
    selection_id: str = ""
    # Computed from tick data
    time_to_mfe_001: Optional[float] = None  # seconds to reach +0.01
    time_to_mfe_002: Optional[float] = None  # seconds to reach +0.02
    mfe_at_60s: float = 0.0
    mfe_at_120s: float = 0.0
    mfe_at_180s: float = 0.0
    mfe_at_240s: float = 0.0
    mfe_at_300s: float = 0.0
    price_path: list = field(default_factory=list)  # [(ts, mid), ...]


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: str, default: int = 0) -> int:
    try:
        return int(val) if val else default
    except (ValueError, TypeError):
        return default


def load_lifecycle_csvs() -> list[Trade]:
    """Load all tennis_trade_lifecycle CSVs, handling all schema versions.

    Schemas:
      v1 (Mar 5-9): no schema_version prefix, 17 columns
      v2 (Mar 10-25): schema_version=2, runner_mode, 32 columns
      v3 (Mar 26+):  schema_version=3, runner_v2_active, 36 columns
    """
    trades = []
    files = sorted(glob.glob(str(DATA_DIR / "tennis_trade_lifecycle_*.csv")))
    print(f"Found {len(files)} lifecycle CSV files")

    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if not headers:
                continue

            has_schema = "schema_version" in headers
            has_mfe = "mfe" in headers
            has_mae_ticks = "mae_ticks" in headers
            has_time_to_mfe = "time_to_mfe" in headers
            has_runner_v2 = "runner_v2_active" in headers
            has_capture_ratio = "capture_ratio" in headers

            file_count = 0
            for row in reader:
                try:
                    # Common fields across all schemas
                    entry_price = _safe_float(row.get("entry_price"))
                    if entry_price <= 0:
                        continue

                    t = Trade(
                        match_id=row.get("match_id", ""),
                        player=row.get("player", ""),
                        trigger=row.get("trigger", ""),
                        entry_price=entry_price,
                        fair_entry=_safe_float(row.get("fair_entry")),
                        edge_entry=_safe_float(row.get("edge_entry")),
                        entry_timestamp=_safe_float(row.get("timestamp_entry")),
                        exit_timestamp=_safe_float(row.get("timestamp_exit")),
                        exit_price=_safe_float(row.get("exit_price")),
                        exit_reason=row.get("exit_reason", ""),
                        R_multiple=_safe_float(row.get("R_multiple")),
                        entry_score=row.get("entry_score", ""),
                        exit_score=row.get("exit_score", ""),
                        duration_seconds=_safe_float(row.get("duration_seconds")),
                        spread=_safe_float(row.get("spread")),
                        peak_price=_safe_float(row.get("peak_price")),
                    )

                    # Schema-dependent fields
                    if has_mfe:
                        t.mfe = _safe_float(row.get("mfe"))
                        t.mae = _safe_float(row.get("mae"))
                        t.min_price_seen = _safe_float(row.get("min_price_seen"))
                    else:
                        # v1: estimate MFE from snapshot prices
                        snapshots = [
                            _safe_float(row.get("price_t5")),
                            _safe_float(row.get("price_t15")),
                            _safe_float(row.get("price_t30")),
                            _safe_float(row.get("price_t60")),
                        ]
                        best_snap = max((s for s in snapshots if s > 0), default=0)
                        if best_snap > entry_price:
                            t.mfe = best_snap - entry_price
                        t.peak_price = max(best_snap, entry_price)

                    if has_mae_ticks:
                        t.mae_ticks = _safe_int(row.get("mae_ticks"))

                    if has_time_to_mfe:
                        t.time_to_mfe = _safe_float(row.get("time_to_mfe"))

                    if has_capture_ratio:
                        t.capture_ratio = _safe_float(row.get("capture_ratio"))

                    if has_runner_v2:
                        t.runner_v2_active = row.get("runner_v2_active") == "1"
                    elif "runner_mode" in headers:
                        t.runner_v2_active = row.get("runner_mode") == "1"

                    # Skip trades with zero duration or invalid timestamps
                    if t.entry_timestamp > 0 and t.exit_timestamp > 0:
                        trades.append(t)
                        file_count += 1
                except (ValueError, IndexError, KeyError) as e:
                    continue

            print(f"  {fname}: {file_count} trades (cols={len(headers)})")

    print(f"Loaded {len(trades)} valid trades")
    return trades


# ═══════════════════════════════════════════════════════════════
#  Tick Data Enrichment
# ═══════════════════════════════════════════════════════════════

def enrich_with_ticks(trades: list[Trade]) -> list[Trade]:
    """Cross-reference trades with tick DB to compute time-to-MFE thresholds."""
    if not TICK_DB.exists():
        print(f"WARNING: tick DB not found at {TICK_DB}, using lifecycle MFE only")
        return trades

    conn = sqlite3.connect(str(TICK_DB))

    # Get all token_labels for matching
    labels = {}
    try:
        cursor = conn.execute("SELECT token_id, market_title, sport FROM token_labels")
        for tid, title, sport in cursor:
            labels[tid] = (title, sport)
    except Exception:
        pass

    # For each trade, try to find matching tick data
    enriched = 0
    for trade in trades:
        # Try to find token_id by match_id in token_labels
        matching_tokens = []
        for tid, (title, sport) in labels.items():
            if sport and "tennis" in sport.lower():
                matching_tokens.append(tid)

        # Try match_id based lookup — the match_id may be stored in game_id
        try:
            cursor = conn.execute(
                "SELECT token_id FROM token_labels WHERE game_id = ?",
                (trade.match_id,)
            )
            token_ids = [r[0] for r in cursor]
        except Exception:
            token_ids = []

        if not token_ids:
            # Try by player name in market_title
            try:
                cursor = conn.execute(
                    "SELECT token_id FROM token_labels WHERE market_title LIKE ?",
                    (f"%{trade.player}%",)
                )
                token_ids = [r[0] for r in cursor]
            except Exception:
                continue

        if not token_ids:
            continue

        # Get tick data for the trade window
        for token_id in token_ids:
            cursor = conn.execute(
                "SELECT timestamp, mid FROM ticks WHERE token_id = ? "
                "AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
                (token_id, trade.entry_timestamp - 5,
                 trade.exit_timestamp + 60)
            )
            ticks = list(cursor)

            if len(ticks) < 3:
                continue

            trade.price_path = ticks
            trade.selection_id = token_id

            # Compute time-to-MFE at thresholds
            for ts, mid in ticks:
                elapsed = ts - trade.entry_timestamp
                favorable = mid - trade.entry_price

                # Time to first +0.01 MFE
                if trade.time_to_mfe_001 is None and favorable >= 0.01:
                    trade.time_to_mfe_001 = elapsed

                # Time to first +0.02 MFE
                if trade.time_to_mfe_002 is None and favorable >= 0.02:
                    trade.time_to_mfe_002 = elapsed

            # Compute MFE at specific time windows
            for ts, mid in ticks:
                elapsed = ts - trade.entry_timestamp
                favorable = mid - trade.entry_price

                if elapsed <= 60:
                    trade.mfe_at_60s = max(trade.mfe_at_60s, favorable)
                if elapsed <= 120:
                    trade.mfe_at_120s = max(trade.mfe_at_120s, favorable)
                if elapsed <= 180:
                    trade.mfe_at_180s = max(trade.mfe_at_180s, favorable)
                if elapsed <= 240:
                    trade.mfe_at_240s = max(trade.mfe_at_240s, favorable)
                if elapsed <= 300:
                    trade.mfe_at_300s = max(trade.mfe_at_300s, favorable)

            enriched += 1
            break  # use first matching token

    conn.close()
    print(f"Enriched {enriched}/{len(trades)} trades with tick data")
    return trades


# ═══════════════════════════════════════════════════════════════
#  Analysis
# ═══════════════════════════════════════════════════════════════

def percentiles(values: list[float], pcts=(25, 50, 75, 90)) -> dict:
    """Compute percentiles for a list of values."""
    if not values:
        return {f"p{p}": None for p in pcts}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result = {}
    for p in pcts:
        k = (p / 100.0) * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        d = k - f
        result[f"p{p}"] = sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])
    return result


def analyze_mfe_timing(trades: list[Trade]) -> dict:
    """Full MFE timing analysis."""
    results = {}

    winners = [t for t in trades if t.R_multiple > 0]
    losers = [t for t in trades if t.R_multiple <= 0]

    print(f"\n{'='*80}")
    print(f"TENNIS EARLY MFE WINDOW OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Total trades: {len(trades)}")
    print(f"Winners: {len(winners)} ({100*len(winners)/len(trades):.1f}%)")
    print(f"Losers:  {len(losers)} ({100*len(losers)/len(trades):.1f}%)")

    # ── Exit reason breakdown ──────────────────────────────────
    print(f"\n{'─'*60}")
    print("EXIT REASON BREAKDOWN")
    print(f"{'─'*60}")
    reasons = {}
    for t in trades:
        r = t.exit_reason
        if r not in reasons:
            reasons[r] = {"count": 0, "total_R": 0.0, "wins": 0}
        reasons[r]["count"] += 1
        reasons[r]["total_R"] += t.R_multiple
        if t.R_multiple > 0:
            reasons[r]["wins"] += 1

    for reason, stats in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        wr = 100 * stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        avg_r = stats["total_R"] / stats["count"]
        print(f"  {reason:<25} n={stats['count']:>3}  WR={wr:>5.1f}%  avgR={avg_r:>+.4f}  ΣR={stats['total_R']:>+.4f}")

    # ── 1. Time-to-MFE distribution (winners) ─────────────────
    print(f"\n{'─'*60}")
    print("1. TIME-TO-MFE DISTRIBUTION (winners only)")
    print(f"{'─'*60}")

    # From lifecycle data (time_to_mfe = time to peak MFE)
    winner_ttm = [t.time_to_mfe for t in winners if t.time_to_mfe > 0]
    if winner_ttm:
        pcts = percentiles(winner_ttm)
        print(f"\n  Time to PEAK MFE (from lifecycle):")
        print(f"    n={len(winner_ttm)}")
        for k, v in pcts.items():
            print(f"    {k}: {v:.1f}s ({v/60:.1f}min)")
        results["time_to_peak_mfe"] = pcts

    # From tick data (time to specific thresholds)
    ttm_001 = [t.time_to_mfe_001 for t in winners if t.time_to_mfe_001 is not None]
    ttm_002 = [t.time_to_mfe_002 for t in winners if t.time_to_mfe_002 is not None]

    if ttm_001:
        pcts = percentiles(ttm_001)
        print(f"\n  Time to +0.01 MFE (tick data):")
        print(f"    n={len(ttm_001)} / {len(winners)} winners")
        for k, v in pcts.items():
            print(f"    {k}: {v:.1f}s ({v/60:.1f}min)")
        results["time_to_mfe_001"] = pcts

    if ttm_002:
        pcts = percentiles(ttm_002)
        print(f"\n  Time to +0.02 MFE (tick data):")
        print(f"    n={len(ttm_002)} / {len(winners)} winners")
        for k, v in pcts.items():
            print(f"    {k}: {v:.1f}s ({v/60:.1f}min)")
        results["time_to_mfe_002"] = pcts

    # ── 2. Loser behavior ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print("2. LOSER BEHAVIOR")
    print(f"{'─'*60}")

    if losers:
        # From lifecycle MFE field
        losers_zero_mfe = sum(1 for t in losers if t.mfe <= 0)
        losers_small_mfe = sum(1 for t in losers if 0 < t.mfe < 0.01)
        losers_mid_mfe = sum(1 for t in losers if 0.01 <= t.mfe < 0.02)
        losers_large_mfe = sum(1 for t in losers if t.mfe >= 0.02)

        print(f"\n  Loser MFE distribution (lifecycle data):")
        print(f"    Zero MFE (never moved up):     {losers_zero_mfe:>3} ({100*losers_zero_mfe/len(losers):.1f}%)")
        print(f"    MFE 0-0.01 (tiny):             {losers_small_mfe:>3} ({100*losers_small_mfe/len(losers):.1f}%)")
        print(f"    MFE 0.01-0.02 (small):         {losers_mid_mfe:>3} ({100*losers_mid_mfe/len(losers):.1f}%)")
        print(f"    MFE >= 0.02 (reached then lost):{losers_large_mfe:>3} ({100*losers_large_mfe/len(losers):.1f}%)")

        results["loser_zero_mfe_pct"] = 100 * losers_zero_mfe / len(losers)
        results["loser_below_001_pct"] = 100 * (losers_zero_mfe + losers_small_mfe) / len(losers)
        results["loser_reached_001_not_002_pct"] = 100 * losers_mid_mfe / len(losers)

        # From tick data
        losers_no_001 = sum(1 for t in losers if t.time_to_mfe_001 is None and t.price_path)
        losers_with_ticks = sum(1 for t in losers if t.price_path)
        if losers_with_ticks > 0:
            print(f"\n  Tick-level loser analysis ({losers_with_ticks} with tick data):")
            print(f"    Never reached +0.01 MFE:  {losers_no_001} ({100*losers_no_001/losers_with_ticks:.1f}%)")

            losers_001_not_002 = sum(
                1 for t in losers
                if t.time_to_mfe_001 is not None and t.time_to_mfe_002 is None and t.price_path
            )
            print(f"    Reached +0.01 not +0.02:  {losers_001_not_002} ({100*losers_001_not_002/losers_with_ticks:.1f}%)")

        # Average time before continuous adverse movement (for losers)
        adverse_times = []
        for t in losers:
            if not t.price_path:
                continue
            # Find first tick that goes below entry and stays below
            for ts, mid in t.price_path:
                elapsed = ts - t.entry_timestamp
                if elapsed > 0 and mid < t.entry_price:
                    adverse_times.append(elapsed)
                    break

        if adverse_times:
            avg_adverse = statistics.mean(adverse_times)
            med_adverse = statistics.median(adverse_times)
            print(f"\n  Time before first adverse move:")
            print(f"    Mean:   {avg_adverse:.1f}s ({avg_adverse/60:.1f}min)")
            print(f"    Median: {med_adverse:.1f}s ({med_adverse/60:.1f}min)")

    # ── 3. All trades MFE distribution ─────────────────────────
    print(f"\n{'─'*60}")
    print("3. ALL TRADES — MFE DISTRIBUTION")
    print(f"{'─'*60}")

    all_mfe = [t.mfe for t in trades if t.mfe > 0]
    if all_mfe:
        pcts = percentiles(all_mfe)
        print(f"  MFE (absolute) percentiles:")
        for k, v in pcts.items():
            print(f"    {k}: {v:.4f}")

    # MFE as % of entry
    mfe_pct = [t.mfe / t.entry_price for t in trades if t.mfe > 0 and t.entry_price > 0]
    if mfe_pct:
        pcts_pct = percentiles(mfe_pct)
        print(f"\n  MFE (% of entry) percentiles:")
        for k, v in pcts_pct.items():
            print(f"    {k}: {100*v:.2f}%")

    # Duration distribution
    durations = [t.duration_seconds for t in trades]
    if durations:
        pcts_dur = percentiles(durations)
        print(f"\n  Duration percentiles:")
        for k, v in pcts_dur.items():
            print(f"    {k}: {v:.0f}s ({v/60:.1f}min)")

    # ── 4. Early exit simulation ───────────────────────────────
    print(f"\n{'─'*60}")
    print("4. EARLY EXIT SIMULATION")
    print(f"    If MFE < threshold at cutoff → exit at current price")
    print(f"{'─'*60}")

    windows = [60, 120, 180, 240, 300]
    thresholds = [0.01, 0.02]
    sim_results = {}

    # Baseline stats
    baseline_R = sum(t.R_multiple for t in trades)
    baseline_wins = sum(1 for t in trades if t.R_multiple > 0)
    baseline_wr = 100 * baseline_wins / len(trades) if trades else 0
    win_rs = [t.R_multiple for t in trades if t.R_multiple > 0]
    loss_rs = [t.R_multiple for t in trades if t.R_multiple <= 0]
    baseline_avg_win = statistics.mean(win_rs) if win_rs else 0
    baseline_avg_loss = statistics.mean(loss_rs) if loss_rs else 0

    print(f"\n  BASELINE: ΣR={baseline_R:+.4f} | WR={baseline_wr:.1f}% | "
          f"avgW={baseline_avg_win:+.4f} avgL={baseline_avg_loss:+.4f} | n={len(trades)}")
    print()

    for threshold in thresholds:
        print(f"  MFE threshold: {threshold}")
        print(f"  {'Window':<8} {'ΣR':>8} {'ΔΣR':>8} {'WR%':>6} {'avgW':>8} {'avgL':>8} {'Exited':>8} {'Saved':>8}")
        print(f"  {'─'*70}")

        for window in windows:
            sim_R = 0.0
            sim_wins = 0
            sim_losses = 0
            sim_early_exits = 0
            saved_by_exit = 0.0

            for t in trades:
                # Determine MFE at this window
                # Use tick-level data if available, else approximate from lifecycle
                mfe_at_window = 0.0

                if t.price_path:
                    # Compute from actual ticks
                    for ts, mid in t.price_path:
                        elapsed = ts - t.entry_timestamp
                        if elapsed <= window:
                            fav = mid - t.entry_price
                            if fav > mfe_at_window:
                                mfe_at_window = fav
                else:
                    # Approximate: use lifecycle MFE and time_to_mfe
                    if t.time_to_mfe > 0 and t.time_to_mfe <= window:
                        mfe_at_window = t.mfe
                    elif t.duration_seconds <= window:
                        mfe_at_window = t.mfe
                    else:
                        # Use snapshot prices to approximate
                        if window <= 300 and t.mfe > 0:
                            if t.time_to_mfe > 0:
                                if window >= t.time_to_mfe:
                                    mfe_at_window = t.mfe
                                else:
                                    # Linear interpolation (rough)
                                    mfe_at_window = t.mfe * (window / t.time_to_mfe)
                            else:
                                mfe_at_window = 0.0

                if mfe_at_window < threshold and t.duration_seconds > window:
                    # Would exit early
                    sim_early_exits += 1

                    # Estimate exit price at window time
                    exit_price_at_window = t.entry_price
                    if t.price_path:
                        # Get actual price at window time
                        for ts, mid in t.price_path:
                            elapsed = ts - t.entry_timestamp
                            if elapsed <= window:
                                exit_price_at_window = mid
                    else:
                        # Approximate: use entry price (conservative)
                        exit_price_at_window = t.entry_price + mfe_at_window * 0.5

                    base = t.entry_price
                    sim_r = (exit_price_at_window - base) / base if base > 0 else 0
                    sim_R += sim_r

                    if t.R_multiple < sim_r:
                        saved_by_exit += (sim_r - t.R_multiple)

                    if sim_r > 0:
                        sim_wins += 1
                    else:
                        sim_losses += 1
                else:
                    # Keep original trade
                    sim_R += t.R_multiple
                    if t.R_multiple > 0:
                        sim_wins += 1
                    else:
                        sim_losses += 1

            total = sim_wins + sim_losses
            wr = 100 * sim_wins / total if total > 0 else 0

            all_wins = [t.R_multiple for t in trades if t.R_multiple > 0]
            all_losses = [t.R_multiple for t in trades if t.R_multiple <= 0]
            avg_w = statistics.mean(all_wins) if all_wins else 0
            avg_l = statistics.mean(all_losses) if all_losses else 0

            delta_r = sim_R - baseline_R

            print(f"  {window:>4}s   {sim_R:>+8.4f} {delta_r:>+8.4f} {wr:>5.1f}% "
                  f"{avg_w:>+8.4f} {avg_l:>+8.4f} {sim_early_exits:>8} {saved_by_exit:>+8.4f}")

            sim_results[f"{window}s_mfe{threshold}"] = {
                "window_seconds": window,
                "mfe_threshold": threshold,
                "total_R": sim_R,
                "delta_R": delta_r,
                "win_rate": wr,
                "early_exits": sim_early_exits,
                "saved_R": saved_by_exit,
            }
        print()

    # ── 5. Per-trade detail table ──────────────────────────────
    print(f"\n{'─'*60}")
    print("5. PER-TRADE DETAIL")
    print(f"{'─'*60}")
    print(f"{'ID':<8} {'Player':<20} {'Entry':>6} {'Exit':>6} {'R':>8} "
          f"{'MFE':>6} {'MAE':>6} {'Dur':>6} {'t2MFE':>6} {'t→01':>6} {'t→02':>6} {'Reason':<20}")
    print(f"{'─'*120}")

    for t in sorted(trades, key=lambda x: x.entry_timestamp):
        t01 = f"{t.time_to_mfe_001:.0f}" if t.time_to_mfe_001 is not None else "—"
        t02 = f"{t.time_to_mfe_002:.0f}" if t.time_to_mfe_002 is not None else "—"
        ttm = f"{t.time_to_mfe:.0f}" if t.time_to_mfe > 0 else "—"
        print(f"{t.match_id:<8} {t.player[:20]:<20} {t.entry_price:>6.4f} {t.exit_price:>6.4f} "
              f"{t.R_multiple:>+8.4f} {t.mfe:>6.4f} {t.mae:>6.4f} "
              f"{t.duration_seconds:>5.0f}s {ttm:>6} {t01:>6} {t02:>6} {t.exit_reason:<20}")

    # ── 6. Recommendation ─────────────────────────────────────
    print(f"\n{'='*80}")
    print("6. RECOMMENDATION")
    print(f"{'='*80}")

    # Find best simulation
    best_key = max(sim_results, key=lambda k: sim_results[k]["delta_R"]) if sim_results else None
    best = sim_results.get(best_key, {})

    # Also check current setting performance
    current_window = 720   # EARLY_EXIT_WINDOW_S
    current_thresh = 0.02  # MIN_MFE_EARLY

    recommendation = {
        "current_settings": {
            "window_seconds": current_window,
            "min_mfe_threshold": current_thresh,
        },
        "baseline": {
            "total_R": round(baseline_R, 4),
            "win_rate": round(baseline_wr, 1),
            "n_trades": len(trades),
            "avg_win_R": round(baseline_avg_win, 4),
            "avg_loss_R": round(baseline_avg_loss, 4),
        },
        "simulation_results": sim_results,
    }

    if best:
        recommendation["optimal_window_seconds"] = best.get("window_seconds", current_window)
        recommendation["min_mfe_threshold"] = best.get("mfe_threshold", current_thresh)
        recommendation["reason"] = (
            f"Based on {len(trades)} trades: {best_key} gives best ΔR={best['delta_R']:+.4f}. "
            f"Would early-exit {best['early_exits']} trades, saving {best['saved_R']:+.4f}R."
        )
        recommendation["expected_R_improvement"] = f"{best['delta_R']:+.4f}"

        print(f"\n  Best simulation: {best_key}")
        print(f"  ΔR improvement:  {best['delta_R']:+.4f}")
        print(f"  Early exits:     {best['early_exits']}")
        print(f"  Saved R:         {best['saved_R']:+.4f}")
        print(f"\n  Current: window={current_window}s, threshold={current_thresh}")
        print(f"  Proposed: window={best.get('window_seconds')}s, threshold={best.get('mfe_threshold')}")
    else:
        print("  No simulation data — insufficient trades with tick data")
        recommendation["reason"] = "Insufficient tick-level data for simulation"

    # ── Output JSON ────────────────────────────────────────────
    output_path = DATA_DIR / "mfe_optimization_results.json"
    with open(output_path, "w") as f:
        json.dump(recommendation, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    return recommendation


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Data directory: {DATA_DIR}")
    print(f"Tick database:  {TICK_DB} (exists={TICK_DB.exists()})")
    print()

    trades = load_lifecycle_csvs()
    if not trades:
        print("ERROR: No trades found!")
        exit(1)

    trades = enrich_with_ticks(trades)
    results = analyze_mfe_timing(trades)
