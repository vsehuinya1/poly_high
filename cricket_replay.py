"""
Cricket Replay Engine — Tick-Level Alpha Measurement.

Replays tick_history.db against cricket event timestamps to measure:
  1. Exact latency advantage (price lag after events)
  2. Optimal hold time (5s → 300s forward curves)
  3. Concurrent position PnL (no POSITION_LOCK)
  4. Match phase filtering (powerplay / middle / death)
  5. Max alpha configuration

Usage: python3 cricket_replay.py
"""

import sqlite3
import re
import json
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field

# ── Configuration ──────────────────────────────────────────────────
TICK_DB = "sports_data/tick_history.db"
LOG_FILE = "logs/cricket_v11_5.log"
# Also check older logs for more data
EXTRA_LOGS = [
    "logs/cricket_v11_4_RELOAD.log",
    "logs/sports_20260427.log",
]

# Forward price snapshot intervals (seconds)
SNAPSHOTS = [5, 10, 15, 30, 60, 120, 180, 300]

# Hold time sweep for optimal exit
HOLD_TIMES = [15, 30, 60, 90, 120, 180, 240, 300]

# Match phases (T20 overs)
PHASES = {
    "POWERPLAY": (0, 6),
    "MIDDLE": (6, 15),
    "DEATH": (15, 20),
}

# Trailing stop configs to test
TRAILING_STOPS = [0.02, 0.03, 0.05, 0.08, 0.10]

# ── Data Classes ───────────────────────────────────────────────────

@dataclass
class CricketEvent:
    timestamp: float        # Unix timestamp
    fixture_id: int
    event_type: str        # WICKET, BOUNDARY, SURGE, DOT
    runs_delta: int
    wickets_delta: int
    total_runs: int
    total_wickets: int
    overs: float
    run_rate: float
    log_time: str          # HH:MM:SS from log

@dataclass
class ReplayResult:
    event: CricketEvent
    entry_mid: float
    direction: str         # LONG or SHORT
    forward_prices: dict   # {seconds: mid_price}
    mfe: float = 0.0       # Max favorable excursion
    mae: float = 0.0       # Max adverse excursion
    mfe_time: float = 0.0  # Time to MFE
    phase: str = ""        # POWERPLAY, MIDDLE, DEATH
    best_exit: float = 0.0
    best_hold: int = 0


# ── Parse Events from Logs ─────────────────────────────────────────

def parse_events_from_log(log_path: str) -> list[CricketEvent]:
    """Extract CRICKET_EVENT_DETECTED entries from log."""
    events = []
    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}).*CRICKET_EVENT_DETECTED \| "
        r"fixture=(\d+) \| event=(\w+) \| "
        r"runs_delta=(\d+) wickets_delta=(\d+) \| "
        r"(\d+)/(\d+) \(([\d.]+) ov\) \| "
        r"RR=([\d.]+)"
    )

    if not os.path.exists(log_path):
        return events

    # Determine the log date from the filename or file modification time
    mtime = os.path.getmtime(log_path)
    log_date = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")

    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            log_time = m.group(1)
            # Parse full timestamp from log line if available
            ts_match = re.match(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
            if ts_match:
                dt = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                ts = dt.timestamp()
            else:
                # Fallback: use log_date + time
                dt = datetime.strptime(f"{log_date} {log_time}", "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                ts = dt.timestamp()

            event = CricketEvent(
                timestamp=ts,
                fixture_id=int(m.group(2)),
                event_type=m.group(3),
                runs_delta=int(m.group(4)),
                wickets_delta=int(m.group(5)),
                total_runs=int(m.group(6)),
                total_wickets=int(m.group(7)),
                overs=float(m.group(8)),
                run_rate=float(m.group(9)),
                log_time=log_time,
            )
            events.append(event)

    return events


# ── Determine Trade Direction ──────────────────────────────────────

def get_direction(event: CricketEvent) -> str:
    """Determine trade direction based on event type.
    
    WICKET → SHORT (batting team weakened, price drops)
    BOUNDARY/SURGE → LONG (batting team strengthened, price rises)
    DOT → skip (no significant price movement expected)
    """
    if event.event_type == "WICKET":
        return "SHORT"
    elif event.event_type in ("BOUNDARY", "SURGE"):
        return "LONG"
    return ""


def get_phase(overs: float) -> str:
    """Classify match phase from overs."""
    for phase, (start, end) in PHASES.items():
        if start <= overs < end:
            return phase
    return "DEATH" if overs >= 20 else "UNKNOWN"


# ── Tick Database Query ────────────────────────────────────────────

def get_token_ids_for_fixture(conn: sqlite3.Connection, fixture_id: int) -> list[str]:
    """Get token IDs that were actively traded during a fixture's time window."""
    cur = conn.cursor()

    # Get IPL token IDs from token_labels
    cur.execute("""
        SELECT token_id, market_title FROM token_labels 
        WHERE sport='cricket' AND market_title LIKE '%Indian Premier League%'
    """)
    ipl_tokens = cur.fetchall()

    # Map fixture IDs to approximate team names
    fixture_teams = {}
    for tid, title in ipl_tokens:
        # Extract team names from title
        clean = title.replace("Indian Premier League: ", "")
        fixture_teams.setdefault(clean, []).append(tid)

    return ipl_tokens


def get_price_at_time(conn: sqlite3.Connection, token_id: str, ts: float, window: float = 3.0) -> float | None:
    """Get the mid price closest to a given timestamp."""
    cur = conn.cursor()
    cur.execute("""
        SELECT mid, ABS(timestamp - ?) as dt FROM ticks 
        WHERE token_id = ? AND timestamp BETWEEN ? AND ?
        ORDER BY dt ASC LIMIT 1
    """, (ts, token_id, ts - window, ts + window))
    row = cur.fetchone()
    return row[0] if row else None


def get_price_series(conn: sqlite3.Connection, token_id: str, start_ts: float, duration: float = 300.0) -> list[tuple[float, float]]:
    """Get all ticks for a token in a time window. Returns [(relative_time, mid), ...]."""
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp - ?, mid FROM ticks 
        WHERE token_id = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """, (start_ts, token_id, start_ts, start_ts + duration))
    return cur.fetchall()


def get_forward_prices(conn: sqlite3.Connection, token_id: str, event_ts: float) -> dict[int, float]:
    """Get price snapshots at each forward interval."""
    result = {}
    for s in SNAPSHOTS:
        p = get_price_at_time(conn, token_id, event_ts + s, window=5.0)
        if p is not None:
            result[s] = p
    return result


# ── Find Active Token for a Fixture ────────────────────────────────

def find_active_token(conn: sqlite3.Connection, event_ts: float, fixture_id: int) -> str | None:
    """Find the IPL token that was actively trading at event time.
    
    Strategy: look at all IPL tokens, find which ones have ticks near the event time.
    The "home team" token is the one with mid > 0.10 and mid < 0.90 (competitive price).
    """
    cur = conn.cursor()
    
    # Get ALL IPL tokens
    cur.execute("""
        SELECT token_id, market_title FROM token_labels 
        WHERE sport='cricket' AND market_title LIKE '%Indian Premier League%'
    """)
    ipl_tokens = cur.fetchall()

    candidates = []
    for tid, title in ipl_tokens:
        # Check if this token has ticks near the event
        cur.execute("""
            SELECT mid, COUNT(*) as cnt FROM ticks 
            WHERE token_id = ? AND timestamp BETWEEN ? AND ?
        """, (tid, event_ts - 30, event_ts + 30))
        row = cur.fetchone()
        if row and row[1] > 0 and row[0] is not None:
            mid = row[0]
            # Only interested in tokens with competitive prices
            if 0.01 < mid < 0.99:
                candidates.append((tid, title, mid, row[1]))

    if not candidates:
        return None

    # Return the token with the most activity
    candidates.sort(key=lambda x: -x[3])
    return candidates[0][0]


# ── Replay Engine ──────────────────────────────────────────────────

def replay_events(events: list[CricketEvent], conn: sqlite3.Connection) -> list[ReplayResult]:
    """Replay all events against tick data."""
    results = []
    
    # Cache: fixture_id → active token_id
    fixture_tokens: dict[int, str] = {}
    
    for i, event in enumerate(events):
        direction = get_direction(event)
        if not direction:
            continue  # Skip DOT events

        # Find active token
        if event.fixture_id not in fixture_tokens:
            token = find_active_token(conn, event.timestamp, event.fixture_id)
            if token:
                fixture_tokens[event.fixture_id] = token
            else:
                continue
        else:
            token = fixture_tokens[event.fixture_id]

        # Get entry price
        entry_mid = get_price_at_time(conn, token, event.timestamp, window=5.0)
        if entry_mid is None:
            continue

        # Get forward prices
        fwd = get_forward_prices(conn, token, event.timestamp)
        if not fwd:
            continue

        # Get full price series for MFE/MAE
        series = get_price_series(conn, token, event.timestamp, duration=310.0)

        # Calculate MFE and MAE
        mfe = 0.0
        mae = 0.0
        mfe_time = 0.0
        for dt, price in series:
            if direction == "LONG":
                move = price - entry_mid
            else:
                move = entry_mid - price

            if move > mfe:
                mfe = move
                mfe_time = dt
            if move < mae:
                mae = move

        # Determine best hold time
        best_exit = 0.0
        best_hold = 0
        for s, price in sorted(fwd.items()):
            if direction == "LONG":
                pnl = price - entry_mid
            else:
                pnl = entry_mid - price
            if pnl > best_exit:
                best_exit = pnl
                best_hold = s

        phase = get_phase(event.overs)

        result = ReplayResult(
            event=event,
            entry_mid=entry_mid,
            direction=direction,
            forward_prices=fwd,
            mfe=mfe,
            mae=mae,
            mfe_time=mfe_time,
            phase=phase,
            best_exit=best_exit,
            best_hold=best_hold,
        )
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(events)} events...")

    return results


# ── Trailing Stop Simulation ──────────────────────────────────────

def simulate_trailing_stop(conn: sqlite3.Connection, token: str, entry_ts: float,
                           entry_mid: float, direction: str, 
                           trail_pct: float, max_hold: int = 300) -> tuple[float, float, float]:
    """Simulate a trailing stop exit. Returns (exit_price, pnl, hold_time)."""
    series = get_price_series(conn, token, entry_ts, duration=max_hold + 10)
    
    peak_pnl = 0.0
    for dt, price in series:
        if direction == "LONG":
            pnl = price - entry_mid
        else:
            pnl = entry_mid - price

        if pnl > peak_pnl:
            peak_pnl = pnl

        # Check trailing stop
        drawdown = peak_pnl - pnl
        if peak_pnl > 0 and drawdown >= trail_pct:
            return (price, pnl, dt)

    # Hit max hold — exit at last price
    if series:
        last_dt, last_price = series[-1]
        if direction == "LONG":
            final_pnl = last_price - entry_mid
        else:
            final_pnl = entry_mid - last_price
        return (last_price, final_pnl, last_dt)

    return (entry_mid, 0.0, 0.0)


# ── Concurrent Position Simulation ─────────────────────────────────

def simulate_concurrent(results: list[ReplayResult], hold_time: int) -> dict:
    """Simulate trading all signals concurrently (no POSITION_LOCK).
    Returns aggregate stats for a given hold time.
    """
    total_pnl = 0.0
    wins = 0
    losses = 0
    trades = 0
    
    for r in results:
        if hold_time in r.forward_prices:
            price = r.forward_prices[hold_time]
            if r.direction == "LONG":
                pnl = price - r.entry_mid
            else:
                pnl = r.entry_mid - price
            total_pnl += pnl
            trades += 1
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

    return {
        "hold_time": hold_time,
        "trades": trades,
        "total_pnl": total_pnl,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / trades if trades > 0 else 0,
        "avg_pnl": total_pnl / trades if trades > 0 else 0,
    }


# ── Main Analysis ─────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CRICKET REPLAY ENGINE — Tick-Level Alpha Measurement")
    print("=" * 70)

    # 1. Parse events from all logs
    all_events = []
    for log_path in [LOG_FILE] + EXTRA_LOGS:
        events = parse_events_from_log(log_path)
        if events:
            print(f"  {log_path}: {len(events)} events")
            all_events.extend(events)

    # Deduplicate by (fixture_id, timestamp)
    seen = set()
    unique_events = []
    for e in all_events:
        key = (e.fixture_id, round(e.timestamp, 0))
        if key not in seen:
            seen.add(key)
            unique_events.append(e)

    # Sort by timestamp
    unique_events.sort(key=lambda e: e.timestamp)
    print(f"\nTotal unique events: {len(unique_events)}")
    
    # Count by type
    by_type = defaultdict(int)
    for e in unique_events:
        by_type[e.event_type] += 1
    print("By type:", dict(by_type))

    # Filter to actionable events (WICKET, BOUNDARY, SURGE)
    actionable = [e for e in unique_events if e.event_type in ("WICKET", "BOUNDARY", "SURGE")]
    print(f"Actionable events (WICKET/BOUNDARY/SURGE): {len(actionable)}")

    # 2. Connect to tick database
    if not os.path.exists(TICK_DB):
        print(f"ERROR: {TICK_DB} not found")
        sys.exit(1)

    conn = sqlite3.connect(TICK_DB)

    # 3. Replay events
    print("\n" + "-" * 70)
    print("REPLAYING EVENTS AGAINST TICK DATA...")
    print("-" * 70)
    results = replay_events(actionable, conn)
    print(f"\nMatched {len(results)} events to tick data")

    if not results:
        print("No results — no tick data matched events")
        conn.close()
        return

    # ═════════════════════════════════════════════════════════════════
    # 4. ANALYSIS: Latency Advantage
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("1. LATENCY ADVANTAGE — Forward Price Curves (all events)")
    print("=" * 70)

    for s in SNAPSHOTS:
        moves = []
        for r in results:
            if s in r.forward_prices:
                if r.direction == "LONG":
                    move = r.forward_prices[s] - r.entry_mid
                else:
                    move = r.entry_mid - r.forward_prices[s]
                moves.append(move)
        
        if moves:
            avg = sum(moves) / len(moves)
            pos = sum(1 for m in moves if m > 0)
            total_r = sum(moves)
            print(f"  t+{s:>3d}s: avg={avg:+.4f} | win_rate={pos}/{len(moves)} ({pos/len(moves)*100:.0f}%) | total_R={total_r:+.4f}")

    # ═════════════════════════════════════════════════════════════════
    # 5. ANALYSIS: Optimal Hold Time (Concurrent Positions)
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("2. OPTIMAL HOLD TIME — Concurrent Positions (no POSITION_LOCK)")
    print("=" * 70)

    best_config = None
    for ht in HOLD_TIMES:
        stats = simulate_concurrent(results, ht)
        label = f"  hold={ht:>3d}s: "
        if stats["trades"] > 0:
            print(f"{label}trades={stats['trades']} | pnl={stats['total_pnl']:+.4f} | "
                  f"wr={stats['win_rate']*100:.0f}% | avg={stats['avg_pnl']:+.4f}")
            if best_config is None or stats["total_pnl"] > best_config["total_pnl"]:
                best_config = stats
        else:
            print(f"{label}no data")

    if best_config:
        print(f"\n  ★ BEST: hold={best_config['hold_time']}s → "
              f"total_R={best_config['total_pnl']:+.4f} | "
              f"wr={best_config['win_rate']*100:.0f}% over {best_config['trades']} trades")

    # ═════════════════════════════════════════════════════════════════
    # 6. ANALYSIS: Match Phase Filtering
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3. MATCH PHASE ANALYSIS")
    print("=" * 70)

    for phase_name in ["POWERPLAY", "MIDDLE", "DEATH"]:
        phase_results = [r for r in results if r.phase == phase_name]
        if not phase_results:
            print(f"\n  {phase_name}: no events")
            continue

        print(f"\n  {phase_name} ({len(phase_results)} events):")

        best_ht = None
        for ht in HOLD_TIMES:
            stats = simulate_concurrent(phase_results, ht)
            if stats["trades"] > 0:
                print(f"    hold={ht:>3d}s: pnl={stats['total_pnl']:+.4f} | "
                      f"wr={stats['win_rate']*100:.0f}% | n={stats['trades']}")
                if best_ht is None or stats["total_pnl"] > best_ht["total_pnl"]:
                    best_ht = stats
        if best_ht:
            print(f"    ★ BEST: hold={best_ht['hold_time']}s → R={best_ht['total_pnl']:+.4f}")

    # ═════════════════════════════════════════════════════════════════
    # 7. ANALYSIS: By Event Type
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("4. BY EVENT TYPE — Which events produce alpha?")
    print("=" * 70)

    for evt_type in ["WICKET", "BOUNDARY", "SURGE"]:
        type_results = [r for r in results if r.event.event_type == evt_type]
        if not type_results:
            continue

        print(f"\n  {evt_type} ({len(type_results)} events):")

        # MFE stats
        mfes = [r.mfe for r in type_results]
        maes = [r.mae for r in type_results]
        avg_mfe = sum(mfes) / len(mfes)
        avg_mae = sum(maes) / len(maes)
        avg_mfe_time = sum(r.mfe_time for r in type_results) / len(type_results)

        print(f"    avg_MFE={avg_mfe:+.4f} | avg_MAE={avg_mae:+.4f} | "
              f"avg_time_to_MFE={avg_mfe_time:.0f}s")

        best_ht = None
        for ht in [30, 60, 120, 180, 300]:
            stats = simulate_concurrent(type_results, ht)
            if stats["trades"] > 0:
                print(f"    hold={ht:>3d}s: pnl={stats['total_pnl']:+.4f} | "
                      f"wr={stats['win_rate']*100:.0f}% | n={stats['trades']}")
                if best_ht is None or stats["total_pnl"] > best_ht["total_pnl"]:
                    best_ht = stats

        if best_ht:
            print(f"    ★ BEST: hold={best_ht['hold_time']}s → R={best_ht['total_pnl']:+.4f}")

    # ═════════════════════════════════════════════════════════════════
    # 8. MFE / MAE Summary
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("5. MFE / MAE — How much alpha is available?")
    print("=" * 70)

    all_mfe = [r.mfe for r in results]
    all_mae = [r.mae for r in results]
    all_mfe_t = [r.mfe_time for r in results]

    print(f"  Avg MFE: {sum(all_mfe)/len(all_mfe):+.4f} "
          f"(max {max(all_mfe):+.4f})")
    print(f"  Avg MAE: {sum(all_mae)/len(all_mae):+.4f} "
          f"(max {min(all_mae):+.4f})")
    print(f"  Avg time to MFE: {sum(all_mfe_t)/len(all_mfe_t):.0f}s "
          f"(max {max(all_mfe_t):.0f}s)")
    print(f"  MFE > 0.05: {sum(1 for m in all_mfe if m > 0.05)}/{len(all_mfe)} "
          f"({sum(1 for m in all_mfe if m > 0.05)/len(all_mfe)*100:.0f}%)")
    print(f"  MFE > 0.10: {sum(1 for m in all_mfe if m > 0.10)}/{len(all_mfe)} "
          f"({sum(1 for m in all_mfe if m > 0.10)/len(all_mfe)*100:.0f}%)")

    # ═════════════════════════════════════════════════════════════════
    # 9. MAX ALPHA CONFIG
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("6. MAX ALPHA CONFIGURATION — Best combo of phase + type + hold")
    print("=" * 70)

    best_combo = None
    for phase_name in ["POWERPLAY", "MIDDLE", "DEATH", "ALL"]:
        for evt_type in ["WICKET", "BOUNDARY", "SURGE", "ALL"]:
            filtered = results
            if phase_name != "ALL":
                filtered = [r for r in filtered if r.phase == phase_name]
            if evt_type != "ALL":
                filtered = [r for r in filtered if r.event.event_type == evt_type]

            if len(filtered) < 3:
                continue

            for ht in HOLD_TIMES:
                stats = simulate_concurrent(filtered, ht)
                if stats["trades"] >= 3:
                    combo = {
                        "phase": phase_name,
                        "event": evt_type,
                        "hold": ht,
                        **stats,
                    }
                    if best_combo is None or stats["avg_pnl"] > best_combo["avg_pnl"]:
                        best_combo = combo

    if best_combo:
        print(f"  ★ BEST COMBO:")
        print(f"    Phase:     {best_combo['phase']}")
        print(f"    Event:     {best_combo['event']}")
        print(f"    Hold:      {best_combo['hold']}s")
        print(f"    Trades:    {best_combo['trades']}")
        print(f"    Total R:   {best_combo['total_pnl']:+.4f}")
        print(f"    Avg R:     {best_combo['avg_pnl']:+.4f}")
        print(f"    Win Rate:  {best_combo['win_rate']*100:.0f}%")

    # ═════════════════════════════════════════════════════════════════
    # 10. Trade-by-trade detail
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("7. TRADE-BY-TRADE DETAIL (top 20 by MFE)")
    print("=" * 70)
    print(f"{'#':>3} {'Time':>8} {'Fix':>6} {'Type':>8} {'Dir':>5} {'Entry':>6} "
          f"{'MFE':>7} {'MAE':>7} {'Phase':>10} {'Best@':>6}")

    sorted_results = sorted(results, key=lambda r: -r.mfe)
    for i, r in enumerate(sorted_results[:20]):
        print(f"{i+1:>3} {r.event.log_time:>8} {r.event.fixture_id:>6} "
              f"{r.event.event_type:>8} {r.direction:>5} {r.entry_mid:>6.3f} "
              f"{r.mfe:>+7.4f} {r.mae:>+7.4f} {r.phase:>10} "
              f"{r.best_hold:>4}s")

    conn.close()
    print("\n" + "=" * 70)
    print("REPLAY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
