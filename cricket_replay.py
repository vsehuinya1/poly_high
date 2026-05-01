"""
Cricket Replay Engine v2 — Tick-Level Alpha Measurement.

Uses the paper_trades CSV (with exact unix timestamps) and builds
direct fixture→token mapping from token_labels.

Usage: python3 cricket_replay.py
"""

import sqlite3
import csv
import re
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass

TICK_DB = "sports_data/tick_history.db"
PAPER_CSV = "sports_data/cricket_paper_trades.csv"
LOG_FILES = [
    "logs/cricket_v11_5.log",
    "logs/cricket_v11_4_RELOAD.log",
]

SNAPSHOTS = [5, 10, 15, 30, 60, 120, 180, 300]
HOLD_TIMES = [15, 30, 60, 90, 120, 180, 240, 300]
PHASES = {"POWERPLAY": (0, 6), "MIDDLE": (6, 15), "DEATH": (15, 20)}

# ── Fixture → Token Mapping ────────────────────────────────────────
# Manually derived from token_labels + paper_trades
FIXTURE_TOKEN_MAP = {}  # populated at runtime


def build_fixture_token_map(conn):
    """Map market titles to token pairs from token_labels."""
    cur = conn.cursor()
    cur.execute("""
        SELECT token_id, market_title FROM token_labels 
        WHERE sport='cricket' AND market_title LIKE '%Indian Premier League%'
    """)
    rows = cur.fetchall()
    
    # Group by market_title → list of token_ids
    by_title = defaultdict(list)
    for tid, title in rows:
        by_title[title].append(tid)
    
    print(f"  Found {len(by_title)} IPL market titles with {sum(len(v) for v in by_title.values())} tokens")
    
    # For each title, check which token has mid > 0.10 (home team token)
    result = {}
    for title, tids in by_title.items():
        if len(tids) < 2:
            continue
        
        # Get latest mid price for each token
        token_mids = []
        for tid in tids:
            cur.execute("SELECT mid FROM ticks WHERE token_id = ? ORDER BY timestamp DESC LIMIT 1", (tid,))
            row = cur.fetchone()
            if row:
                token_mids.append((tid, row[0]))
        
        if len(token_mids) >= 2:
            # Sort by mid descending — the one with higher mid is typically the "home" token
            token_mids.sort(key=lambda x: -x[1])
            result[title] = {
                "home_token": token_mids[0][0],
                "away_token": token_mids[1][0],
                "home_mid": token_mids[0][1],
                "away_mid": token_mids[1][1],
            }
    
    return result


@dataclass
class CricketEvent:
    timestamp: float
    fixture_id: int
    event_type: str
    overs: float
    total_runs: int
    total_wickets: int
    direction: str  # LONG or SHORT
    phase: str

@dataclass 
class ReplayResult:
    event: CricketEvent
    entry_mid: float
    forward_prices: dict  # {seconds: mid}
    forward_pnl: dict     # {seconds: pnl}
    mfe: float = 0.0
    mae: float = 0.0
    mfe_time: float = 0.0


def parse_events_from_logs():
    """Parse all cricket events from log files with proper timestamps."""
    events = []
    
    pattern = re.compile(
        r"CRICKET_EVENT_DETECTED \| "
        r"fixture=(\d+) \| event=(\w+) \| "
        r"runs_delta=(\d+) wickets_delta=(\d+) \| "
        r"(\d+)/(\d+) \(([\d.]+) ov\) \| "
        r"RR=([\d.]+)"
    )
    
    for log_path in LOG_FILES:
        if not os.path.exists(log_path):
            continue
        
        # Determine the date from the log filename or first dated line
        log_date = None
        with open(log_path, "r") as f:
            for line in f:
                m = re.match(r"(\d{4}-\d{2}-\d{2})", line)
                if m:
                    log_date = m.group(1)
                    break
        
        if not log_date:
            # Try file modification time
            mtime = os.path.getmtime(log_path)
            log_date = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")
        
        current_date = log_date
        
        with open(log_path, "r") as f:
            for line in f:
                # Update date if line has one
                date_m = re.match(r"(\d{4}-\d{2}-\d{2})", line)
                if date_m:
                    current_date = date_m.group(1)
                
                m = pattern.search(line)
                if not m:
                    continue
                
                # Extract time from line
                time_m = re.match(r"(?:\d{4}-\d{2}-\d{2}\s+)?(\d{2}:\d{2}:\d{2})", line)
                if not time_m:
                    continue
                
                time_str = time_m.group(1)
                dt = datetime.strptime(f"{current_date} {time_str}", "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                ts = dt.timestamp()
                
                fixture_id = int(m.group(1))
                event_type = m.group(2)
                overs = float(m.group(7))
                total_runs = int(m.group(5))
                total_wickets = int(m.group(6))
                
                if event_type == "DOT":
                    continue
                
                if event_type == "WICKET":
                    direction = "SHORT"
                elif event_type in ("BOUNDARY", "SURGE"):
                    direction = "LONG"
                else:
                    continue
                
                phase = "POWERPLAY" if overs < 6 else "DEATH" if overs >= 15 else "MIDDLE"
                
                events.append(CricketEvent(
                    timestamp=ts,
                    fixture_id=fixture_id,
                    event_type=event_type,
                    overs=overs,
                    total_runs=total_runs,
                    total_wickets=total_wickets,
                    direction=direction,
                    phase=phase,
                ))
    
    # Dedup
    seen = set()
    unique = []
    for e in events:
        key = (e.fixture_id, round(e.timestamp))
        if key not in seen:
            seen.add(key)
            unique.append(e)
    
    unique.sort(key=lambda e: e.timestamp)
    return unique


def find_token_for_event(conn, event, title_map):
    """Find the correct home-team token for an event's fixture."""
    cur = conn.cursor()
    
    # Get all IPL tokens that have ticks within 15 minutes of the event
    cur.execute("""
        SELECT DISTINCT t.token_id, tl.market_title
        FROM ticks t
        JOIN token_labels tl ON t.token_id = tl.token_id
        WHERE tl.sport = 'cricket' 
          AND tl.market_title LIKE '%Indian Premier League%'
          AND t.timestamp BETWEEN ? AND ?
        LIMIT 20
    """, (event.timestamp - 60, event.timestamp + 60))
    
    active = cur.fetchall()
    if not active:
        return None, None
    
    # Find the token with a competitive mid price
    best = None
    for tid, title in active:
        cur.execute("""
            SELECT mid FROM ticks 
            WHERE token_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY ABS(timestamp - ?) LIMIT 1
        """, (tid, event.timestamp - 10, event.timestamp + 10, event.timestamp))
        row = cur.fetchone()
        if row and row[0] and 0.05 < row[0] < 0.95:
            if best is None or abs(row[0] - 0.5) < abs(best[2] - 0.5):
                best = (tid, title, row[0])
    
    if best:
        return best[0], best[1]
    
    # Fallback: return first active token
    return active[0][0], active[0][1]


def get_forward_prices(conn, token_id, event_ts):
    """Get price snapshots at each forward interval."""
    cur = conn.cursor()
    result = {}
    for s in SNAPSHOTS:
        cur.execute("""
            SELECT mid FROM ticks 
            WHERE token_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY ABS(timestamp - ?) LIMIT 1
        """, (token_id, event_ts + s - 10, event_ts + s + 10, event_ts + s))
        row = cur.fetchone()
        if row and row[0]:
            result[s] = row[0]
    return result


def get_price_series(conn, token_id, start_ts, duration=310.0):
    """Get all ticks for a token in a time window."""
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp - ?, mid FROM ticks 
        WHERE token_id = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """, (start_ts, token_id, start_ts, start_ts + duration))
    return cur.fetchall()


def replay_all(events, conn):
    """Replay all events against tick data."""
    results = []
    title_map = build_fixture_token_map(conn)
    
    # Cache fixture → token
    fixture_tokens = {}
    
    for i, event in enumerate(events):
        # Find token
        if event.fixture_id not in fixture_tokens:
            token, title = find_token_for_event(conn, event, title_map)
            if token:
                fixture_tokens[event.fixture_id] = (token, title)
                print(f"  Fixture {event.fixture_id} → {title[:50]}")
            else:
                continue
        
        token, title = fixture_tokens[event.fixture_id]
        
        # Entry price
        cur = conn.cursor()
        cur.execute("""
            SELECT mid FROM ticks 
            WHERE token_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY ABS(timestamp - ?) LIMIT 1
        """, (token, event.timestamp - 10, event.timestamp + 10, event.timestamp))
        row = cur.fetchone()
        if not row or not row[0]:
            continue
        entry_mid = row[0]
        
        # Forward prices
        fwd = get_forward_prices(conn, token, event.timestamp)
        if not fwd:
            continue
        
        # Forward PnL
        fwd_pnl = {}
        for s, price in fwd.items():
            if event.direction == "LONG":
                fwd_pnl[s] = price - entry_mid
            else:
                fwd_pnl[s] = entry_mid - price
        
        # MFE/MAE from full series
        series = get_price_series(conn, token, event.timestamp, 310.0)
        mfe = 0.0
        mae = 0.0
        mfe_time = 0.0
        for dt, price in series:
            if event.direction == "LONG":
                move = price - entry_mid
            else:
                move = entry_mid - price
            if move > mfe:
                mfe = move
                mfe_time = dt
            if move < mae:
                mae = move
        
        results.append(ReplayResult(
            event=event,
            entry_mid=entry_mid,
            forward_prices=fwd,
            forward_pnl=fwd_pnl,
            mfe=mfe,
            mae=mae,
            mfe_time=mfe_time,
        ))
    
    return results


def print_report(results):
    """Print full analysis report."""
    if not results:
        print("NO RESULTS — no events matched tick data")
        return
    
    # ═══ 1. LATENCY ADVANTAGE ═══
    print("\n" + "=" * 70)
    print("1. LATENCY ADVANTAGE — Forward Price Curves")
    print("=" * 70)
    for s in SNAPSHOTS:
        pnls = [r.forward_pnl[s] for r in results if s in r.forward_pnl]
        if pnls:
            avg = sum(pnls) / len(pnls)
            pos = sum(1 for p in pnls if p > 0)
            total = sum(pnls)
            print(f"  t+{s:>3d}s: avg_R={avg:+.4f} | wr={pos}/{len(pnls)} ({pos/len(pnls)*100:.0f}%) | total_R={total:+.4f}")

    # ═══ 2. OPTIMAL HOLD TIME ═══
    print("\n" + "=" * 70)
    print("2. OPTIMAL HOLD TIME — Concurrent Positions")
    print("=" * 70)
    best = None
    for ht in HOLD_TIMES:
        pnls = [r.forward_pnl[ht] for r in results if ht in r.forward_pnl]
        if pnls:
            total = sum(pnls)
            avg = total / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            print(f"  hold={ht:>3d}s: n={len(pnls):>3} | total_R={total:+.4f} | avg_R={avg:+.4f} | wr={wr*100:.0f}%")
            if best is None or total > best[0]:
                best = (total, ht, len(pnls), wr)
    if best:
        print(f"\n  ★ BEST: hold={best[1]}s → total_R={best[0]:+.4f} over {best[2]} trades (wr={best[3]*100:.0f}%)")

    # ═══ 3. MATCH PHASE ═══
    print("\n" + "=" * 70)
    print("3. MATCH PHASE ANALYSIS")
    print("=" * 70)
    for phase in ["POWERPLAY", "MIDDLE", "DEATH"]:
        pr = [r for r in results if r.event.phase == phase]
        if not pr:
            print(f"\n  {phase}: no data")
            continue
        print(f"\n  {phase} ({len(pr)} events):")
        best_ph = None
        for ht in HOLD_TIMES:
            pnls = [r.forward_pnl[ht] for r in pr if ht in r.forward_pnl]
            if pnls:
                total = sum(pnls)
                avg = total / len(pnls)
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                print(f"    hold={ht:>3d}s: n={len(pnls):>2} | R={total:+.4f} | avg={avg:+.4f} | wr={wr*100:.0f}%")
                if best_ph is None or total > best_ph[0]:
                    best_ph = (total, ht, len(pnls))
        if best_ph:
            print(f"    ★ BEST: hold={best_ph[1]}s → R={best_ph[0]:+.4f}")

    # ═══ 4. BY EVENT TYPE ═══
    print("\n" + "=" * 70)
    print("4. BY EVENT TYPE")
    print("=" * 70)
    for evt in ["WICKET", "BOUNDARY", "SURGE"]:
        er = [r for r in results if r.event.event_type == evt]
        if not er:
            continue
        mfes = [r.mfe for r in er]
        maes = [r.mae for r in er]
        print(f"\n  {evt} ({len(er)} events):")
        print(f"    MFE: avg={sum(mfes)/len(mfes):+.4f} max={max(mfes):+.4f}")
        print(f"    MAE: avg={sum(maes)/len(maes):+.4f} max={min(maes):+.4f}")
        print(f"    Avg time to MFE: {sum(r.mfe_time for r in er)/len(er):.0f}s")
        best_et = None
        for ht in [30, 60, 120, 300]:
            pnls = [r.forward_pnl[ht] for r in er if ht in r.forward_pnl]
            if pnls:
                total = sum(pnls)
                avg = total / len(pnls)
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                print(f"    hold={ht:>3d}s: R={total:+.4f} | avg={avg:+.4f} | wr={wr*100:.0f}%")
                if best_et is None or total > best_et[0]:
                    best_et = (total, ht)
        if best_et:
            print(f"    ★ BEST: hold={best_et[1]}s → R={best_et[0]:+.4f}")

    # ═══ 5. MFE / MAE SUMMARY ═══
    print("\n" + "=" * 70)
    print("5. MFE / MAE SUMMARY")
    print("=" * 70)
    all_mfe = [r.mfe for r in results]
    all_mae = [r.mae for r in results]
    print(f"  Avg MFE: {sum(all_mfe)/len(all_mfe):+.4f} (max {max(all_mfe):+.4f})")
    print(f"  Avg MAE: {sum(all_mae)/len(all_mae):+.4f} (worst {min(all_mae):+.4f})")
    print(f"  Avg time to MFE: {sum(r.mfe_time for r in results)/len(results):.0f}s")
    print(f"  MFE > 0.02: {sum(1 for m in all_mfe if m > 0.02)}/{len(all_mfe)}")
    print(f"  MFE > 0.05: {sum(1 for m in all_mfe if m > 0.05)}/{len(all_mfe)}")
    print(f"  MFE > 0.10: {sum(1 for m in all_mfe if m > 0.10)}/{len(all_mfe)}")

    # ═══ 6. MAX ALPHA CONFIG ═══
    print("\n" + "=" * 70)
    print("6. MAX ALPHA CONFIG")
    print("=" * 70)
    best_combo = None
    for phase in ["POWERPLAY", "MIDDLE", "DEATH", "ALL"]:
        for evt in ["WICKET", "BOUNDARY", "SURGE", "ALL"]:
            filtered = results
            if phase != "ALL":
                filtered = [r for r in filtered if r.event.phase == phase]
            if evt != "ALL":
                filtered = [r for r in filtered if r.event.event_type == evt]
            if len(filtered) < 2:
                continue
            for ht in HOLD_TIMES:
                pnls = [r.forward_pnl[ht] for r in filtered if ht in r.forward_pnl]
                if len(pnls) >= 2:
                    total = sum(pnls)
                    avg = total / len(pnls)
                    wr = sum(1 for p in pnls if p > 0) / len(pnls)
                    if best_combo is None or avg > best_combo["avg"]:
                        best_combo = {"phase": phase, "event": evt, "hold": ht,
                                      "n": len(pnls), "total": total, "avg": avg, "wr": wr}
    if best_combo:
        print(f"  ★ Phase:    {best_combo['phase']}")
        print(f"  ★ Event:    {best_combo['event']}")
        print(f"  ★ Hold:     {best_combo['hold']}s")
        print(f"  ★ Trades:   {best_combo['n']}")
        print(f"  ★ Total R:  {best_combo['total']:+.4f}")
        print(f"  ★ Avg R:    {best_combo['avg']:+.4f}")
        print(f"  ★ Win Rate: {best_combo['wr']*100:.0f}%")

    # ═══ 7. TRADE BY TRADE ═══
    print("\n" + "=" * 70)
    print("7. TRADE DETAIL (sorted by MFE)")
    print("=" * 70)
    print(f"{'#':>3} {'Fix':>6} {'Type':>8} {'Dir':>5} {'Overs':>5} {'Phase':>10} "
          f"{'Entry':>6} {'MFE':>7} {'MAE':>7} {'t_MFE':>5}")
    for i, r in enumerate(sorted(results, key=lambda x: -x.mfe)):
        print(f"{i+1:>3} {r.event.fixture_id:>6} {r.event.event_type:>8} "
              f"{r.event.direction:>5} {r.event.overs:>5.1f} {r.event.phase:>10} "
              f"{r.entry_mid:>6.3f} {r.mfe:>+7.4f} {r.mae:>+7.4f} {r.mfe_time:>4.0f}s")


def main():
    print("=" * 70)
    print("CRICKET REPLAY ENGINE v2 — Max Alpha Analysis")
    print("=" * 70)

    # Parse events
    events = parse_events_from_logs()
    print(f"Parsed {len(events)} actionable events from logs")
    
    by_type = defaultdict(int)
    by_fixture = defaultdict(int)
    for e in events:
        by_type[e.event_type] += 1
        by_fixture[e.fixture_id] += 1
    print(f"  By type: {dict(by_type)}")
    print(f"  By fixture: {dict(by_fixture)}")

    # Connect to tick DB
    conn = sqlite3.connect(TICK_DB)
    
    # Create index for faster queries if not exists
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_token_ts ON ticks(token_id, timestamp)")
        conn.commit()
        print("  Index created/verified")
    except:
        pass

    # Replay
    print("\nReplaying events against tick data...")
    results = replay_all(events, conn)
    print(f"Matched {len(results)}/{len(events)} events to tick data")

    # Report
    print_report(results)

    conn.close()
    print("\n" + "=" * 70)
    print("REPLAY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
