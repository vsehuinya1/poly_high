"""
Tick Recorder — records every Polymarket price tick to SQLite for offline replay.

Hooks into the existing PolymarketFeed to capture every BBO update.
Data stored in sports_data/tick_history.db with full bid/ask/mid/spread.

Usage in main.py:
    from sports.tick_recorder import TickRecorder
    recorder = TickRecorder()
    # After engine.tg is set up:
    poly_feed._tick_recorder = recorder
    # The recorder hooks into BookState updates automatically

Standalone query:
    python -m sports.tick_recorder --token TOKEN_ID --last 100
    python -m sports.tick_recorder --match "Lakers" --date 20260319
"""
import csv
import logging
import os
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("sports.tick_recorder")

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sports_data", "tick_history.db"
)


class TickRecorder:
    """Records every BBO update to SQLite for offline analysis and replay."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._setup_tables()
        self._tick_count = 0
        self._last_log = 0
        # Token → market title mapping (set externally)
        self.token_labels: dict[str, str] = {}
        log.info("TICK_RECORDER: initialized → %s", db_path)

    def _setup_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                token_id TEXT NOT NULL,
                best_bid REAL,
                best_ask REAL,
                mid REAL,
                spread REAL,
                source TEXT DEFAULT 'ws'
            );
            CREATE INDEX IF NOT EXISTS idx_ticks_token ON ticks(token_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_ticks_time ON ticks(timestamp);

            CREATE TABLE IF NOT EXISTS token_labels (
                token_id TEXT PRIMARY KEY,
                market_title TEXT,
                sport TEXT,
                game_id TEXT,
                updated_at REAL
            );
        """)
        self._conn.commit()

    def record_tick(self, token_id: str, best_bid: float, best_ask: float,
                    mid: float, spread: float, source: str = "ws"):
        """Record a single BBO update."""
        now = time.time()
        try:
            self._conn.execute(
                "INSERT INTO ticks (timestamp, token_id, best_bid, best_ask, mid, spread, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now, token_id, best_bid, best_ask, mid, spread, source)
            )
            self._tick_count += 1
            # Batch commit every 100 ticks
            if self._tick_count % 100 == 0:
                self._conn.commit()
            # Log every 60s
            if now - self._last_log > 60:
                self._last_log = now
                db_size = os.path.getsize(self.db_path) / 1024 / 1024
                log.info("TICK_RECORDER: %d ticks recorded | db=%.1fMB",
                         self._tick_count, db_size)
        except Exception as e:
            log.error("TICK_RECORDER: error: %s", e)

    def set_label(self, token_id: str, market_title: str, sport: str = "",
                  game_id: str = ""):
        """Associate a token ID with a human-readable label."""
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO token_labels (token_id, market_title, sport, game_id, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (token_id, market_title, sport, game_id, time.time())
            )
            self._conn.commit()
            self.token_labels[token_id] = market_title
        except Exception as e:
            log.error("TICK_RECORDER: label error: %s", e)

    def get_ticks(self, token_id: str, since: float = 0, limit: int = 1000) -> list[dict]:
        """Query ticks for a token."""
        cursor = self._conn.execute(
            "SELECT timestamp, best_bid, best_ask, mid, spread, source "
            "FROM ticks WHERE token_id = ? AND timestamp > ? ORDER BY timestamp DESC LIMIT ?",
            (token_id, since, limit)
        )
        return [{"ts": r[0], "bid": r[1], "ask": r[2], "mid": r[3],
                 "spread": r[4], "source": r[5]} for r in cursor]

    def get_price_path(self, token_id: str, start_ts: float, end_ts: float) -> list[tuple]:
        """Get (timestamp, mid) pairs for offline replay."""
        cursor = self._conn.execute(
            "SELECT timestamp, mid FROM ticks WHERE token_id = ? "
            "AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
            (token_id, start_ts, end_ts)
        )
        return list(cursor)

    def flush(self):
        """Force commit pending writes."""
        self._conn.commit()

    def close(self):
        """Close the database."""
        self._conn.commit()
        self._conn.close()

    def stats(self) -> dict:
        """Get recorder stats."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM ticks")
        total = cursor.fetchone()[0]
        cursor = self._conn.execute("SELECT COUNT(DISTINCT token_id) FROM ticks")
        tokens = cursor.fetchone()[0]
        db_size = os.path.getsize(self.db_path) / 1024 / 1024
        return {"total_ticks": total, "tokens": tokens, "db_size_mb": round(db_size, 1)}


# ── CLI for querying recorded data ──────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query tick history")
    parser.add_argument("--token", help="Token ID to query")
    parser.add_argument("--match", help="Search by market title substring")
    parser.add_argument("--last", type=int, default=20, help="Number of ticks")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("--export", help="Export token ticks to CSV file")
    args = parser.parse_args()

    rec = TickRecorder()

    if args.stats:
        s = rec.stats()
        print(f"Total ticks: {s['total_ticks']:,}")
        print(f"Unique tokens: {s['tokens']}")
        print(f"DB size: {s['db_size_mb']}MB")

    elif args.match:
        cursor = rec._conn.execute(
            "SELECT token_id, market_title, sport FROM token_labels WHERE market_title LIKE ?",
            (f"%{args.match}%",)
        )
        for tid, title, sport in cursor:
            ticks = rec.get_ticks(tid, limit=1)
            last = ticks[0] if ticks else {}
            print(f"  {title} [{sport}] | last mid={last.get('mid', '?'):.4f}"
                  if last else f"  {title} [{sport}] | no ticks")

    elif args.token:
        ticks = rec.get_ticks(args.token, limit=args.last)
        if args.export:
            with open(args.export, "w") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "bid", "ask", "mid", "spread"])
                for t in reversed(ticks):
                    w.writerow([t["ts"], t["bid"], t["ask"], t["mid"], t["spread"]])
            print(f"Exported {len(ticks)} ticks to {args.export}")
        else:
            for t in reversed(ticks):
                ts = time.strftime("%H:%M:%S", time.localtime(t["ts"]))
                print(f"  {ts} | bid={t['bid']:.4f} ask={t['ask']:.4f} "
                      f"mid={t['mid']:.4f} spread={t['spread']:.4f}")
    else:
        s = rec.stats()
        print(f"Tick DB: {s['total_ticks']:,} ticks, {s['tokens']} tokens, {s['db_size_mb']}MB")
        print("Use --help for query options")
