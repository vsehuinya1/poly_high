"""
Tennis Lower-Band Shadow Trade Tracker v11.4

Compressed signal pipeline for learning mode.
Filters raw signals (~50k/day) to ~200-500 quality observations.

Filters:
  1. Dedup: 1 signal per (match_id, player) per 60s
  2. Liquidity: book update <= 120s ago
  3. Score context: player losing, opponent games >= 3
  4. Forward prices: +30s, +120s, +300s snapshots
  5. Output: sports_data/tennis_low_band_compressed_YYYYMMDD.csv
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("sports.tennis_lb_shadow")

LB_DEDUP_S = 60.0
LB_STALE_BOOK_S = 120.0


@dataclass
class ShadowTrade:
    trade_id: str
    timestamp: float
    tournament: str
    match_id: str
    player: str
    score: str
    entry_price: float
    edge: float
    token_id: str
    price_t30: float = 0.0
    price_t120: float = 0.0
    price_t300: float = 0.0
    max_price_300s: float = 0.0
    min_price_300s: float = 999.0
    time_to_first_move: float = -1.0
    _snap_done: set = field(default_factory=set)
    _written: bool = False


class TennisLBShadow:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._last_signal: dict[tuple, float] = {}
        self._active: dict[str, ShadowTrade] = {}

    def should_track(self, match_id, player, book_update_ts,
                     sets_a, sets_b, games_a, games_b):
        now = time.time()
        key = (match_id, player)
        if now - self._last_signal.get(key, 0) < LB_DEDUP_S:
            return False, "DEDUP"
        if book_update_ts > 0 and (now - book_update_ts) > LB_STALE_BOOK_S:
            log.info("TENNIS_LOW_BAND_SKIPPED | reason=STALE_BOOK | price_age=%.0f",
                     now - book_update_ts)
            return False, "STALE_BOOK"
        if games_b < 3 or games_a >= games_b:
            log.info("TENNIS_LOW_BAND_SKIPPED | reason=EARLY_SCORE | %d-%d %d-%d",
                     sets_a, sets_b, games_a, games_b)
            return False, "EARLY_SCORE"
        return True, "OK"

    def register(self, match_id, player, market_price, edge,
                 tournament, score, token_id):
        now = time.time()
        self._last_signal[(match_id, player)] = now
        tid = f"LB_{match_id}_{int(now)}"
        t = ShadowTrade(
            trade_id=tid, timestamp=now, tournament=tournament,
            match_id=match_id, player=player, score=score,
            entry_price=market_price, edge=edge, token_id=token_id,
            max_price_300s=market_price, min_price_300s=market_price,
        )
        self._active[tid] = t
        log.info("TENNIS_LOW_BAND_VALID | %s | mkt=%.4f | edge=%.4f | score=%s",
                 player, market_price, edge, score)
        return t

    def tick(self, books):
        now = time.time()
        done = []
        for tid, t in list(self._active.items()):
            book = books.get(t.token_id)
            if not book or book.mid <= 0:
                continue
            mkt = book.mid
            elapsed = now - t.timestamp
            if mkt > t.max_price_300s:
                t.max_price_300s = mkt
            if mkt < t.min_price_300s:
                t.min_price_300s = mkt
            if t.time_to_first_move < 0 and abs(mkt - t.entry_price) > 0.01:
                t.time_to_first_move = elapsed
            if elapsed >= 30 and 30 not in t._snap_done:
                t.price_t30 = mkt; t._snap_done.add(30)
            if elapsed >= 120 and 120 not in t._snap_done:
                t.price_t120 = mkt; t._snap_done.add(120)
            if elapsed >= 300 and 300 not in t._snap_done:
                t.price_t300 = mkt; t._snap_done.add(300)
                done.append(tid)
        for tid in done:
            t = self._active.pop(tid, None)
            if t and not t._written:
                self._write(t)

    def _write(self, t):
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        fp = self.data_dir / f"tennis_low_band_compressed_{today}.csv"
        hdr = not fp.exists()
        e = t.entry_price
        r_max = (t.max_price_300s - e) / e if e > 0.001 else 0
        r_min = (t.min_price_300s - e) / e if e > 0.001 else 0
        row = {
            "trade_id": t.trade_id, "timestamp": f"{t.timestamp:.3f}",
            "tournament": t.tournament, "match_id": t.match_id,
            "player": t.player, "score": t.score,
            "entry_price": f"{e:.4f}", "edge": f"{t.edge:.4f}",
            "price_t30": f"{t.price_t30:.4f}",
            "price_t120": f"{t.price_t120:.4f}",
            "price_t300": f"{t.price_t300:.4f}",
            "max_price_300s": f"{t.max_price_300s:.4f}",
            "min_price_300s": f"{t.min_price_300s:.4f}",
            "R_max_300s": f"{r_max:+.4f}", "R_min_300s": f"{r_min:+.4f}",
            "hit_2R": str(t.max_price_300s >= e * 2),
            "hit_3R": str(t.max_price_300s >= e * 3),
            "dead_market": str((t.max_price_300s - e) < 0.01),
            "time_to_first_move": f"{t.time_to_first_move:.1f}" if t.time_to_first_move >= 0 else "-1",
        }
        try:
            with open(fp, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if hdr:
                    w.writeheader()
                w.writerow(row)
        except Exception as ex:
            log.warning("TENNIS_LB_CSV_ERR | %s", ex)
        t._written = True
