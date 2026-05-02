"""
Tennis Lower-Band Shadow Trade Tracker v11.4B

Compressed signal pipeline with measurement completeness patches.
Loosened score filter, funnel counters, time_to_2R/3R, spread capture.
"""
from __future__ import annotations

import csv
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("sports.tennis_lb_shadow")

LB_DEDUP_S = 60.0
LB_STALE_BOOK_S = 120.0
ENGINE_VERSION = "v11.4B_shadow"


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
    spread: float = 0.0
    price_t30: float = 0.0
    price_t120: float = 0.0
    price_t300: float = 0.0
    max_price_300s: float = 0.0
    min_price_300s: float = 999.0
    time_to_first_move: float = -1.0
    time_to_2R: float = -1.0
    time_to_3R: float = -1.0
    match_signal_count: int = 0
    _snap_done: set = field(default_factory=set)
    _written: bool = False


class TennisLBShadow:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._last_signal: dict[tuple, float] = {}
        self._active: dict[str, ShadowTrade] = {}
        self._match_counts: dict[str, int] = defaultdict(int)
        self._funnel = {"raw": 0, "dedup": 0, "liquidity": 0, "score": 0, "final": 0}
        self._last_funnel_log = time.time()

    def should_track(self, match_id, player, book_update_ts,
                     sets_a, sets_b, games_a, games_b,
                     spread: float = 0.0):
        now = time.time()
        key = (match_id, player)
        self._funnel["raw"] += 1

        if now - self._last_signal.get(key, 0) < LB_DEDUP_S:
            return False, "DEDUP"
        self._funnel["dedup"] += 1

        if book_update_ts > 0 and (now - book_update_ts) > LB_STALE_BOOK_S:
            return False, "STALE_BOOK"
        self._funnel["liquidity"] += 1

        # v11.4B: loosened — opponent games >= 2, allow tied
        if games_b < 2:
            return False, "EARLY_SCORE"
        if games_a > games_b + 1:
            return False, "PLAYER_AHEAD"
        self._funnel["score"] += 1
        self._funnel["final"] += 1

        # Hourly funnel log
        if now - self._last_funnel_log >= 3600:
            log.info("TENNIS_LOW_BAND_FUNNEL | raw=%d dedup=%d liq=%d score=%d final=%d",
                     self._funnel["raw"], self._funnel["dedup"],
                     self._funnel["liquidity"], self._funnel["score"],
                     self._funnel["final"])
            self._funnel = {"raw": 0, "dedup": 0, "liquidity": 0, "score": 0, "final": 0}
            self._last_funnel_log = now

        return True, "OK"

    def register(self, match_id, player, market_price, edge,
                 tournament, score, token_id, spread: float = 0.0):
        now = time.time()
        self._last_signal[(match_id, player)] = now
        self._match_counts[str(match_id)] += 1
        tid = f"LB_{match_id}_{int(now)}"
        t = ShadowTrade(
            trade_id=tid, timestamp=now, tournament=tournament,
            match_id=match_id, player=player, score=score,
            entry_price=market_price, edge=edge, token_id=token_id,
            spread=spread,
            max_price_300s=market_price, min_price_300s=market_price,
            match_signal_count=self._match_counts[str(match_id)],
        )
        self._active[tid] = t
        log.info("TENNIS_LOW_BAND_VALID | %s | mkt=%.4f | edge=%.4f | score=%s | sp=%.4f",
                 player, market_price, edge, score, spread)
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
            if t.time_to_2R < 0 and t.entry_price > 0 and mkt >= t.entry_price * 2:
                t.time_to_2R = elapsed
            if t.time_to_3R < 0 and t.entry_price > 0 and mkt >= t.entry_price * 3:
                t.time_to_3R = elapsed
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
        dead = t.max_price_300s < e * 1.10  # v11.4B: normalized
        row = {
            "trade_id": t.trade_id, "timestamp": f"{t.timestamp:.3f}",
            "engine_version": ENGINE_VERSION,
            "tournament": t.tournament, "match_id": t.match_id,
            "player": t.player, "score": t.score,
            "entry_price": f"{e:.4f}", "edge": f"{t.edge:.4f}",
            "spread": f"{t.spread:.4f}",
            "price_t30": f"{t.price_t30:.4f}", "price_t120": f"{t.price_t120:.4f}",
            "price_t300": f"{t.price_t300:.4f}",
            "max_price_300s": f"{t.max_price_300s:.4f}",
            "min_price_300s": f"{t.min_price_300s:.4f}",
            "R_max_300s": f"{r_max:+.4f}", "R_min_300s": f"{r_min:+.4f}",
            "hit_2R": str(t.max_price_300s >= e * 2),
            "hit_3R": str(t.max_price_300s >= e * 3),
            "dead_market": str(dead),
            "time_to_first_move": f"{t.time_to_first_move:.1f}" if t.time_to_first_move >= 0 else "",
            "time_to_2R": f"{t.time_to_2R:.1f}" if t.time_to_2R >= 0 else "",
            "time_to_3R": f"{t.time_to_3R:.1f}" if t.time_to_3R >= 0 else "",
            "match_signal_count": str(t.match_signal_count),
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
