"""
Tennis Alpha Strategy — Tight-Spread Lower-Band (TSLB)

Data-driven strategy from tick mining of 3.27M tennis ticks:
  - SIGNAL: price < 0.20 AND spread <= 0.02 → BUY
  - Tight spread = informed flow arriving, directional continuation
  - 28% win rate (4x better than wide spread 7%)
  - 15% of entries produce >3c moves (fat tails)
  - DO NOT fade drops (momentum continues, mean reversion is negative EV)

Paper trading only. Records all trades to CSV for analysis.
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("tennis.alpha")

# ── Strategy Parameters ────────────────────────────────────────────
MAX_PRICE = 0.20          # Only enter below this price
MAX_SPREAD = 0.02         # Only enter when spread is tight
MIN_PRICE = 0.03          # Ignore near-zero (dead markets)
DEDUP_S = 90.0            # Min seconds between signals on same token
MAX_CONCURRENT = 10       # Max concurrent paper positions
HOLD_TIME_S = 300.0       # Max hold time (5 minutes)
TRAILING_STOP = 0.02      # Exit if price drops 2c from peak
MIN_BOOK_AGE_S = 120.0    # Book must have updated within 120s


@dataclass
class AlphaTrade:
    trade_id: str
    timestamp: float
    token_id: str
    match_title: str
    entry_bid: float
    entry_ask: float
    entry_mid: float
    entry_spread: float
    # Forward tracking
    peak_price: float = 0.0
    min_price: float = 999.0
    price_t30: float = 0.0
    price_t60: float = 0.0
    price_t120: float = 0.0
    price_t300: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_ts: float = 0.0
    _snap_done: set = field(default_factory=set)
    _written: bool = False


class TennisAlpha:
    """Tight-Spread Lower-Band paper trading engine."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._last_signal: dict[str, float] = {}   # token_id → last signal ts
        self._active: dict[str, AlphaTrade] = {}    # trade_id → AlphaTrade
        self._total_entries = 0
        self._total_exits = 0
        self._total_pnl = 0.0
        log.info("TENNIS_ALPHA: initialized | max_price=%.2f max_spread=%.2f",
                 MAX_PRICE, MAX_SPREAD)

    def should_enter(self, token_id: str, bid: float, ask: float,
                     mid: float, spread: float, book_age: float) -> tuple[bool, str]:
        """Check if we should open a paper position."""
        if mid <= MIN_PRICE or mid >= MAX_PRICE:
            return False, "PRICE_RANGE"
        if spread > MAX_SPREAD:
            return False, "WIDE_SPREAD"
        if book_age > MIN_BOOK_AGE_S:
            return False, "STALE_BOOK"
        if len(self._active) >= MAX_CONCURRENT:
            return False, "MAX_CONCURRENT"

        now = time.time()
        if now - self._last_signal.get(token_id, 0) < DEDUP_S:
            return False, "DEDUP"

        return True, "OK"

    def enter(self, token_id: str, bid: float, ask: float,
              mid: float, spread: float, match_title: str) -> AlphaTrade:
        """Open a paper position."""
        now = time.time()
        self._last_signal[token_id] = now
        self._total_entries += 1

        tid = f"TSLB_{self._total_entries}_{int(now)}"
        trade = AlphaTrade(
            trade_id=tid,
            timestamp=now,
            token_id=token_id,
            match_title=match_title,
            entry_bid=bid,
            entry_ask=ask,
            entry_mid=mid,
            entry_spread=spread,
            peak_price=mid,
            min_price=mid,
        )
        self._active[tid] = trade

        log.info("TSLB_ENTRY | %s | mid=%.4f | spread=%.4f | %s",
                 tid, mid, spread, match_title[:50])
        return trade

    def tick(self, books: dict):
        """Update all active positions with latest prices."""
        now = time.time()
        done = []

        for tid, t in list(self._active.items()):
            book = books.get(t.token_id)
            if not book or book.mid <= 0:
                continue

            mid = book.mid
            elapsed = now - t.timestamp

            # Track peaks and mins
            if mid > t.peak_price:
                t.peak_price = mid
            if mid < t.min_price:
                t.min_price = mid

            # Forward snapshots
            if elapsed >= 30 and 30 not in t._snap_done:
                t.price_t30 = mid
                t._snap_done.add(30)
            if elapsed >= 60 and 60 not in t._snap_done:
                t.price_t60 = mid
                t._snap_done.add(60)
            if elapsed >= 120 and 120 not in t._snap_done:
                t.price_t120 = mid
                t._snap_done.add(120)
            if elapsed >= 300 and 300 not in t._snap_done:
                t.price_t300 = mid
                t._snap_done.add(300)

            # ── EXIT CONDITIONS ──
            exit_reason = ""

            # 1. Trailing stop: price dropped TRAILING_STOP from peak
            if t.peak_price - mid >= TRAILING_STOP and t.peak_price > t.entry_mid:
                exit_reason = "TRAIL_STOP"

            # 2. Max hold time
            if elapsed >= HOLD_TIME_S:
                exit_reason = "TIMEOUT"

            # 3. Big win: price doubled (2R)
            if mid >= t.entry_mid * 2:
                exit_reason = "HIT_2R"

            if exit_reason:
                t.exit_price = mid
                t.exit_reason = exit_reason
                t.exit_ts = now
                done.append(tid)

        # Process exits
        for tid in done:
            t = self._active.pop(tid, None)
            if t and not t._written:
                pnl = t.exit_price - t.entry_mid
                self._total_exits += 1
                self._total_pnl += pnl
                r_mult = pnl / t.entry_mid if t.entry_mid > 0 else 0

                log.info(
                    "TSLB_EXIT | %s | pnl=%+.4f | R=%+.1f%% | exit=%s | "
                    "entry=%.4f exit=%.4f peak=%.4f | %s",
                    t.trade_id, pnl, r_mult * 100, t.exit_reason,
                    t.entry_mid, t.exit_price, t.peak_price,
                    t.match_title[:40],
                )
                self._write_csv(t, pnl)
                t._written = True

    def _write_csv(self, t: AlphaTrade, pnl: float):
        """Write completed trade to CSV."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        fp = self.data_dir / f"tslb_trades_{today}.csv"
        hdr = not fp.exists()

        e = t.entry_mid
        r_max = (t.peak_price - e) / e if e > 0.001 else 0
        r_min = (t.min_price - e) / e if e > 0.001 else 0

        row = {
            "trade_id": t.trade_id,
            "timestamp": f"{t.timestamp:.3f}",
            "match_title": t.match_title,
            "token_id": t.token_id[:20],
            "entry_mid": f"{e:.4f}",
            "entry_spread": f"{t.entry_spread:.4f}",
            "exit_price": f"{t.exit_price:.4f}",
            "exit_reason": t.exit_reason,
            "pnl": f"{pnl:+.4f}",
            "peak_price": f"{t.peak_price:.4f}",
            "min_price": f"{t.min_price:.4f}",
            "R_max": f"{r_max:+.2%}",
            "R_min": f"{r_min:+.2%}",
            "hit_2R": str(t.peak_price >= e * 2),
            "price_t30": f"{t.price_t30:.4f}",
            "price_t60": f"{t.price_t60:.4f}",
            "price_t120": f"{t.price_t120:.4f}",
            "price_t300": f"{t.price_t300:.4f}",
            "hold_s": f"{(t.exit_ts - t.timestamp):.0f}",
        }

        try:
            with open(fp, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if hdr:
                    w.writeheader()
                w.writerow(row)
        except Exception as ex:
            log.warning("TSLB_CSV_ERR | %s", ex)

    def stats(self) -> str:
        """Return a status line for the status printer."""
        return (f"TSLB: entries={self._total_entries} exits={self._total_exits} "
                f"active={len(self._active)} pnl={self._total_pnl:+.4f}")
