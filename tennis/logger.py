"""
Tennis edge validation CSV logger.

Writes structured trade/signal data to CSV for post-hoc analysis.
Follows the same CSVLogger pattern from sports/engine.py.

v9.7: Added signal snapshot logging (post-signal price captures).
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Optional

from tennis.state import TennisState, compute_momentum_delta
from tennis.strategy import TennisSignal

log = logging.getLogger("tennis.logger")


# ═══════════════════════════════════════════════════════════════════════
#  Trade Log Schema
# ═══════════════════════════════════════════════════════════════════════

TRADE_LOG_HEADERS = [
    "timestamp",
    "match_id",
    "trigger_type",
    "set_score_at_entry",
    "game_score_at_entry",
    "point_score_at_entry",
    "server_id",
    "market_price_at_entry",
    "fair_price",
    "edge",
    "momentum_delta",
    "market_price_at_bp",
    "market_price_after_hold",
    "lag_detected",
    "exit_reason",
    "R_multiple",
]

SIGNAL_LOG_HEADERS = [
    "timestamp",
    "match_id",
    "trigger_type",
    "edge",
    "fair_price",
    "market_price",
    "set_score",
    "game_score",
    "point_score",
    "server_id",
    "is_break_point",
    "is_tiebreak",
    "momentum_delta",
    "p_a",
    "p_b",
]

# v9.7: Post-signal price snapshot schema
SNAPSHOT_LOG_HEADERS = [
    "signal_time",
    "match_id",
    "token_id",
    "trigger_type",
    "price_signal",
    "price_t5",
    "price_t10",
    "price_t30",
    "time_to_first_tick",
]


# ═══════════════════════════════════════════════════════════════════════
#  CSV Logger
# ═══════════════════════════════════════════════════════════════════════

class TennisCSVLogger:
    """Writes tennis signal and trade data to daily-rotated CSV files.

    Files are created in the data directory with daily rotation:
        tennis_trade_log_YYYYMMDD.csv
        tennis_signals_YYYYMMDD.csv
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, csv.writer] = {}
        self._handles: dict[str, object] = {}

    def _get_writer(self, name: str, headers: list[str]) -> csv.writer:
        if name not in self._files:
            today = time.strftime("%Y%m%d")
            path = self.data_dir / f"{name}_{today}.csv"
            write_header = not path.exists()
            fh = open(path, "a", newline="", buffering=1)
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(headers)
            self._files[name] = writer
            self._handles[name] = fh
        return self._files[name]

    # ── Signal Logging ────────────────────────────────────────────

    def log_signal(self, signal: TennisSignal) -> None:
        """Log a detected strategy signal (before execution gating)."""
        w = self._get_writer("tennis_signals", SIGNAL_LOG_HEADERS)
        st = signal.state_snapshot
        w.writerow([
            signal.timestamp,
            signal.match_id,
            signal.trigger_type,
            f"{signal.edge:.4f}",
            f"{signal.fair_price:.4f}",
            f"{signal.market_price:.4f}",
            f"{st.sets_a}-{st.sets_b}",
            f"{st.games_a}-{st.games_b}",
            f"{st.point_a}-{st.point_b}",
            st.server_id,
            st.is_break_point,
            st.is_tiebreak,
            f"{signal.momentum_delta:.4f}",
            f"{signal.model_output.p_a:.4f}",
            f"{signal.model_output.p_b:.4f}",
        ])

    # ── Trade Entry Logging ───────────────────────────────────────

    def log_trade_entry(self, signal: TennisSignal,
                        market_price_at_bp: float = 0.0) -> None:
        """Log a paper trade entry."""
        w = self._get_writer("tennis_trade_log", TRADE_LOG_HEADERS)
        st = signal.state_snapshot
        w.writerow([
            signal.timestamp,
            signal.match_id,
            signal.trigger_type,
            f"{st.sets_a}-{st.sets_b}",
            f"{st.games_a}-{st.games_b}",
            f"{st.point_a}-{st.point_b}",
            st.server_id,
            f"{signal.market_price:.4f}",
            f"{signal.fair_price:.4f}",
            f"{signal.edge:.4f}",
            f"{signal.momentum_delta:.4f}",
            f"{market_price_at_bp:.4f}" if market_price_at_bp else "",
            "",  # market_price_after_hold — filled on update
            "",  # lag_detected — filled on update
            "",  # exit_reason — filled on exit
            "",  # R_multiple — filled on exit
        ])

    # ── Trade Exit / Update Logging ───────────────────────────────

    def log_trade_exit(self, match_id: str, exit_reason: str,
                       market_price_after_hold: float = 0.0,
                       lag_detected: bool = False,
                       r_multiple: float = 0.0) -> None:
        """Log trade exit as a separate row (linked by match_id)."""
        w = self._get_writer("tennis_trade_log", TRADE_LOG_HEADERS)
        w.writerow([
            time.time(),
            match_id,
            "EXIT",
            "", "", "", "",  # scores not relevant at exit
            "",  # market_price_at_entry
            "",  # fair_price
            "",  # edge
            "",  # momentum_delta
            "",  # market_price_at_bp
            f"{market_price_after_hold:.4f}" if market_price_after_hold else "",
            str(lag_detected),
            exit_reason,
            f"{r_multiple:.4f}" if r_multiple else "",
        ])

    # ── v9.7: Signal Snapshot Logging ─────────────────────────────

    def log_signal_snapshot(self, snap) -> None:
        """Log a post-signal price snapshot row.

        Args:
            snap: SignalSnapshot dataclass from signal_snapshots module.
        """
        w = self._get_writer("tennis_signal_snapshots", SNAPSHOT_LOG_HEADERS)
        w.writerow([
            f"{snap.signal_time:.3f}",
            snap.match_id,
            snap.token_id,
            snap.trigger_type,
            f"{snap.price_signal:.4f}",
            f"{snap.price_t5:.4f}" if snap.price_t5 is not None else "",
            f"{snap.price_t10:.4f}" if snap.price_t10 is not None else "",
            f"{snap.price_t30:.4f}" if snap.price_t30 is not None else "",
            f"{snap.time_to_first_tick:.3f}" if snap.time_to_first_tick is not None else "",
        ])

    # ── Cleanup ───────────────────────────────────────────────────

    def close(self) -> None:
        for fh in self._handles.values():
            try:
                fh.close()
            except Exception:
                pass
        self._files.clear()
        self._handles.clear()
