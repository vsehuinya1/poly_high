"""
Tennis Exit Manager — lifecycle tracking for paper trades.

Tracks open paper trades, detects exit conditions, captures
post-entry price snapshots, and logs complete lifecycle to CSV.

v1.1 — 2026-03-06
  - Spread capture logging (paper PnL uses adjusted entry)
  - Runner mode (trailing stop after +0.20 gain)
  - T+60 snapshot
"""
import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("tennis.exit_manager")


@dataclass
class TennisPaperTrade:
    """A single tennis paper trade with full lifecycle data."""
    match_id: str
    selection_id: str        # Polymarket token_id
    player: str              # player name (favorite usually)
    trigger_type: str        # SET_MEAN_REVERSION or PANIC_DISCOUNT
    entry_price: float
    fair_value_entry: float
    edge_entry: float
    entry_timestamp: float
    entry_score: str         # e.g. "0-1 2-0"
    # Spread capture (v1.1)
    spread: float = 0.0
    spread_capture: bool = False
    adjusted_entry_price: float = 0.0   # entry_price - 0.01 if spread_capture
    # Post-entry snapshots
    price_t5: Optional[float] = None
    price_t15: Optional[float] = None
    price_t30: Optional[float] = None
    price_t60: Optional[float] = None
    _snapshot_5_done: bool = field(default=False, repr=False)
    _snapshot_15_done: bool = field(default=False, repr=False)
    _snapshot_30_done: bool = field(default=False, repr=False)
    _snapshot_60_done: bool = field(default=False, repr=False)
    # Runner mode (v1.1)
    runner_mode: bool = False
    peak_price: float = 0.0
    runner_exit_triggered: bool = False
    # Exit data
    exit_price: Optional[float] = None
    exit_timestamp: Optional[float] = None
    exit_score: Optional[str] = None
    exit_reason: Optional[str] = None
    R_multiple: Optional[float] = None
    is_open: bool = True
    # ── Execution metrics (v2.0) ──────────────────────────────────
    spread_at_signal: float = 0.0
    spread_at_entry: float = 0.0
    spread_at_exit: float = 0.0
    mid_price_signal: float = 0.0
    mid_price_entry: float = 0.0
    mid_price_exit: float = 0.0
    mfe: float = 0.0               # max favorable excursion (absolute)
    mae: float = 0.0               # max adverse excursion (absolute)
    min_price_seen: float = 0.0    # worst price seen during trade

    @property
    def duration_seconds(self) -> float:
        if self.exit_timestamp and self.entry_timestamp:
            return self.exit_timestamp - self.entry_timestamp
        return time.time() - self.entry_timestamp

    @property
    def age_minutes(self) -> float:
        return (time.time() - self.entry_timestamp) / 60.0


class TennisExitManager:
    """Manages tennis paper trade lifecycle: snapshots, exits, CSV logging.

    Non-blocking — designed to run inside the existing async polling loop.
    Call check_all() on every tick.
    """

    CONVERGENCE_THRESHOLD = 0.01   # exit when abs(fair - mkt) < this
    TIMEOUT_S = 7200.0             # 2 hours
    RUNNER_TRIGGER = 0.20          # price gain to activate runner mode
    RUNNER_TRAIL_PCT = 0.30        # trailing stop = 30% of (peak - entry)
    RUNNER_TRAIL_MIN = 0.02        # minimum trail distance
    RUNNER_TRAIL_MAX = 0.05        # maximum trail distance
    SPREAD_CAPTURE_THRESHOLD = 0.04  # wide spread logging threshold
    SNAPSHOT_TIMES = [
        (5 * 60,   "price_t5",  "_snapshot_5_done"),
        (15 * 60,  "price_t15", "_snapshot_15_done"),
        (30 * 60,  "price_t30", "_snapshot_30_done"),
        (60 * 60,  "price_t60", "_snapshot_60_done"),
    ]

    def __init__(self, data_dir: Path, on_close=None):
        self.open_trades: dict[str, TennisPaperTrade] = {}  # match_id → trade
        self.closed_trades: list[TennisPaperTrade] = []
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._csv_writer = None
        self._csv_fh = None
        self._csv_initialized = False
        self._on_close = on_close  # callback(trade) for live sell hook

    # ── Trade Registration ──────────────────────────────────────

    def register_trade(
        self,
        match_id: str,
        selection_id: str,
        player: str,
        trigger_type: str,
        entry_price: float,
        fair_value: float,
        edge: float,
        entry_score: str,
        spread: float = 0.0,
        spread_at_signal: float = 0.0,
        mid_price_signal: float = 0.0,
        mid_price_entry: float = 0.0,
    ) -> TennisPaperTrade:
        """Register a new paper trade after a signal is accepted."""
        # Spread capture: log-only adjusted entry for paper PnL
        spread_capture = spread > self.SPREAD_CAPTURE_THRESHOLD
        adjusted = entry_price - 0.01 if spread_capture else entry_price

        trade = TennisPaperTrade(
            match_id=match_id,
            selection_id=selection_id,
            player=player,
            trigger_type=trigger_type,
            entry_price=entry_price,
            fair_value_entry=fair_value,
            edge_entry=edge,
            entry_timestamp=time.time(),
            entry_score=entry_score,
            spread=spread,
            spread_capture=spread_capture,
            adjusted_entry_price=adjusted,
            peak_price=entry_price,
            min_price_seen=entry_price,
            # Execution metrics
            spread_at_signal=spread_at_signal,
            spread_at_entry=spread,
            mid_price_signal=mid_price_signal,
            mid_price_entry=mid_price_entry,
        )

        # If there's already an open trade for this match, close it first
        if match_id in self.open_trades:
            self._close_trade(
                self.open_trades[match_id],
                exit_price=entry_price,
                exit_reason="REPLACED",
                exit_score=entry_score,
            )

        self.open_trades[match_id] = trade
        sc_tag = " [SPREAD_CAPTURE]" if spread_capture else ""
        log.info("EXIT_MGR OPEN | %s | %s | entry=%.4f adj=%.4f fair=%.4f edge=%+.4f | spread=%.3f%s | %s",
                 match_id, trigger_type, entry_price, adjusted, fair_value,
                 edge, spread, sc_tag, entry_score)
        return trade

    # ── Tick Processing ─────────────────────────────────────────

    def check_all(
        self,
        get_market_price: callable,
        get_fair_value: callable,
        get_score: callable,
        is_match_finished: callable,
        get_spread: callable = None,
    ):
        """Check all open trades for snapshots and exit conditions.

        Args:
            get_market_price: fn(match_id, selection_id) → float or None
            get_fair_value: fn(match_id) → float or None
            get_score: fn(match_id) → str or None
            is_match_finished: fn(match_id) → bool
            get_spread: fn(match_id, selection_id) → float or None (optional)
        """
        now = time.time()

        for match_id in list(self.open_trades.keys()):
            trade = self.open_trades[match_id]

            mkt = get_market_price(match_id, trade.selection_id)
            if mkt is None or mkt <= 0:
                continue

            fair = get_fair_value(match_id)
            score = get_score(match_id)
            elapsed = now - trade.entry_timestamp

            # ── Post-entry snapshots ───────────────────────────
            for delay_s, attr, flag_attr in self.SNAPSHOT_TIMES:
                if not getattr(trade, flag_attr) and elapsed >= delay_s:
                    setattr(trade, attr, mkt)
                    setattr(trade, flag_attr, True)
                    log.info("EXIT_MGR SNAP | %s | %s=%.4f (T+%dm)",
                             match_id, attr, mkt, delay_s // 60)

            # ── MFE / MAE tracking ────────────────────────────
            favorable = mkt - trade.entry_price
            adverse = trade.entry_price - mkt
            if favorable > trade.mfe:
                trade.mfe = favorable
            if adverse > trade.mae:
                trade.mae = adverse

            # ── Peak / Min tracking ───────────────────────────
            if mkt > trade.peak_price:
                trade.peak_price = mkt
            if mkt < trade.min_price_seen or trade.min_price_seen == 0:
                trade.min_price_seen = mkt

            price_gain = mkt - trade.entry_price
            if not trade.runner_mode and price_gain >= self.RUNNER_TRIGGER:
                trade.runner_mode = True
                log.info("EXIT_MGR RUNNER | %s | activated at mkt=%.4f (gain=%.4f, peak=%.4f)",
                         match_id, mkt, price_gain, trade.peak_price)

            # ── Exit conditions ──────────────────────────────

            # 0. Runner trailing stop (dynamic, overrides convergence)
            if trade.runner_mode:
                trail_distance = max(
                    self.RUNNER_TRAIL_MIN,
                    min(self.RUNNER_TRAIL_MAX,
                        (trade.peak_price - trade.entry_price) * self.RUNNER_TRAIL_PCT),
                )
                if mkt <= trade.peak_price - trail_distance:
                    trade.runner_exit_triggered = True
                    # Capture spread at exit
                    if get_spread:
                        trade.spread_at_exit = get_spread(match_id, trade.selection_id) or 0.0
                    trade.mid_price_exit = mkt
                    self._close_trade(trade, exit_price=mkt,
                                      exit_reason="EXIT_RUNNER_TRAIL",
                                      exit_score=score)
                    continue
                # Skip convergence exit — let runner run
            else:
                # 1. Momentum-aware convergence
                if fair is not None and mkt >= fair:
                    overshoot_pct = (mkt - fair) / fair if fair > 0 else 0
                    if overshoot_pct > 0.03 and mkt > trade.entry_price:
                        # Price is OVERSHOOTING fair value — switch to runner
                        trade.runner_mode = True
                        log.info("EXIT_MGR CONVERGENCE_TO_RUNNER | %s | mkt=%.4f > fair=%.4f (overshoot=%.1f%%)",
                                 match_id, mkt, fair, overshoot_pct * 100)
                    else:
                        # Price hit fair value and stopped — exit as convergence
                        if get_spread:
                            trade.spread_at_exit = get_spread(match_id, trade.selection_id) or 0.0
                        trade.mid_price_exit = mkt
                        self._close_trade(trade, exit_price=mkt,
                                          exit_reason="EXIT_CONVERGENCE",
                                          exit_score=score)
                        continue
                elif fair is not None and abs(fair - mkt) < self.CONVERGENCE_THRESHOLD:
                    # Original convergence: within threshold of fair value
                    if get_spread:
                        trade.spread_at_exit = get_spread(match_id, trade.selection_id) or 0.0
                    trade.mid_price_exit = mkt
                    self._close_trade(trade, exit_price=mkt,
                                      exit_reason="EXIT_CONVERGENCE",
                                      exit_score=score)
                    continue

            # 2. Match end (always applies)
            if is_match_finished(match_id):
                if get_spread:
                    trade.spread_at_exit = get_spread(match_id, trade.selection_id) or 0.0
                trade.mid_price_exit = mkt
                self._close_trade(trade, exit_price=mkt,
                                  exit_reason="EXIT_MATCH_END",
                                  exit_score=score)
                continue

            # 3. Timeout — 2 hours (always applies)
            if elapsed >= self.TIMEOUT_S:
                if get_spread:
                    trade.spread_at_exit = get_spread(match_id, trade.selection_id) or 0.0
                trade.mid_price_exit = mkt
                self._close_trade(trade, exit_price=mkt,
                                  exit_reason="EXIT_TIMEOUT",
                                  exit_score=score)
                continue

    # ── Internal ────────────────────────────────────────────────

    def _close_trade(self, trade: TennisPaperTrade, exit_price: float,
                     exit_reason: str, exit_score: Optional[str]):
        """Close a trade, compute R, log to CSV, move to closed list."""
        trade.exit_price = exit_price
        trade.exit_timestamp = time.time()
        trade.exit_reason = exit_reason
        trade.exit_score = exit_score or ""
        trade.is_open = False

        # R_multiple: use adjusted_entry_price for paper PnL if spread_capture
        base = trade.adjusted_entry_price if trade.spread_capture else trade.entry_price
        if base > 0:
            trade.R_multiple = (exit_price - base) / base
        else:
            trade.R_multiple = 0.0

        # Move from open to closed
        self.open_trades.pop(trade.match_id, None)
        self.closed_trades.append(trade)

        log.info(
            "EXIT_MGR CLOSE | %s | %s | entry=%.4f → exit=%.4f | R=%+.4f | %s | %.0fs",
            trade.match_id, trade.exit_reason,
            trade.entry_price, exit_price,
            trade.R_multiple, trade.exit_score,
            trade.duration_seconds,
        )

        self._write_lifecycle_row(trade)

        # v4.4: fire on_close callback for live sell
        if self._on_close:
            try:
                self._on_close(trade)
            except Exception as e:
                log.error("on_close callback error: %s", e)

    def _ensure_csv(self):
        """Lazily create CSV writer."""
        if self._csv_initialized:
            return
        today = time.strftime("%Y%m%d")
        path = self._data_dir / f"tennis_trade_lifecycle_{today}.csv"
        write_header = not path.exists()
        self._csv_fh = open(path, "a", newline="", buffering=1)
        self._csv_writer = csv.writer(self._csv_fh)
        if write_header:
            self._csv_writer.writerow([
                "schema_version",
                "match_id", "player", "trigger", "entry_price", "fair_entry",
                "edge_entry", "spread", "spread_capture", "adjusted_entry_price",
                "price_t5", "price_t15", "price_t30", "price_t60",
                "runner_mode", "peak_price",
                "exit_price", "exit_reason", "R_multiple",
                "entry_score", "exit_score", "duration_seconds",
                "timestamp_entry", "timestamp_exit",
                # v2.0 fields
                "mfe", "mae", "min_price_seen",
                "spread_at_signal", "spread_at_entry", "spread_at_exit",
                "mid_price_signal", "mid_price_entry", "mid_price_exit",
            ])
        self._csv_initialized = True

    def _write_lifecycle_row(self, t: TennisPaperTrade):
        """Write one complete lifecycle row to CSV."""
        self._ensure_csv()
        self._csv_writer.writerow([
            "2",  # schema_version
            t.match_id,
            t.player,
            t.trigger_type,
            f"{t.entry_price:.4f}",
            f"{t.fair_value_entry:.4f}",
            f"{t.edge_entry:+.4f}",
            f"{t.spread:.4f}",
            "1" if t.spread_capture else "0",
            f"{t.adjusted_entry_price:.4f}",
            f"{t.price_t5:.4f}" if t.price_t5 is not None else "",
            f"{t.price_t15:.4f}" if t.price_t15 is not None else "",
            f"{t.price_t30:.4f}" if t.price_t30 is not None else "",
            f"{t.price_t60:.4f}" if t.price_t60 is not None else "",
            "1" if t.runner_mode else "0",
            f"{t.peak_price:.4f}",
            f"{t.exit_price:.4f}" if t.exit_price is not None else "",
            t.exit_reason or "",
            f"{t.R_multiple:+.4f}" if t.R_multiple is not None else "",
            t.entry_score,
            t.exit_score or "",
            f"{t.duration_seconds:.0f}",
            f"{t.entry_timestamp:.3f}",
            f"{t.exit_timestamp:.3f}" if t.exit_timestamp else "",
            # v2.0 fields
            f"{t.mfe:.4f}",
            f"{t.mae:.4f}",
            f"{t.min_price_seen:.4f}",
            f"{t.spread_at_signal:.4f}",
            f"{t.spread_at_entry:.4f}",
            f"{t.spread_at_exit:.4f}",
            f"{t.mid_price_signal:.4f}",
            f"{t.mid_price_entry:.4f}",
            f"{t.mid_price_exit:.4f}",
        ])

    def close(self):
        """Flush and close CSV file handle."""
        if self._csv_fh:
            try:
                self._csv_fh.close()
            except Exception:
                pass

    # ── Stats ───────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return exit stats for health dashboard integration."""
        closed = self.closed_trades
        convergence = sum(1 for t in closed if t.exit_reason == "EXIT_CONVERGENCE")
        match_end = sum(1 for t in closed if t.exit_reason == "EXIT_MATCH_END")
        timeout = sum(1 for t in closed if t.exit_reason == "EXIT_TIMEOUT")
        runner_trail = sum(1 for t in closed if t.exit_reason == "EXIT_RUNNER_TRAIL")
        spread_captures = sum(1 for t in closed if t.spread_capture)
        runner_trades = sum(1 for t in closed if t.runner_mode)
        r_values = [t.R_multiple for t in closed if t.R_multiple is not None]
        runner_r = [t.R_multiple for t in closed if t.runner_mode and t.R_multiple is not None]
        avg_r = sum(r_values) / len(r_values) if r_values else 0.0
        avg_runner_r = sum(runner_r) / len(runner_r) if runner_r else 0.0

        return {
            "trades_opened": len(self.open_trades) + len(closed),
            "trades_closed": len(closed),
            "trades_open": len(self.open_trades),
            "exit_convergence": convergence,
            "exit_match_end": match_end,
            "exit_timeout": timeout,
            "exit_runner_trail": runner_trail,
            "spread_capture_entries": spread_captures,
            "runner_mode_trades": runner_trades,
            "avg_R_multiple": avg_r,
            "avg_runner_R": avg_runner_r,
        }
