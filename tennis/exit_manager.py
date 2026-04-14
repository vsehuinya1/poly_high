"""
Tennis Exit Manager — Runner V2 (tick-based trailing).

Tracks open paper trades, detects exit conditions, captures
post-entry price snapshots, and logs complete lifecycle to CSV.

v6.0 — 2026-04-04  Premature-exit elimination
  - REMOVED EXIT_NO_MFE entirely (was causing zero-MFE collapse)
  - REMOVED STAGNATION exit (subsumed by timeout)
  - MIN_HOLD_TIME = 180s — no exit before this except STOP_LOSS / MATCH_END
  - STOP_LOSS tightened to -8% (was -15%)
  - TIMEOUT shortened to 600s (was 2700s)
  - Strict exit hierarchy: STOP_LOSS → MATCH_END → TICK_STOP → RUNNER_V2 → TIMEOUT
v5.0 — 2026-03-28  Hardened exit system
v3.0 — 2026-03-26  Runner V2
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
    # Runner V2 fields
    runner_v2_active: bool = False
    peak_price: float = 0.0
    runner_exit_triggered: bool = False
    consecutive_adverse_ticks: int = 0
    last_tick_price: float = 0.0
    mfe_timestamp: float = 0.0         # when peak MFE was reached
    # Legacy compat — kept for CSV schema but always True under V2
    runner_mode: bool = False
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
    mae_ticks: int = 0             # max adverse excursion in ticks (1 tick = $0.01)
    min_price_seen: float = 0.0    # worst price seen during trade

    @property
    def duration_seconds(self) -> float:
        if self.exit_timestamp and self.entry_timestamp:
            return self.exit_timestamp - self.entry_timestamp
        return time.time() - self.entry_timestamp

    @property
    def age_minutes(self) -> float:
        return (time.time() - self.entry_timestamp) / 60.0

    @property
    def capture_ratio(self) -> float:
        """Ratio of realized gain to MFE. 1.0 = perfect exit at peak."""
        if self.mfe <= 0 or self.exit_price is None:
            return 0.0
        realized = self.exit_price - self.entry_price
        return max(0.0, realized / self.mfe)

    @property
    def time_to_mfe_seconds(self) -> float:
        """Seconds from entry to when MFE was reached."""
        if self.mfe_timestamp and self.entry_timestamp:
            return self.mfe_timestamp - self.entry_timestamp
        return 0.0


# v6.0: Only these exit reasons are valid — anything else is a bug
_VALID_EXIT_REASONS = {
    "EXIT_STOP_LOSS",
    "EXIT_TICK_STOP",
    "EXIT_RUNNER_V2",
    "EXIT_MATCH_END",
    "EXIT_TIMEOUT",
    "REPLACED",           # internal: replaced by new trade for same match
}


class TennisExitManager:
    """Manages tennis paper trade lifecycle: snapshots, exits, CSV logging.

    Runner V2 — tick-based trailing exit system.
    Strict exit hierarchy — NO convergence, NO price floor, NO hard stop.

    Non-blocking — designed to run inside the existing async polling loop.
    Call check_all() on every tick.
    """

    # ── Runner V2 Parameters (v8.1) ───────────────────────────────
    RUNNER_V2_MFE_THRESHOLD = 0.06   # 6% MFE before trailing activates
    RUNNER_V2_TRAIL_PCT = 0.30       # give back up to 30% of gains
    RUNNER_V2_TRAIL_PCT_LATE = 0.20  # tighten to 20% after 20 minutes
    RUNNER_V2_LATE_S = 1200.0        # 20 minutes
    RUNNER_V2_CONFIRM_TICKS = 2      # consecutive adverse ticks to confirm reversal

    # ── Risk Control (v6.0) ──────────────────────────────────────
    STOP_LOSS_R = 0.08                 # stop-loss at -8% of entry price
    TICK_SIZE = 0.01                   # Polymarket tick size ($0.01)
    MAX_ADVERSE_TICKS = 10             # tick-based stop (secondary)

    # ── Hold Time & Timeout (v6.0) ──────────────────────────────
    MIN_HOLD_TIME_S = 180.0            # 3 minutes — NO exit before this (except STOP_LOSS / MATCH_END)
    TIMEOUT_S = 600.0                  # 10 minutes hard timeout
    SPREAD_CAPTURE_THRESHOLD = 0.04
    SNAPSHOT_TIMES = [
        (5 * 60,   "price_t5",  "_snapshot_5_done"),
        (15 * 60,  "price_t15", "_snapshot_15_done"),
        (30 * 60,  "price_t30", "_snapshot_30_done"),
        (60 * 60,  "price_t60", "_snapshot_60_done"),
    ]

    def __init__(self, data_dir: Path, on_close=None):
        # v6.0: Validate critical configuration
        assert self.MIN_HOLD_TIME_S == 180.0, \
            f"MIN_HOLD_TIME_S must be 180, got {self.MIN_HOLD_TIME_S}"
        assert self.TIMEOUT_S == 600.0, \
            f"TIMEOUT_S must be 600, got {self.TIMEOUT_S}"

        self.open_trades: dict[str, TennisPaperTrade] = {}  # match_id → trade
        self.closed_trades: list[TennisPaperTrade] = []
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._csv_writer = None
        self._csv_fh = None
        self._csv_initialized = False
        self._on_close = on_close  # callback(trade) for live sell hook

        # v6.0: MFE distribution tracker
        self._mfe_buckets: dict[str, int] = {
            "lt_0.01": 0,
            "0.01_0.02": 0,
            "ge_0.02": 0,
        }
        self._total_closed = 0

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
        # ═══ GLOBAL EDGE GUARD (v6.1 — inside function, non-bypassable) ═══
        from sports.guards import validate_trade_execution, circuit_breaker
        can_exec, block_reason = validate_trade_execution(
            edge=edge, price=entry_price, sport="tennis",
            context=f"{trigger_type} | {player} | {match_id}",
        )
        if not can_exec:
            circuit_breaker.record_signal_result(was_blocked=True)
            return None
        circuit_breaker.record_signal_result(was_blocked=False)

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
            mfe_timestamp=time.time(),
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

        Runner V2 logic:
          1. Track MFE in real time
          2. Activate trailing ONLY after MFE >= 5%
          3. Dynamic trail at 50% of MFE
          4. Exit on confirmed reversal (2+ consecutive adverse ticks)
          5. Stagnation safeguard (30min + MFE < 3%)

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
                trade.mfe_timestamp = now  # update time-to-MFE
            if adverse > trade.mae:
                trade.mae = adverse
                trade.mae_ticks = round(adverse / 0.01)

            # ── Peak / Min tracking ───────────────────────────
            if mkt > trade.peak_price:
                trade.peak_price = mkt
            if mkt < trade.min_price_seen or trade.min_price_seen == 0:
                trade.min_price_seen = mkt

            # ── Runner V2: Track consecutive adverse ticks ────
            if trade.last_tick_price > 0:
                if mkt < trade.last_tick_price:
                    trade.consecutive_adverse_ticks += 1
                else:
                    trade.consecutive_adverse_ticks = 0
            trade.last_tick_price = mkt

            # ── Runner V2: Activate trailing after MFE threshold ──
            if not trade.runner_v2_active and trade.mfe >= self.RUNNER_V2_MFE_THRESHOLD:
                trade.runner_v2_active = True
                trade.runner_mode = True  # legacy compat
                log.info("EXIT_MGR RUNNER_V2 ACTIVATED | %s | mfe=%.4f >= %.4f | mkt=%.4f peak=%.4f",
                         match_id, trade.mfe, self.RUNNER_V2_MFE_THRESHOLD,
                         mkt, trade.peak_price)

            # ═══════════════════════════════════════════════════════
            #  EXIT HIERARCHY (v6.0 — premature-exit elimination)
            #  Priority: STOP_LOSS → MATCH_END → [MIN_HOLD gate] →
            #            TICK_STOP → RUNNER_V2 → TIMEOUT
            #  NO convergence. NO NO_MFE. NO stagnation.
            # ═══════════════════════════════════════════════════════

            def _capture_spread():
                if get_spread:
                    trade.spread_at_exit = get_spread(match_id, trade.selection_id) or 0.0
                trade.mid_price_exit = mkt

            # 1. STOP_LOSS — always active, no min hold gate (flat -8% cap)
            stop_price = trade.entry_price * (1.0 - self.STOP_LOSS_R)
            if mkt <= stop_price:
                _capture_spread()
                log.info(
                    "EXIT_MGR STOP_LOSS | %s | entry=%.4f stop=%.4f mkt=%.4f | elapsed=%.0fs",
                    match_id, trade.entry_price, stop_price, mkt, elapsed,
                )
                self._close_trade(trade, exit_price=mkt,
                                  exit_reason="EXIT_STOP_LOSS",
                                  exit_score=score)
                continue

            # 2. MATCH_END — always active, no min hold gate
            if is_match_finished(match_id):
                _capture_spread()
                self._close_trade(trade, exit_price=mkt,
                                  exit_reason="EXIT_MATCH_END",
                                  exit_score=score)
                continue

            # ── MIN HOLD TIME GATE (v6.0) ─────────────────────────
            # Nothing below this line fires before MIN_HOLD_TIME_S
            if elapsed < self.MIN_HOLD_TIME_S:
                continue

            # 3. TICK_STOP (SECONDARY — capped by flat stop-loss)
            tick_stop_price = trade.entry_price - (self.MAX_ADVERSE_TICKS * self.TICK_SIZE)
            flat_stop_price = trade.entry_price * (1.0 - self.STOP_LOSS_R)
            effective_stop_price = max(tick_stop_price, flat_stop_price)

            if mkt <= effective_stop_price:
                _capture_spread()
                log.info(
                    "EXIT_MGR TICK_STOP | %s | entry=%.4f mkt=%.4f | "
                    "tick_stop=%.4f flat_stop=%.4f effective=%.4f | elapsed=%.0fs",
                    match_id, trade.entry_price, mkt,
                    tick_stop_price, flat_stop_price, effective_stop_price, elapsed,
                )
                self._close_trade(trade, exit_price=mkt,
                                  exit_reason="EXIT_TICK_STOP",
                                  exit_score=score)
                continue

            # 4. RUNNER_V2 trailing stop (confirmed reversal)
            if trade.runner_v2_active:
                # v8.1: time-based tightening — less giveback after 20min
                trail_pct = self.RUNNER_V2_TRAIL_PCT
                if elapsed > self.RUNNER_V2_LATE_S:
                    trail_pct = self.RUNNER_V2_TRAIL_PCT_LATE
                trail_level = trade.mfe * trail_pct
                exit_threshold = trade.entry_price + trail_level

                if (mkt <= exit_threshold
                        and trade.consecutive_adverse_ticks >= self.RUNNER_V2_CONFIRM_TICKS):
                    trade.runner_exit_triggered = True
                    _capture_spread()
                    self._close_trade(trade, exit_price=mkt,
                                      exit_reason="EXIT_RUNNER_V2",
                                      exit_score=score)
                    continue

            # 5. TIMEOUT — 10 min hard cap (after min hold)
            #    v8.0: Skip timeout when runner is active and profitable —
            #    let the trailing stop capture multi-R moves.
            if elapsed >= self.TIMEOUT_S:
                if trade.runner_v2_active and mkt > trade.entry_price:
                    pass  # runner rides — trail stop will handle exit
                else:
                    _capture_spread()
                    self._close_trade(trade, exit_price=mkt,
                                      exit_reason="EXIT_TIMEOUT",
                                      exit_score=score)
                    continue

    # ── Internal ────────────────────────────────────────────────

    def _close_trade(self, trade: TennisPaperTrade, exit_price: float,
                     exit_reason: str, exit_score: Optional[str]):
        """Close a trade, compute R, log to CSV, move to closed list."""
        # v5.0: Validate exit reason
        assert exit_reason in _VALID_EXIT_REASONS, \
            f"Invalid exit_reason '{exit_reason}' — must be one of {_VALID_EXIT_REASONS}"

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

        elapsed = trade.duration_seconds

        # Move from open to closed
        self.open_trades.pop(trade.match_id, None)
        self.closed_trades.append(trade)

        # v5.0: Diagnostic summary log
        log.info(
            "TENNIS_EXIT_SUMMARY | "
            "reason=%s | mfe=%.4f | mae=%.4f | "
            "time_to_mfe=%.1fs | duration=%.1fs | "
            "entry=%.4f exit=%.4f R=%+.4f | %s",
            exit_reason, trade.mfe, trade.mae,
            trade.time_to_mfe_seconds, elapsed,
            trade.entry_price, exit_price, trade.R_multiple,
            trade.match_id,
        )

        # v7.0: MISSED_RUNNER — log trades that had high MFE but exited poorly
        if trade.mfe >= 0.05 and trade.R_multiple < 0.05:
            log.warning(
                "MISSED_RUNNER | mfe=%.4f exit_R=%+.4f | "
                "exit_reason=%s | duration=%.0fs | "
                "runner_active=%s | %s",
                trade.mfe, trade.R_multiple,
                exit_reason, elapsed,
                trade.runner_v2_active, trade.match_id,
            )

        # v7.0: Feed circuit breaker with trade outcome
        from sports.guards import circuit_breaker
        circuit_breaker.record_trade_outcome(trade.R_multiple)

        # v5.0: Update MFE distribution buckets
        self._update_mfe_buckets(trade)

        self._write_lifecycle_row(trade)

        # Fire on_close callback for live sell
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
                "runner_v2_active", "peak_price",
                "exit_price", "exit_reason", "R_multiple",
                "entry_score", "exit_score", "duration_seconds",
                "timestamp_entry", "timestamp_exit",
                # v3.0 fields
                "mfe", "mae", "mae_ticks", "min_price_seen",
                "capture_ratio", "time_to_mfe",
                "consecutive_adverse_ticks",
                "spread_at_signal", "spread_at_entry", "spread_at_exit",
                "mid_price_signal", "mid_price_entry", "mid_price_exit",
            ])
        self._csv_initialized = True

    def _write_lifecycle_row(self, t: TennisPaperTrade):
        """Write one complete lifecycle row to CSV."""
        self._ensure_csv()
        self._csv_writer.writerow([
            "3",  # schema_version — Runner V2
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
            "1" if t.runner_v2_active else "0",
            f"{t.peak_price:.4f}",
            f"{t.exit_price:.4f}" if t.exit_price is not None else "",
            t.exit_reason or "",
            f"{t.R_multiple:+.4f}" if t.R_multiple is not None else "",
            t.entry_score,
            t.exit_score or "",
            f"{t.duration_seconds:.0f}",
            f"{t.entry_timestamp:.3f}",
            f"{t.exit_timestamp:.3f}" if t.exit_timestamp else "",
            # v3.0 fields
            f"{t.mfe:.4f}",
            f"{t.mae:.4f}",
            str(t.mae_ticks),
            f"{t.min_price_seen:.4f}",
            f"{t.capture_ratio:.4f}",
            f"{t.time_to_mfe_seconds:.1f}",
            str(t.consecutive_adverse_ticks),
            f"{t.spread_at_signal:.4f}",
            f"{t.spread_at_entry:.4f}",
            f"{t.spread_at_exit:.4f}",
            f"{t.mid_price_signal:.4f}",
            f"{t.mid_price_entry:.4f}",
            f"{t.mid_price_exit:.4f}",
        ])

    # ── MFE Distribution Tracker (v5.0) ──────────────────────────

    def _update_mfe_buckets(self, trade: TennisPaperTrade):
        """Update MFE distribution buckets and log every 10 trades."""
        if trade.mfe < 0.01:
            self._mfe_buckets["lt_0.01"] += 1
        elif trade.mfe < 0.02:
            self._mfe_buckets["0.01_0.02"] += 1
        else:
            self._mfe_buckets["ge_0.02"] += 1

        self._total_closed += 1
        if self._total_closed % 10 == 0:
            log.info(
                "TENNIS_MFE_DIST | lt_0.01=%d | "
                "0.01_0.02=%d | ge_0.02=%d | total=%d",
                self._mfe_buckets["lt_0.01"],
                self._mfe_buckets["0.01_0.02"],
                self._mfe_buckets["ge_0.02"],
                self._total_closed,
            )

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
        runner_v2 = sum(1 for t in closed if t.exit_reason == "EXIT_RUNNER_V2")
        match_end = sum(1 for t in closed if t.exit_reason == "EXIT_MATCH_END")
        timeout = sum(1 for t in closed if t.exit_reason == "EXIT_TIMEOUT")
        stop_loss = sum(1 for t in closed if t.exit_reason == "EXIT_STOP_LOSS")
        tick_stop = sum(1 for t in closed if t.exit_reason == "EXIT_TICK_STOP")
        spread_captures = sum(1 for t in closed if t.spread_capture)
        runner_trades = sum(1 for t in closed if t.runner_v2_active)
        r_values = [t.R_multiple for t in closed if t.R_multiple is not None]
        runner_r = [t.R_multiple for t in closed if t.runner_v2_active and t.R_multiple is not None]
        capture_ratios = [t.capture_ratio for t in closed if t.mfe > 0]
        avg_r = sum(r_values) / len(r_values) if r_values else 0.0
        avg_runner_r = sum(runner_r) / len(runner_r) if runner_r else 0.0
        avg_capture = sum(capture_ratios) / len(capture_ratios) if capture_ratios else 0.0

        return {
            "trades_opened": len(self.open_trades) + len(closed),
            "trades_closed": len(closed),
            "trades_open": len(self.open_trades),
            "exit_runner_v2": runner_v2,
            "exit_match_end": match_end,
            "exit_timeout": timeout,
            "exit_stop_loss": stop_loss,
            "exit_tick_stop": tick_stop,
            "spread_capture_entries": spread_captures,
            "runner_v2_trades": runner_trades,
            "avg_R_multiple": avg_r,
            "avg_runner_R": avg_runner_r,
            "avg_capture_ratio": avg_capture,
            "mfe_buckets": dict(self._mfe_buckets),
        }
