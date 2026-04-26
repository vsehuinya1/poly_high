"""
Cricket Exit Manager — paper trade lifecycle tracking with MAE/MFE.

Tracks open paper trades, captures post-entry price snapshots (T+5s, T+15s,
T+30s, T+60s), computes live MAE/MFE excursion, and logs complete lifecycle
to CSV for backtesting analysis.

Modeled on tennis/exit_manager.py — same data columns for cross-sport analysis.
"""
import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("cricket.exit_manager")


@dataclass
class CricketPaperTrade:
    """A single cricket paper trade with full lifecycle data."""
    match_id: str
    selection_id: str           # Polymarket token_id
    signal_type: str            # MOMENTUM_EDGE, WICKET_OVERREACTION, etc.
    entry_price: float
    fair_value_entry: float
    edge_entry: float
    entry_timestamp: float
    entry_score: str            # e.g. "AFG 123/4 (18.2) vs SL 89/2 (12.0)"
    match_title: str = ""
    # Delayed entry tracking
    entry_delay_s: float = 0.0  # actual delay from signal → entry
    # Post-entry snapshots
    price_t5: Optional[float] = None
    price_t15: Optional[float] = None
    price_t30: Optional[float] = None
    price_t60: Optional[float] = None
    _snapshot_5_done: bool = field(default=False, repr=False)
    _snapshot_15_done: bool = field(default=False, repr=False)
    _snapshot_30_done: bool = field(default=False, repr=False)
    _snapshot_60_done: bool = field(default=False, repr=False)
    # Excursion tracking
    peak_price: float = 0.0         # best price seen (MFE direction)
    min_price_seen: float = 0.0     # worst price seen (MAE direction)
    mfe: float = 0.0                # max favorable excursion (absolute)
    mae: float = 0.0                # max adverse excursion (absolute)
    # Spread capture
    spread_at_entry: float = 0.0
    spread_at_exit: float = 0.0
    # Exit data
    exit_price: Optional[float] = None
    exit_timestamp: Optional[float] = None
    exit_score: Optional[str] = None
    exit_reason: Optional[str] = None
    is_open: bool = True

    @property
    def duration_seconds(self) -> float:
        if self.exit_timestamp and self.entry_timestamp:
            return self.exit_timestamp - self.entry_timestamp
        return time.time() - self.entry_timestamp

    @property
    def age_minutes(self) -> float:
        return (time.time() - self.entry_timestamp) / 60.0

    @property
    def paper_pnl(self) -> float:
        """Simple PnL: exit_price - entry_price."""
        if self.exit_price is not None:
            return self.exit_price - self.entry_price
        return 0.0


class CricketExitManager:
    """Paper trade lifecycle manager for cricket.

    Responsibilities:
      1. Register new paper entries with full metadata
      2. Track post-entry price snapshots (T+5, T+15, T+30, T+60)
      3. Compute live MAE/MFE on each check
      4. Detect exit conditions (timeout, match end, trailing stop)
      5. Log complete lifecycle to CSV
    """

    # Exit thresholds
    STOP_LOSS_TICKS = 15        # MAE-based stop (15 ticks = 0.15)
    TIMEOUT_MINUTES = 12.0      # v2.1: ~2 overs forced exit for paper mode
    RUNNER_GAIN = 0.15          # enter "runner mode" after +0.15
    RUNNER_TRAIL_PCT = 0.40     # trailing stop at 40% of peak gain

    def __init__(self, data_dir: Path = Path("sports_data"), on_close=None):
        self.open_trades: dict[str, CricketPaperTrade] = {}
        self.closed_trades: list[CricketPaperTrade] = []
        self._csv_path = data_dir / "cricket_paper_trades.csv"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._csv_written_header = self._csv_path.exists()
        self.stats = _CricketExitStats()
        self._on_close = on_close  # callback(trade) on close

    def register_trade(
        self,
        match_id: str,
        selection_id: str,
        signal_type: str,
        entry_price: float,
        fair_value: float,
        edge: float,
        entry_score: str,
        spread: float = 0.0,
        match_title: str = "",
        entry_delay_s: float = 0.0,
    ):
        """Register a new paper trade."""
        # ═══ GLOBAL EDGE GUARD (DISABLED v2.0 — paper mode, edge=0) ═══
        # In paper mode edge is always 0.0 (no model), so this guard
        # blocked every trade. Disabled for paper trading.
        # from sports.guards import validate_trade_execution
        # can_exec, block_reason = validate_trade_execution(
        #     edge=edge, price=entry_price, sport="cricket",
        #     context=f"{signal_type} | {match_id}",
        # )
        # if not can_exec:
        #     return

        trade = CricketPaperTrade(
            match_id=match_id,
            selection_id=selection_id,
            signal_type=signal_type,
            entry_price=entry_price,
            fair_value_entry=fair_value,
            edge_entry=edge,
            entry_timestamp=time.time(),
            entry_score=entry_score,
            match_title=match_title,
            entry_delay_s=entry_delay_s,
            spread_at_entry=spread,
            peak_price=entry_price,
            min_price_seen=entry_price,
        )
        self.open_trades[match_id] = trade
        self.stats.total_entries += 1
        log.info("CRICKET EXIT MGR: registered trade %s | %s | entry=%.4f edge=%.4f",
                 match_id, signal_type, entry_price, edge)

    def check_all(self, books: dict, match_states: dict = None):
        """Check all open trades for snapshots, MAE/MFE updates, and exits.

        Args:
            books: {token_id: BookState} from PolymarketFeed
            match_states: {match_id: CricketState} optional, for score tracking
        """
        to_close = []
        for match_id, trade in self.open_trades.items():
            book = books.get(trade.selection_id)

            # STEP 7: Synthetic price fallback for paper mode
            if book and book.mid > 0:
                mkt = book.mid
                exit_spread = book.spread
            else:
                mkt = trade.entry_price  # synthetic: use entry price
                exit_spread = 0.01
            now = time.time()
            elapsed = now - trade.entry_timestamp

            # ── Post-entry snapshots ──────────────────────────────
            if elapsed >= 5 and not trade._snapshot_5_done:
                trade.price_t5 = mkt
                trade._snapshot_5_done = True
            if elapsed >= 15 and not trade._snapshot_15_done:
                trade.price_t15 = mkt
                trade._snapshot_15_done = True
            if elapsed >= 30 and not trade._snapshot_30_done:
                trade.price_t30 = mkt
                trade._snapshot_30_done = True
            if elapsed >= 60 and not trade._snapshot_60_done:
                trade.price_t60 = mkt
                trade._snapshot_60_done = True

            # ── MAE/MFE tracking ──────────────────────────────────
            favorable = mkt - trade.entry_price
            adverse = trade.entry_price - mkt

            if favorable > trade.mfe:
                trade.mfe = favorable
            if adverse > trade.mae:
                trade.mae = adverse
            if mkt > trade.peak_price:
                trade.peak_price = mkt
            if mkt < trade.min_price_seen or trade.min_price_seen == 0:
                trade.min_price_seen = mkt

            # ── Exit conditions ───────────────────────────────────
            # 1. Stop loss (MAE-based)
            adverse_ticks = adverse * 100
            if adverse_ticks >= self.STOP_LOSS_TICKS:
                to_close.append((match_id, mkt, "STOP_LOSS",
                                 self._score_str(match_states, match_id), exit_spread))
                continue

            # 2. Timeout
            if trade.age_minutes >= self.TIMEOUT_MINUTES:
                to_close.append((match_id, mkt, "TIMEOUT",
                                 self._score_str(match_states, match_id), exit_spread))
                continue

            # 3. Runner mode — trailing stop after big gain
            price_gain = mkt - trade.entry_price
            if price_gain >= self.RUNNER_GAIN:
                trail_distance = max(0.03,
                    (trade.peak_price - trade.entry_price) * self.RUNNER_TRAIL_PCT)
                if mkt <= trade.peak_price - trail_distance:
                    to_close.append((match_id, mkt, "RUNNER_TRAIL",
                                     self._score_str(match_states, match_id), exit_spread))
                    continue

        # ── Close trades ──────────────────────────────────────────
        for match_id, exit_price, reason, exit_score, exit_spread in to_close:
            self._close_trade(match_id, exit_price, reason, exit_score, exit_spread)

    def close_match_end(self, match_id: str, final_price: float, final_score: str = ""):
        """Close trade when match ends."""
        if match_id in self.open_trades:
            self._close_trade(match_id, final_price, "MATCH_END", final_score, 0.0)

    def _close_trade(self, match_id: str, exit_price: float, reason: str,
                     exit_score: str, exit_spread: float):
        trade = self.open_trades.pop(match_id, None)
        if not trade:
            return
        trade.exit_price = exit_price
        trade.exit_timestamp = time.time()
        trade.exit_reason = reason
        trade.exit_score = exit_score
        trade.spread_at_exit = exit_spread
        trade.is_open = False
        self.closed_trades.append(trade)
        self.stats.total_exits += 1

        pnl = trade.paper_pnl
        if pnl > 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1
        self.stats.total_pnl += pnl

        log.info("CRICKET EXIT | %s | %s | entry=%.4f exit=%.4f pnl=%+.4f | "
                 "MAE=%.4f MFE=%.4f | hold=%.0fs | %s",
                 match_id, reason, trade.entry_price, exit_price, pnl,
                 trade.mae, trade.mfe, trade.duration_seconds, trade.match_title[:30])

        self._write_csv(trade)

        # Fire callback for live execution (exit sell)
        if self._on_close:
            try:
                self._on_close(trade)
            except Exception as e:
                log.error("CRICKET EXIT on_close callback error: %s", e)

        return pnl

    def _score_str(self, match_states: dict, match_id: str) -> str:
        if not match_states:
            return ""
        state = match_states.get(match_id)
        return str(state) if state else ""

    def _write_csv(self, t: CricketPaperTrade):
        """Append trade to CSV."""
        header = [
            "match_id", "signal_type", "match_title",
            "entry_price", "fair_value", "edge",
            "entry_timestamp", "entry_delay_s",
            "price_t5", "price_t15", "price_t30", "price_t60",
            "peak_price", "min_price_seen", "mfe", "mae",
            "spread_at_entry", "spread_at_exit",
            "exit_price", "exit_reason", "exit_timestamp",
            "duration_s", "paper_pnl",
            "entry_score", "exit_score",
        ]
        row = [
            t.match_id, t.signal_type, t.match_title,
            f"{t.entry_price:.4f}", f"{t.fair_value_entry:.4f}", f"{t.edge_entry:.4f}",
            f"{t.entry_timestamp:.1f}", f"{t.entry_delay_s:.1f}",
            f"{t.price_t5:.4f}" if t.price_t5 else "",
            f"{t.price_t15:.4f}" if t.price_t15 else "",
            f"{t.price_t30:.4f}" if t.price_t30 else "",
            f"{t.price_t60:.4f}" if t.price_t60 else "",
            f"{t.peak_price:.4f}", f"{t.min_price_seen:.4f}",
            f"{t.mfe:.4f}", f"{t.mae:.4f}",
            f"{t.spread_at_entry:.4f}", f"{t.spread_at_exit:.4f}",
            f"{t.exit_price:.4f}" if t.exit_price else "",
            t.exit_reason or "", f"{t.exit_timestamp:.1f}" if t.exit_timestamp else "",
            f"{t.duration_seconds:.0f}", f"{t.paper_pnl:.4f}",
            t.entry_score, t.exit_score or "",
        ]
        try:
            with open(self._csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if not self._csv_written_header:
                    w.writerow(header)
                    self._csv_written_header = True
                w.writerow(row)
        except Exception as e:
            log.error("CRICKET EXIT CSV: %s", e)

    def summary(self) -> str:
        """One-line summary for STATUS log."""
        open_count = len(self.open_trades)
        closed = len(self.closed_trades)
        if closed == 0:
            return f"CricketExit: {open_count} open, 0 closed"
        wr = self.stats.wins / closed if closed else 0
        return (f"CricketExit: {open_count} open, {closed} closed, "
                f"WR={wr:.0%}, PnL={self.stats.total_pnl:+.4f}")


class _CricketExitStats:
    def __init__(self):
        self.total_entries = 0
        self.total_exits = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
"""
    # This is a module-level attribute accessible from the exit stats
"""
