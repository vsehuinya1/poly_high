"""
Signal engine + paper trading.
Matches game state to Polymarket book state, detects edges,
runs paper trades, logs everything to CSV, sends Telegram alerts.
"""
import asyncio
import csv
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from sports.config import (
    ENTRY_EDGE_THRESHOLD, EXIT_CONVERGENCE,
    MAX_POSITION_PER_MARKET, MAX_CONCURRENT_POSITIONS, MAX_DAILY_LOSS,
    DATA_DIR,
    # Execution hygiene (v3.4 — controlled participation)
    PRICE_BAND_LO, PRICE_BAND_HI, MAX_SPREAD, MAX_BOOK_AGE_S, MAX_SCORE_DIFF,
    SELL_PRICE_BAND_LO, SELL_PRICE_BAND_HI, BUY_PRICE_BAND_LO, BUY_PRICE_BAND_HI,
    EDGE_TRADE_THRESHOLD, MAX_ELAPSED_PCT,
    LATE_GAME_HARD_STOP_NBA, LATE_GAME_HARD_STOP_FB,
    NBA_TRADE_WINDOW_START, NBA_TRADE_WINDOW_END,
    FB_TRADE_WINDOW_START, FB_TRADE_WINDOW_END,
    MAX_POS_PER_DIRECTION, SELL_ONLY_MODE,
    GATE_FRESH_THRESHOLD, GATE_STREAK_S, GATE_ROLLING_WINDOW_S, GATE_ROLLING_FRESH_PCT,
    FREEZE_STALE_THRESHOLD, FREEZE_STALE_DURATION_S, UNFREEZE_STREAK_S,
    COOLDOWN_S, PER_GAME_STOP,
    # Execution stability (v4.0)
    MIN_HOLD_S, EDGE_CONFIRM_TICKS, MAX_TRADES_PER_GAME,
    POST_EXIT_COOLDOWN_S, STOP_LOSS_TICKS, EDGE_FLIP_THRESHOLD,
    ENTRY_MAX_SPREAD, ENTRY_MAX_BOOK_AGE_S,
    # Football risk controls (v4.3)
    FOOTBALL_STOP_LOSS_TICKS, FOOTBALL_FAST_MOVE_TICKS,
    FOOTBALL_FAST_MOVE_S, FOOTBALL_TIMEOUT_S, DEFAULT_TIMEOUT_S,
)
from sports.feeds import GameState, BookState
from sports.models import (
    nba_win_probability, football_win_probability,
    compute_edge, ModelOutput, EdgeSignal,
)
from sports.telegram import TelegramNotifier

log = logging.getLogger("sports.engine")


# ═══════════════════════════════════════════════════════════════════════
#  Game-Market Mapping
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GameMarketLink:
    """Links a live game to its Polymarket tokens."""
    game_id: str
    sport: str
    league: str
    home_team: str
    away_team: str
    polymarket_event_id: str
    polymarket_title: str
    polymarket_slug: str
    home_token_id: str = ""
    away_token_id: str = ""
    draw_token_id: str = ""
    all_token_ids: list[str] = field(default_factory=list)
    pregame_home_prob: float = 0.5
    pregame_draw_prob: float = 0.0
    pregame_away_prob: float = 0.0
    lambda_home: Optional[float] = None   # pre-warmed via invert_1x2_to_lambdas
    lambda_away: Optional[float] = None   # pre-warmed via invert_1x2_to_lambdas


# ═══════════════════════════════════════════════════════════════════════
#  Paper Trading Position
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PaperPosition:
    """Simulated position in a Polymarket outcome."""
    position_id: str
    game_id: str
    token_id: str
    outcome: str
    direction: str
    entry_price: float
    entry_edge: float
    entry_time: float
    entry_game_state: str
    market_title: str = ""
    size_usd: float = 100.0
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    is_open: bool = True
    # v4.0 stability fields
    edge_at_exit: float = 0.0
    stop_loss_triggered: bool = False
    # v4.3 sport + MAE/MFE tracking
    sport: str = ""
    mae_ticks: float = 0.0
    mfe_ticks: float = 0.0
    time_to_mae_s: float = 0.0
    time_to_mfe_s: float = 0.0
    momentum_exit_triggered: bool = False


# ═══════════════════════════════════════════════════════════════════════
#  CSV Data Storage
# ═══════════════════════════════════════════════════════════════════════

class CSVLogger:
    """Writes structured data to CSV files with auto-rotation."""

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

    def log_snapshot(self, row: dict):
        headers = [
            "timestamp", "game_id", "home_score", "away_score", "period", "elapsed",
            "home_p_model", "away_p_model", "home_p_mkt", "away_p_mkt", "edge",
            # New columns
            "adjusted_seconds", "sigma", "strength_adjustment", "s_eff", "z", "pregame_probability"
        ]
        writer = self._get_writer("snapshots", headers)
        writer.writerow([row.get(h, "") for h in headers])

    def log_edge_signal(self, signal: EdgeSignal):
        headers = [
            "timestamp", "game_id", "sport", "market_title",
            "token_id", "outcome", "model_prob", "market_prob",
            "edge", "confidence", "direction", "game_state",
        ]
        w = self._get_writer("signals", headers)
        w.writerow([
            signal.timestamp, signal.game_id, signal.sport,
            signal.market_title, signal.token_id, signal.outcome,
            f"{signal.model_prob:.4f}", f"{signal.market_prob:.4f}",
            f"{signal.edge:.4f}", f"{signal.confidence:.4f}",
            signal.direction, signal.game_state,
        ])

    def log_trade(self, pos: PaperPosition, event: str,
                  edge_confirm_count: int = 0, game_trade_count: int = 0):
        headers = [
            "timestamp", "event", "position_id", "game_id",
            "token_id", "outcome", "direction",
            "entry_price", "exit_price", "entry_edge",
            "size_usd", "pnl", "exit_reason", "game_state",
            # v4.0 stability fields
            "hold_duration_s", "edge_at_exit", "stop_loss_triggered",
            "edge_confirm_count", "game_trade_count",
            # v4.3 football diagnostics
            "sport", "mae_ticks", "mfe_ticks",
            "time_to_mae_s", "time_to_mfe_s", "momentum_exit_triggered",
        ]
        hold_dur = ""
        if pos.exit_time and pos.entry_time:
            hold_dur = f"{pos.exit_time - pos.entry_time:.2f}"
        w = self._get_writer("paper_trades", headers)
        w.writerow([
            time.time(), event, pos.position_id, pos.game_id,
            pos.token_id, pos.outcome, pos.direction,
            f"{pos.entry_price:.4f}",
            f"{pos.exit_price:.4f}" if pos.exit_price else "",
            f"{pos.entry_edge:.4f}",
            f"{pos.size_usd:.2f}",
            f"{pos.pnl:.2f}" if pos.pnl else "",
            pos.exit_reason, pos.entry_game_state,
            hold_dur,
            f"{pos.edge_at_exit:.4f}" if pos.edge_at_exit else "",
            "1" if pos.stop_loss_triggered else "0",
            edge_confirm_count,
            game_trade_count,
            pos.sport,
            f"{pos.mae_ticks:.1f}",
            f"{pos.mfe_ticks:.1f}",
            f"{pos.time_to_mae_s:.1f}",
            f"{pos.time_to_mfe_s:.1f}",
            "1" if pos.momentum_exit_triggered else "0",
        ])

    def log_book_update(self, token_id: str, book: BookState):
        headers = [
            "timestamp", "token_id", "best_bid", "best_ask",
            "mid", "spread", "bid_size", "ask_size",
            "last_trade_price", "last_trade_size",
        ]
        w = self._get_writer("book_updates", headers)
        w.writerow([
            book.timestamp, token_id,
            f"{book.best_bid:.4f}", f"{book.best_ask:.4f}",
            f"{book.mid:.4f}", f"{book.spread:.4f}",
            f"{book.bid_size:.2f}", f"{book.ask_size:.2f}",
            f"{book.last_trade_price:.4f}", f"{book.last_trade_size:.2f}",
        ])

    def close(self):
        for fh in self._handles.values():
            try:
                fh.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════
#  Game Trade State — per-game execution hygiene tracking
# ═══════════════════════════════════════════════════════════════════════

class GameStatus(Enum):
    INACTIVE = "INACTIVE"   # gate not yet passed
    ACTIVE   = "ACTIVE"     # gate passed, trading allowed
    FROZEN   = "FROZEN"     # temporarily frozen (stale book)
    STOPPED  = "STOPPED"    # per-game stop triggered


class GameTradeState:
    """Per-game execution hygiene state."""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.status = GameStatus.INACTIVE
        self.pnl: float = 0.0
        self.trade_count: int = 0
        self.entry_count: int = 0       # v4.0: counts entries (trade_count counts exits)
        self.last_exit_time: float = 0.0
        self.band_rejects: int = 0

        # Per-direction tracking (v3.4)
        self._last_exit_time_by_dir: dict[str, float] = {}  # {"BUY": ts, "SELL": ts}

        # Gate tracking
        self._fresh_streak_start: float = 0.0  # when current fresh streak began
        self._rolling_ticks: deque = deque()    # (timestamp, is_fresh_20s) pairs

        # Freeze tracking
        self._stale_streak_start: float = 0.0   # when current stale streak began
        self._unfreeze_fresh_start: float = 0.0 # when fresh streak began during freeze

    def record_direction_exit(self, direction: str, now: float):
        """Record exit time for a specific direction (for per-direction cooldown)."""
        self._last_exit_time_by_dir[direction] = now

    def direction_cooldown_remaining(self, direction: str, now: float) -> float:
        """Seconds remaining on cooldown for a specific direction."""
        last = self._last_exit_time_by_dir.get(direction, 0)
        if last == 0:
            return 0
        return max(0, COOLDOWN_S - (now - last))

    def update_gate(self, book_age: float, now: float) -> str | None:
        """Update game activation gate. Returns state transition or None."""
        if self.status == GameStatus.STOPPED:
            return None

        is_fresh_30 = book_age <= GATE_FRESH_THRESHOLD
        is_fresh_20 = book_age <= MAX_BOOK_AGE_S

        # Update rolling window
        self._rolling_ticks.append((now, is_fresh_20))
        cutoff = now - GATE_ROLLING_WINDOW_S
        while self._rolling_ticks and self._rolling_ticks[0][0] < cutoff:
            self._rolling_ticks.popleft()

        # Track fresh streak (age <= 30s)
        if is_fresh_30:
            if self._fresh_streak_start == 0:
                self._fresh_streak_start = now
        else:
            self._fresh_streak_start = 0

        fresh_streak_s = (now - self._fresh_streak_start) if self._fresh_streak_start > 0 else 0

        # Rolling fresh %
        if self._rolling_ticks:
            fresh_count = sum(1 for _, f in self._rolling_ticks if f)
            rolling_pct = fresh_count / len(self._rolling_ticks)
        else:
            rolling_pct = 0

        # Gate activation check
        gate_pass = (fresh_streak_s >= GATE_STREAK_S and rolling_pct >= GATE_ROLLING_FRESH_PCT)

        old_status = self.status
        if self.status == GameStatus.INACTIVE:
            if gate_pass:
                self.status = GameStatus.ACTIVE
                return "GAME_ACTIVATED"
        elif self.status == GameStatus.ACTIVE:
            if not gate_pass:
                self.status = GameStatus.INACTIVE
                return "GAME_DEACTIVATED"

        return None

    def update_freeze(self, book_age: float, now: float) -> str | None:
        """Update intra-game freeze state. Returns state transition or None."""
        if self.status == GameStatus.STOPPED:
            return None

        is_stale = book_age > FREEZE_STALE_THRESHOLD
        is_fresh_30 = book_age <= GATE_FRESH_THRESHOLD

        if self.status == GameStatus.ACTIVE:
            # Check if we should freeze
            if is_stale:
                if self._stale_streak_start == 0:
                    self._stale_streak_start = now
                elif (now - self._stale_streak_start) >= FREEZE_STALE_DURATION_S:
                    self.status = GameStatus.FROZEN
                    self._stale_streak_start = 0
                    self._unfreeze_fresh_start = 0
                    return "GAME_FROZEN"
            else:
                self._stale_streak_start = 0

        elif self.status == GameStatus.FROZEN:
            # Check if we should unfreeze
            if is_fresh_30:
                if self._unfreeze_fresh_start == 0:
                    self._unfreeze_fresh_start = now
                elif (now - self._unfreeze_fresh_start) >= UNFREEZE_STREAK_S:
                    self.status = GameStatus.ACTIVE
                    self._stale_streak_start = 0
                    self._unfreeze_fresh_start = 0
                    return "GAME_UNFROZEN"
            else:
                self._unfreeze_fresh_start = 0

        return None

    def record_exit(self, pnl: float, now: float) -> str | None:
        """Record a trade exit. Returns 'GAME_STOPPED' if stop triggered."""
        self.pnl += pnl
        self.trade_count += 1
        self.last_exit_time = now
        if self.pnl <= -PER_GAME_STOP and self.status != GameStatus.STOPPED:
            self.status = GameStatus.STOPPED
            return "GAME_STOPPED"
        return None

    def cooldown_remaining(self, now: float) -> float:
        """Seconds remaining in cooldown after last exit."""
        if self.last_exit_time == 0:
            return 0.0
        elapsed = now - self.last_exit_time
        return max(0.0, COOLDOWN_S - elapsed)

    @property
    def can_trade(self) -> bool:
        return self.status == GameStatus.ACTIVE


# ═══════════════════════════════════════════════════════════════════════
#  Signal Engine
# ═══════════════════════════════════════════════════════════════════════

class SignalEngine:
    """
    Core engine: matches game states to market states,
    computes fair value, detects edges, runs paper trades.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.csv = CSVLogger(data_dir)
        self.tg = TelegramNotifier()
        self.links: dict[str, GameMarketLink] = {}
        self.positions: dict[str, PaperPosition] = {}
        self.daily_pnl: float = 0.0
        self._position_counter = 0
        self._killed = False
        self._signal_count = 0
        self._finished_games: set[str] = set()
        self._last_states: dict[str, GameState] = {}
        self._game_states: dict[str, GameTradeState] = {}  # per-game hygiene

        # Block counters (v3.4 — daily summary stats)
        self._blocks = {
            "BLOCK_PRICE": 0,
            "BLOCK_EDGE": 0,
            "BLOCK_SPREAD": 0,
            "BLOCK_TIME": 0,
            "BLOCK_POSITION_LIMIT": 0,
            "BLOCK_COOLDOWN": 0,
            "BLOCK_BOOK_AGE": 0,
            "BLOCK_SCORE_DIFF": 0,
            "BLOCK_DIRECTION": 0,
            # v4.0 stability patch counters
            "BLOCK_GAME_TRADE_LIMIT": 0,
            "BLOCK_EDGE_CONFIRM": 0,
            "BLOCK_ENTRY_SPREAD": 0,
            "BLOCK_ENTRY_FRESHNESS": 0,
        }
        self._trades_taken = 0
        self._last_summary_ts = 0.0
        # Rate-limited gate logging: {(game_id, gate): last_log_ts}
        self._gate_log_times: dict[tuple, float] = {}
        # Rate-limited BLOCK_TIME logging (v3.8): {key: last_log_ts}
        self._block_time_log: dict[str, float] = {}
        # v4.0: edge confirmation counters: {(game_id, direction): consecutive_count}
        self._edge_confirm: dict[tuple[str, str], int] = {}
        # v4.0: timestamp of last edge signal per (game_id, direction) for staleness detection
        self._edge_confirm_last_ts: dict[tuple[str, str], float] = {}

    def _get_game_state(self, game_id: str) -> GameTradeState:
        if game_id not in self._game_states:
            self._game_states[game_id] = GameTradeState(game_id)
        return self._game_states[game_id]

    def _log_gate_block(self, game_id: str, gate: str, signal,
                        gs_str: str, detail: str):
        """Rate-limited gate block logging — once per match per gate per 60s."""
        now = time.time()
        key = (game_id, gate)
        last = self._gate_log_times.get(key, 0)
        if now - last < 60.0:
            return
        self._gate_log_times[key] = now
        log.info("GATE_BLOCK | %s | %s %s | edge=%.3f mkt=%.3f | %s",
                 gate, signal.direction, signal.outcome,
                 signal.edge, signal.market_prob, detail)

    def log_daily_summary(self):
        """Log daily block/trade summary stats."""
        now = time.time()
        if now - self._last_summary_ts < 300:  # every 5 min
            return
        self._last_summary_ts = now
        total_blocked = sum(self._blocks.values())
        if total_blocked == 0 and self._trades_taken == 0:
            return

        # Per-game trade counts
        game_trades = {}
        for p in self.positions.values():
            game_trades[p.game_id] = game_trades.get(p.game_id, 0) + 1

        # Avg entry price and edge of taken trades
        taken = [p for p in self.positions.values()]
        avg_price = sum(p.entry_price for p in taken) / len(taken) if taken else 0
        avg_edge = sum(abs(p.entry_edge) for p in taken) / len(taken) if taken else 0

        log.info(
            "DAILY_SUMMARY | taken=%d | blocked=%d | %s | "
            "games=%s | avg_price=%.3f avg_edge=%.3f",
            self._trades_taken, total_blocked,
            " ".join(f"{k}={v}" for k, v in self._blocks.items() if v > 0),
            dict(game_trades) if game_trades else "{}",
            avg_price, avg_edge,
        )

    def register_link(self, link: GameMarketLink):
        self.links[link.polymarket_event_id] = link
        log.info("registered link: %s ↔ %s (game_id=%s)",
                 link.polymarket_title, link.home_team, link.game_id)

    async def process_tick(
        self,
        game_state: Optional[GameState],
        books: dict[str, BookState],
        link: GameMarketLink,
    ) -> list[EdgeSignal]:
        if self._killed:
            return []

        if game_state is None:
            return []

        self._last_states[game_state.game_id] = game_state # Store last game state

        # Handle game-end notifications
        if game_state.game_id not in self._finished_games:
            if game_state.status == "finished" or (
                game_state.sport == "nba" and game_state.elapsed_minutes >= 48
                and game_state.status not in ("scheduled", "live", "Q1", "Q2", "Q3", "Q4")
            ):
                self._finished_games.add(game_state.game_id)
                await self._handle_game_end(game_state, link)
                return []

        if not game_state.is_live:
            return []

        now = time.time()

        # ── Compute model fair value ──────────────────────────────
        if game_state.sport == "nba":
            # Local clock decay between polls
            elapsed = time.time() - game_state.timestamp
            seconds_remaining = (game_state.total_minutes - game_state.elapsed_minutes) * 60.0
            adj_seconds = max(0, seconds_remaining - elapsed)
            
            model = nba_win_probability(
                game_state.home_score, game_state.away_score,
                adj_seconds, game_state.period, link.pregame_home_prob
            )
        else:
            if link.lambda_home is None or link.lambda_away is None:
                raise RuntimeError(
                    f"Football λ not pre-warmed for {link.polymarket_title} "
                    f"(game_id={link.game_id}). Run prewarm_football_lambdas() "
                    f"before starting live polling."
                )
            model = football_win_probability(
                game_state.home_score, game_state.away_score,
                game_state.minutes_remaining,
                game_state.home_red_cards, game_state.away_red_cards,
                lambda_home_90=link.lambda_home,
                lambda_away_90=link.lambda_away,
            )
            adj_seconds = game_state.minutes_remaining * 60.0

        # ── Get market prices ─────────────────────────────────────
        home_book = books.get(link.home_token_id)
        away_book = books.get(link.away_token_id)
        draw_book = books.get(link.draw_token_id) if link.draw_token_id else None

        home_mid = home_book.mid if home_book and home_book.mid > 0 else 0
        away_mid = away_book.mid if away_book and away_book.mid > 0 else 0
        draw_mid = draw_book.mid if draw_book and draw_book.mid > 0 else 0

        if home_mid == 0 and away_mid == 0:
            return []

        # ── Compute edges ─────────────────────────────────────────
        edges = compute_edge(model, home_mid, away_mid, draw_mid)
        
        # Log snapshot with new fields
        # edge: log the single largest absolute edge for this tick
        max_edge = max(edges, key=lambda x: abs(x[3]))[0:4] if edges else ("none", 0, 0, 0)
        self.csv.log_snapshot({
            "timestamp": time.time(),
            "game_id": game_state.game_id,
            "home_score": game_state.home_score,
            "away_score": game_state.away_score,
            "period": game_state.period,
            "elapsed": game_state.elapsed_minutes,
            "home_p_model": model.p_home,
            "away_p_model": model.p_away,
            "home_p_mkt": home_mid,
            "away_p_mkt": away_mid,
            "edge": max_edge[3],
            "adjusted_seconds": adj_seconds,
            "sigma": model.sigma,
            "strength_adjustment": model.strength_adjustment,
            "s_eff": model.s_eff,
            "z": model.z,
            "pregame_probability": link.pregame_home_prob
        })

        # Full state snapshot — one line per tick for debugging
        hb = home_book
        book_age = now - hb.timestamp if hb and hb.timestamp > 0 else -1

        # ── Update game trade state (gate + freeze) ───────────────
        gts = self._get_game_state(game_state.game_id)

        gate_transition = gts.update_gate(book_age, now)
        if gate_transition:
            log.info("%s: %s (game_id=%s)", gate_transition,
                     link.polymarket_title, game_state.game_id)

        freeze_transition = gts.update_freeze(book_age, now)
        if freeze_transition:
            log.info("%s: %s (game_id=%s)", freeze_transition,
                     link.polymarket_title, game_state.game_id)

        cd_remain = gts.cooldown_remaining(now)

        log.info(
            "SNAP %s %d-%d | adj=%3.0f σ=%.2f seff=%.1f z=%.2f | "
            "model=%.3f | bid=%.3f ask=%.3f mid=%.3f | "
            "bsz=%.0f asz=%.0f sprd=%.3f age=%.0fs | edge=%+.4f | "
            "gs=%s cd=%.0f gpnl=$%.0f",
            game_state.home_team[:3],
            game_state.home_score, game_state.away_score,
            adj_seconds, model.sigma, model.s_eff, model.z,
            model.p_home,
            hb.best_bid if hb else 0, hb.best_ask if hb else 0,
            home_mid,
            hb.bid_size if hb else 0, hb.ask_size if hb else 0,
            hb.spread if hb else 1, book_age,
            max_edge[3] if edges else 0,
            gts.status.value, cd_remain, gts.pnl,
        )

        # ── Detect edge signals ───────────────────────────────────
        signals = []
        game_state_str = (
            f"{game_state.home_team} {game_state.home_score}-"
            f"{game_state.away_score} {game_state.away_team} "
            f"({game_state.elapsed_minutes:.0f}')"
        )

        for outcome, model_p, market_mid, edge in edges:
            token_id = ""
            if outcome == "home": token_id = link.home_token_id
            elif outcome == "away": token_id = link.away_token_id
            elif outcome == "draw": token_id = link.draw_token_id

            if not token_id or market_mid <= 0:
                continue

            # Signal threshold (ENTRY_EDGE_THRESHOLD=0.05) is looser than
            # trade filter (0.07) — signals are informational, trades are gated tighter
            if abs(edge) >= ENTRY_EDGE_THRESHOLD:
                direction = "BUY" if edge > 0 else "SELL"
                signal = EdgeSignal(
                    timestamp=now,
                    game_id=game_state.game_id,
                    sport=game_state.sport,
                    market_title=link.polymarket_title,
                    token_id=token_id,
                    outcome=outcome,
                    model_prob=model_p,
                    market_prob=market_mid,
                    edge=edge,
                    confidence=model.confidence,
                    game_state=game_state_str,
                    direction=direction,
                )
                signals.append(signal)
                self.csv.log_edge_signal(signal)
                self._signal_count += 1
                log.info("EDGE: %s %s %.3f (model=%.3f mkt=%.3f) %s",
                         direction, outcome, edge, model_p, market_mid,
                         game_state_str)

                # Telegram alert for edge signals — DISABLED due to spam/rate limits
                # await self.tg.notify_edge_signal(
                #     direction, outcome, edge, model_p, market_mid,
                #     game_state_str, link.polymarket_title,
                # )

        # ── Deduplicate signals — keep best per direction ────────
        # v3.6: direction-aware dedup. Previously kept only 1 signal
        # (highest abs edge), which discarded the SELL signal when
        # BUY had the same magnitude — then SELL_ONLY_MODE killed
        # the BUY → zero trades. Now we keep best BUY + best SELL.
        if signals:
            best_by_dir: dict[str, EdgeSignal] = {}
            for s in signals:
                if s.direction not in best_by_dir or abs(s.edge) > abs(best_by_dir[s.direction].edge):
                    best_by_dir[s.direction] = s
            signals = list(best_by_dir.values())

        # ── Paper trading logic ───────────────────────────────────
        for sig in signals:
            await self._evaluate_paper_trade(sig, books, link)

        # ── Check existing positions for exit ─────────────────────
        await self._check_exits(model, books, link, game_state)

        return signals

    async def _evaluate_paper_trade(self, signal: EdgeSignal,
                                     books: dict[str, BookState],
                                     link: GameMarketLink):
        if self.daily_pnl <= -MAX_DAILY_LOSS:
            if not self._killed:
                log.warning("KILL SWITCH: daily PnL %.2f ≤ -%.0f",
                           self.daily_pnl, MAX_DAILY_LOSS)
                self._killed = True
            return

        open_count = sum(1 for p in self.positions.values() if p.is_open)
        if open_count >= MAX_CONCURRENT_POSITIONS:
            return

        # Duplicate token check
        for p in self.positions.values():
            if p.is_open and p.token_id == signal.token_id:
                return

        book = books.get(signal.token_id)
        if not book:
            return

        now = time.time()
        game_state = self._last_states.get(link.game_id)
        if not game_state:
            return

        gts = self._get_game_state(link.game_id)
        gs_str = signal.game_state  # for log messages

        # ══════════════════════════════════════════════════════════
        # CONTROLLED PARTICIPATION GATES (v3.4)
        # Each gate logs its block reason for daily summary.
        # ══════════════════════════════════════════════════════════

        elapsed = now - game_state.timestamp
        adj_sec = max(0, (game_state.total_minutes - game_state.elapsed_minutes) * 60.0 - elapsed)
        book_age = now - book.timestamp if book.timestamp > 0 else 9999
        score_diff = abs(game_state.home_score - game_state.away_score)

        # 0. Trade window gate
        if game_state.sport == "nba":
            # NBA: trade only between minute 18 and minute 36 elapsed
            # adj_sec is seconds REMAINING: 720 (12min left) ≤ adj_sec < 1800 (30min left)
            if not (NBA_TRADE_WINDOW_START <= adj_sec < NBA_TRADE_WINDOW_END):
                return
        elif game_state.sport == "football":
            # Football: trade only between minute 15 and minute 70 elapsed
            # adj_sec is seconds REMAINING: 1200 (20min left) ≤ adj_sec < 4500 (75min left)
            if not (FB_TRADE_WINDOW_START <= adj_sec < FB_TRADE_WINDOW_END):
                return
        else:
            # Other sports: original fallback
            if adj_sec > 600:
                return

        # 1. BLOCK_TIME — late-game hard stop + percentage gate
        hard_stop = LATE_GAME_HARD_STOP_NBA if game_state.sport == "nba" else LATE_GAME_HARD_STOP_FB
        if game_state.elapsed_minutes > hard_stop:
            self._blocks["BLOCK_TIME"] += 1
            # Rate-limit log: 1 per game per 60s (v3.8)
            bt_key = f"bt_{signal.game_id}"
            bt_last = self._block_time_log.get(bt_key, 0)
            if now - bt_last >= 60:
                log.info("BLOCK_TIME | %s | min=%.0f > hard_stop=%.0f",
                         gs_str, game_state.elapsed_minutes, hard_stop)
                self._block_time_log[bt_key] = now
            return
        elapsed_pct = game_state.elapsed_minutes / game_state.total_minutes if game_state.total_minutes > 0 else 1.0
        if elapsed_pct > MAX_ELAPSED_PCT:
            self._blocks["BLOCK_TIME"] += 1
            bt_key = f"bt_pct_{signal.game_id}"
            bt_last = self._block_time_log.get(bt_key, 0)
            if now - bt_last >= 60:
                log.info("BLOCK_TIME | %s | elapsed=%.0f%% > %.0f%%",
                         gs_str, elapsed_pct * 100, MAX_ELAPSED_PCT * 100)
                self._block_time_log[bt_key] = now
            return

        # 2. BLOCK_EDGE
        if abs(signal.edge) < EDGE_TRADE_THRESHOLD:
            self._blocks["BLOCK_EDGE"] += 1
            self._log_gate_block(link.game_id, "BLOCK_EDGE", signal, gs_str,
                                 f"edge={abs(signal.edge):.3f} < {EDGE_TRADE_THRESHOLD}")
            return

        # 2b. BLOCK_DIRECTION — SELL-only mode (v3.5)
        if SELL_ONLY_MODE and signal.direction == "BUY":
            self._blocks["BLOCK_DIRECTION"] += 1
            self._log_gate_block(link.game_id, "BLOCK_DIRECTION", signal, gs_str,
                                 "SELL_ONLY_MODE blocks BUY")
            return

        # 3. BLOCK_PRICE — on market mid (v3.8: direction-specific bands)
        if signal.direction == "SELL":
            band_lo, band_hi = SELL_PRICE_BAND_LO, SELL_PRICE_BAND_HI
        else:
            band_lo, band_hi = BUY_PRICE_BAND_LO, BUY_PRICE_BAND_HI
        if not (band_lo <= signal.market_prob <= band_hi):
            self._blocks["BLOCK_PRICE"] += 1
            log.info("BLOCK_PRICE | %s | %s mid=%.3f | band=[%.2f,%.2f]",
                     gs_str, signal.direction, signal.market_prob, band_lo, band_hi)
            return

        # 4. BLOCK_SPREAD
        if book.spread > MAX_SPREAD:
            self._blocks["BLOCK_SPREAD"] += 1
            log.info("BLOCK_SPREAD | %s | spread=%.3f > %.3f",
                     gs_str, book.spread, MAX_SPREAD)
            return

        # 5. BLOCK_BOOK_AGE
        if book_age > MAX_BOOK_AGE_S:
            self._blocks["BLOCK_BOOK_AGE"] += 1
            self._log_gate_block(link.game_id, "BLOCK_BOOK_AGE", signal, gs_str,
                                 f"age={book_age:.0f}s > {MAX_BOOK_AGE_S}s")
            return

        # 6. BLOCK_SCORE_DIFF
        if score_diff > MAX_SCORE_DIFF:
            self._blocks["BLOCK_SCORE_DIFF"] += 1
            return

        # 7. Game must be ACTIVE (gate passed)
        #    v3.8: NBA bypasses activation gate — window gate is sufficient
        if game_state.sport != "nba":
            if not gts.can_trade:
                self._log_gate_block(link.game_id, "BLOCK_GAME_STATE", signal, gs_str,
                                     f"gs={gts.status.value} (need ACTIVE)")
                return
            if gts.status == GameStatus.FROZEN:
                self._log_gate_block(link.game_id, "BLOCK_GAME_STATE", signal, gs_str,
                                     "gs=FROZEN")
                return

        # 8. BLOCK_POSITION_LIMIT — 1 per direction per game
        dir_open = sum(1 for p in self.positions.values()
                       if p.is_open and p.game_id == signal.game_id
                       and p.direction == signal.direction)
        if dir_open >= MAX_POS_PER_DIRECTION:
            self._blocks["BLOCK_POSITION_LIMIT"] += 1
            log.info("BLOCK_POSITION_LIMIT | %s | %s already open in game %s",
                     gs_str, signal.direction, signal.game_id)
            return

        # 9a. BLOCK_COOLDOWN — game-level post-exit cooldown (v4.0 Patch 4)
        if gts.last_exit_time > 0:
            game_cd = now - gts.last_exit_time
            if game_cd < POST_EXIT_COOLDOWN_S:
                self._blocks["BLOCK_COOLDOWN"] += 1
                log.info("BLOCK_COOLDOWN | %s | game cooldown %.0fs < %.0fs",
                         gs_str, game_cd, POST_EXIT_COOLDOWN_S)
                return

        # 9b. BLOCK_COOLDOWN — per-direction cooldown after close (existing)
        dir_cd = gts.direction_cooldown_remaining(signal.direction, now)
        if dir_cd > 0:
            self._blocks["BLOCK_COOLDOWN"] += 1
            log.info("BLOCK_COOLDOWN | %s | %s cooldown %.0fs remaining",
                     gs_str, signal.direction, dir_cd)
            return

        # 10. Per-game stop
        if gts.status == GameStatus.STOPPED:
            return

        # 11. BLOCK_GAME_TRADE_LIMIT — max entries per game (v4.0 Patch 3)
        if gts.entry_count >= MAX_TRADES_PER_GAME:
            self._blocks["BLOCK_GAME_TRADE_LIMIT"] += 1
            log.info("BLOCK_GAME_TRADE_LIMIT | %s | entries=%d >= %d",
                     gs_str, gts.entry_count, MAX_TRADES_PER_GAME)
            return

        # 12. BLOCK_EDGE_CONFIRM — edge must persist N ticks (v4.0 Patch 2)
        confirm_key = (signal.game_id, signal.direction)
        # Decay counter if last signal for this key was >15s ago (prevents stale counters)
        last_ts = self._edge_confirm_last_ts.get(confirm_key, 0)
        if last_ts > 0 and (now - last_ts) > 15.0:
            self._edge_confirm[confirm_key] = 0
        self._edge_confirm[confirm_key] = self._edge_confirm.get(confirm_key, 0) + 1
        self._edge_confirm_last_ts[confirm_key] = now
        # Reset opposite direction counter for this game
        opp_key = (signal.game_id, "BUY" if signal.direction == "SELL" else "SELL")
        self._edge_confirm[opp_key] = 0

        if self._edge_confirm[confirm_key] < EDGE_CONFIRM_TICKS:
            self._blocks["BLOCK_EDGE_CONFIRM"] += 1
            log.info("BLOCK_EDGE_CONFIRM | %s | %s confirm=%d/%d",
                     gs_str, signal.direction,
                     self._edge_confirm[confirm_key], EDGE_CONFIRM_TICKS)
            return

        # ── Determine execution price ─────────────────────────────
        if signal.direction == "BUY":
            entry_price = book.best_ask if book.best_ask > 0 else signal.market_prob
        else:
            entry_price = book.best_bid if book.best_bid > 0 else signal.market_prob

        # 13. BLOCK_PRICE on ACTUAL execution price (v3.8: direction-specific)
        if not (band_lo <= entry_price <= band_hi):
            self._blocks["BLOCK_PRICE"] += 1
            log.info("BLOCK_PRICE | %s | %s entry=%.3f (actual) outside [%.2f,%.2f]",
                     gs_str, signal.direction, entry_price, band_lo, band_hi)
            return

        # 14. BLOCK_ENTRY_SPREAD — tight spread check at execution (v4.0 Patch 7)
        if book.spread > ENTRY_MAX_SPREAD:
            self._blocks["BLOCK_ENTRY_SPREAD"] += 1
            log.info("BLOCK_ENTRY_SPREAD | %s | spread=%.3f > %.3f",
                     gs_str, book.spread, ENTRY_MAX_SPREAD)
            return

        # 15. BLOCK_ENTRY_FRESHNESS — book must be recent (v4.0 Patch 7)
        entry_book_age = now - book.timestamp if book.timestamp > 0 else 9999
        if entry_book_age > ENTRY_MAX_BOOK_AGE_S:
            self._blocks["BLOCK_ENTRY_FRESHNESS"] += 1
            log.info("BLOCK_ENTRY_FRESHNESS | %s | book_age=%.1fs > %.1fs",
                     gs_str, entry_book_age, ENTRY_MAX_BOOK_AGE_S)
            return

        # ── Log daily summary periodically ────────────────────────
        self.log_daily_summary()

        # ── Size Calculation ──────────────────────────────────────
        # size = abs(edge) * 1000, clamped [50, 300]
        size = abs(signal.edge) * 1000
        size = max(50.0, min(300.0, size))

        self._position_counter += 1
        pos = PaperPosition(
            position_id=f"P{self._position_counter:04d}",
            game_id=signal.game_id,
            token_id=signal.token_id,
            outcome=signal.outcome,
            direction=signal.direction,
            entry_price=entry_price,
            entry_edge=signal.edge,
            entry_time=signal.timestamp,
            entry_game_state=signal.game_state,
            market_title=link.polymarket_title,
            size_usd=size,
            sport=link.sport,
        )

        self.positions[pos.position_id] = pos
        # v4.0: track entry count (for per-game trade limit)
        gts.entry_count += 1
        edge_cc = self._edge_confirm.get(confirm_key, 0)
        self.csv.log_trade(pos, "ENTRY",
                           edge_confirm_count=edge_cc,
                           game_trade_count=gts.entry_count)
        self._trades_taken += 1
        # v4.0: reset edge confirmation counter after entry
        self._edge_confirm[confirm_key] = 0

        log.info("PAPER ENTRY: %s %s %s @ %.3f ($%.0f) edge=%.3f confirm=%d | %s",
                 pos.position_id, pos.direction, pos.outcome,
                 entry_price, size, signal.edge, edge_cc, signal.game_state)

        await self.tg.notify_paper_entry(
            pos.position_id, pos.direction, pos.outcome,
            entry_price, size, signal.edge, signal.game_state,
        )

    async def _check_exits(self, model: ModelOutput, books: dict[str, BookState],
                           link: GameMarketLink, game_state: GameState):
        for pos in list(self.positions.values()):
            if not pos.is_open or pos.game_id != game_state.game_id:
                continue

            book = books.get(pos.token_id)
            if not book:
                continue

            current_mid = book.mid if book.mid > 0 else book.last_trade_price
            if pos.outcome == "home":
                model_p = model.p_home
            elif pos.outcome == "away":
                model_p = model.p_away
            else:
                model_p = model.p_draw

            current_edge = model_p - current_mid
            if pos.direction == "SELL":
                current_edge = -current_edge

            now = time.time()
            time_since_entry = now - pos.entry_time

            # ── v4.3: Compute adverse move + live MAE/MFE tracking ──
            if pos.direction == "BUY":
                adverse = pos.entry_price - current_mid
                favorable = current_mid - pos.entry_price
            else:
                adverse = current_mid - pos.entry_price
                favorable = pos.entry_price - current_mid

            adverse_ticks = max(0.0, adverse * 100)
            favorable_ticks = max(0.0, favorable * 100)

            # Update running MAE/MFE on the position
            if adverse_ticks > pos.mae_ticks:
                pos.mae_ticks = adverse_ticks
                pos.time_to_mae_s = time_since_entry
            if favorable_ticks > pos.mfe_ticks:
                pos.mfe_ticks = favorable_ticks
                pos.time_to_mfe_s = time_since_entry

            # ── Determine sport-specific parameters ────────────────
            is_football = link.sport.lower() == "football"
            sl_ticks = FOOTBALL_STOP_LOSS_TICKS if is_football else STOP_LOSS_TICKS
            timeout_s = FOOTBALL_TIMEOUT_S if is_football else DEFAULT_TIMEOUT_S

            exit_reason = ""

            # ── v4.3 exit priority ─────────────────────────────────
            # Priority: stop_loss > momentum_exit > convergence > edge_flip > game_end > timeout

            # HARD STOP-LOSS — sport-specific threshold
            stop_price = sl_ticks * 0.01
            if adverse >= stop_price:
                exit_reason = "stop_loss"

            # v4.3: MOMENTUM EXIT — football only, early trend detection
            elif is_football and time_since_entry <= FOOTBALL_FAST_MOVE_S \
                    and adverse_ticks >= FOOTBALL_FAST_MOVE_TICKS:
                exit_reason = "momentum_exit"
                pos.momentum_exit_triggered = True

            # Convergence — allowed during hold window (edge closed = good exit)
            elif abs(current_edge) < EXIT_CONVERGENCE:
                exit_reason = "convergence"

            # edge_flip — only after hold window, require meaningful reversal
            elif current_edge < -EDGE_FLIP_THRESHOLD and time_since_entry >= MIN_HOLD_S:
                exit_reason = "edge_flip"

            # game_end and timeout override everything (absolute exits)
            if not game_state.is_live:
                exit_reason = "game_end"
            if time_since_entry > timeout_s:
                exit_reason = "timeout"

            if exit_reason:
                if pos.direction == "BUY":
                    exit_price = book.best_bid if book.best_bid > 0 else current_mid
                else:
                    exit_price = book.best_ask if book.best_ask > 0 else current_mid

                if game_state.status == "finished":
                    if pos.outcome == "home":
                        exit_price = 1.0 if game_state.home_score > game_state.away_score else 0.0
                    elif pos.outcome == "away":
                        exit_price = 1.0 if game_state.away_score > game_state.home_score else 0.0
                    elif pos.outcome == "draw":
                        exit_price = 1.0 if game_state.home_score == game_state.away_score else 0.0

                if pos.direction == "BUY":
                    pnl = (exit_price - pos.entry_price) * pos.size_usd
                else:
                    pnl = (pos.entry_price - exit_price) * pos.size_usd

                pos.exit_price = exit_price
                pos.exit_time = now
                pos.exit_reason = exit_reason
                pos.pnl = pnl
                pos.is_open = False
                # v4.0 stability fields
                pos.edge_at_exit = current_edge
                pos.stop_loss_triggered = (exit_reason == "stop_loss")
                self.daily_pnl += pnl

                # Per-game PnL + cooldown tracking
                gts = self._get_game_state(game_state.game_id)
                stop_transition = gts.record_exit(pnl, now)
                gts.record_direction_exit(pos.direction, now)  # v3.4 per-direction cooldown
                if stop_transition:
                    log.info("%s: %s (pnl=$%.2f, game_id=%s)",
                             stop_transition, link.polymarket_title,
                             gts.pnl, game_state.game_id)
                self.csv.log_trade(pos, "EXIT",
                                   game_trade_count=gts.entry_count)

                log.info("PAPER EXIT: %s %s @ %.3f → %.3f PnL=$%.2f (%s) "
                         "hold=%.1fs edge_now=%.3f | daily=$%.2f",
                         pos.position_id, pos.outcome,
                         pos.entry_price, exit_price, pnl,
                         exit_reason, time_since_entry, current_edge,
                         self.daily_pnl)

                await self.tg.notify_paper_exit(
                    pos.position_id, pos.outcome,
                    pos.entry_price, exit_price, pnl,
                    exit_reason, self.daily_pnl,
                )

    async def _handle_game_end(self, game_state: GameState, link: GameMarketLink):
        """Handle game completion — close positions, send summary."""
        # Count trades and PnL for this game
        game_trades = [p for p in self.positions.values()
                       if p.game_id == game_state.game_id and not p.is_open]
        game_pnl = sum(p.pnl for p in game_trades)
        game_signals = self._signal_count  # approximate

        log.info("GAME OVER: %s %d-%d %s | trades=%d pnl=$%.2f",
                 game_state.home_team, game_state.home_score,
                 game_state.away_score, game_state.away_team,
                 len(game_trades), game_pnl)

        await self.tg.notify_game_summary(
            game_state.home_team, game_state.away_team,
            game_state.home_score, game_state.away_score,
            game_state.league, len(game_trades),
            game_pnl, game_signals,
        )

    def get_summary(self) -> dict:
        closed = [p for p in self.positions.values() if not p.is_open]
        open_pos = [p for p in self.positions.values() if p.is_open]

        if not closed:
            return {
                "total_trades": 0,
                "open_positions": len(open_pos),
                "daily_pnl": 0.0,
            }

        wins = [p for p in closed if p.pnl > 0]
        losses = [p for p in closed if p.pnl <= 0]

        return {
            "total_trades": len(closed),
            "open_positions": len(open_pos),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl": sum(p.pnl for p in closed),
            "avg_pnl": sum(p.pnl for p in closed) / len(closed),
            "avg_edge_at_entry": sum(abs(p.entry_edge) for p in closed) / len(closed),
            "best_trade": max(p.pnl for p in closed),
            "worst_trade": min(p.pnl for p in closed),
            "daily_pnl": self.daily_pnl,
        }

    async def close(self):
        # Send session summary via Telegram
        summary = self.get_summary()
        await self.tg.notify_session_summary(summary)
        await self.tg.close()
        self.csv.close()
