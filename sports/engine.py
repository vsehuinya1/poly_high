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
    # Execution hygiene (v1.4)
    PRICE_BAND_LO, PRICE_BAND_HI, MAX_SPREAD, MAX_BOOK_AGE_S, MAX_SCORE_DIFF,
    GATE_FRESH_THRESHOLD, GATE_STREAK_S, GATE_ROLLING_WINDOW_S, GATE_ROLLING_FRESH_PCT,
    FREEZE_STALE_THRESHOLD, FREEZE_STALE_DURATION_S, UNFREEZE_STREAK_S,
    COOLDOWN_S, PER_GAME_STOP,
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

    def log_trade(self, pos: PaperPosition, event: str):
        headers = [
            "timestamp", "event", "position_id", "game_id",
            "token_id", "outcome", "direction",
            "entry_price", "exit_price", "entry_edge",
            "size_usd", "pnl", "exit_reason", "game_state",
        ]
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
        self.last_exit_time: float = 0.0
        self.band_rejects: int = 0

        # Gate tracking
        self._fresh_streak_start: float = 0.0  # when current fresh streak began
        self._rolling_ticks: deque = deque()    # (timestamp, is_fresh_20s) pairs

        # Freeze tracking
        self._stale_streak_start: float = 0.0   # when current stale streak began
        self._unfreeze_fresh_start: float = 0.0 # when fresh streak began during freeze

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

    def _get_game_state(self, game_id: str) -> GameTradeState:
        if game_id not in self._game_states:
            self._game_states[game_id] = GameTradeState(game_id)
        return self._game_states[game_id]

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

                # Telegram alert for edge signals
                await self.tg.notify_edge_signal(
                    direction, outcome, edge, model_p, market_mid,
                    game_state_str, link.polymarket_title,
                )

        # ── Deduplicate signals (only best edge per game) ────────
        if signals:
            signals.sort(key=lambda x: abs(x.edge), reverse=True)
            signals = [signals[0]]

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

        # ══════════════════════════════════════════════════════════
        # EXECUTION HYGIENE — all must pass
        # ══════════════════════════════════════════════════════════

        elapsed = now - game_state.timestamp
        adj_sec = max(0, (game_state.total_minutes - game_state.elapsed_minutes) * 60.0 - elapsed)
        book_age = now - book.timestamp if book.timestamp > 0 else 9999
        score_diff = abs(game_state.home_score - game_state.away_score)

        # Original time gate
        if adj_sec > 600:
            return

        # Edge threshold
        if abs(signal.edge) < 0.07:
            return

        # 1. Hard entry filters
        if not (PRICE_BAND_LO <= signal.market_prob <= PRICE_BAND_HI):
            gts.band_rejects += 1
            log.info(
                "REJECT PRICE_BAND | %s | price=%.3f | model=%.3f | edge=%.3f",
                signal.game_state, signal.market_prob,
                signal.model_prob, signal.edge,
            )
            return
        if book.spread > MAX_SPREAD:
            return
        if book_age > MAX_BOOK_AGE_S:
            return
        if score_diff > MAX_SCORE_DIFF:
            return

        # 2. Game must be ACTIVE (gate passed)
        if not gts.can_trade:
            return

        # 3. Not frozen
        if gts.status == GameStatus.FROZEN:
            return  # covered by can_trade but explicit for clarity

        # 4. Cooldown
        cd = gts.cooldown_remaining(now)
        if cd > 0:
            return

        # 5. Per-game stop
        if gts.status == GameStatus.STOPPED:
            return

        if signal.direction == "BUY":
            entry_price = book.best_ask if book.best_ask > 0 else signal.market_prob
        else:
            entry_price = book.best_bid if book.best_bid > 0 else signal.market_prob

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
        )

        self.positions[pos.position_id] = pos
        self.csv.log_trade(pos, "ENTRY")

        log.info("PAPER ENTRY: %s %s %s @ %.3f ($%.0f) edge=%.3f | %s",
                 pos.position_id, pos.direction, pos.outcome,
                 entry_price, size, signal.edge, signal.game_state)

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

            exit_reason = ""

            # Check for convergence or edge flip
            if abs(current_edge) < EXIT_CONVERGENCE:
                exit_reason = "convergence"
            elif current_edge < -ENTRY_EDGE_THRESHOLD:
                exit_reason = "edge_flip"
            
            if not game_state.is_live:
                exit_reason = "game_end"
            if time.time() - pos.entry_time > 1800: # 30 min timeout
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
                pos.exit_time = time.time()
                pos.exit_reason = exit_reason
                pos.pnl = pnl
                pos.is_open = False
                self.daily_pnl += pnl

                # Per-game PnL + cooldown tracking
                gts = self._get_game_state(game_state.game_id)
                stop_transition = gts.record_exit(pnl, time.time())
                if stop_transition:
                    log.info("%s: %s (pnl=$%.2f, game_id=%s)",
                             stop_transition, link.polymarket_title,
                             gts.pnl, game_state.game_id)
                self.csv.log_trade(pos, "EXIT")

                log.info("PAPER EXIT: %s %s @ %.3f → %.3f PnL=$%.2f (%s) | daily=$%.2f",
                         pos.position_id, pos.outcome,
                         pos.entry_price, exit_price, pnl,
                         exit_reason, self.daily_pnl)

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
