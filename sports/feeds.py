"""
Live data feeds — ESPN Football, NBA scores, and Polymarket WS.
All feeds are async and produce structured game state / market state.
"""
import asyncio
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import websockets

from sports.config import (
    ESPN_FOOTBALL_BASE, ESPN_LEAGUES,
    NBA_SCOREBOARD_URL,
    POLYMARKET_WS_URL,
    SCORE_POLL_INTERVAL_S,
)

log = logging.getLogger("sports.feeds")


# ═══════════════════════════════════════════════════════════════════════
#  Data structures for game state
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GameState:
    """Universal game state — works for both football and NBA."""
    game_id: str
    sport: str                  # "nba" or "football"
    league: str
    status: str                 # "scheduled", "live", "finished", "halftime"
    home_team: str
    away_team: str
    home_score: int = 0
    away_score: int = 0
    elapsed_minutes: float = 0.0
    total_minutes: float = 90.0  # 90 for football, 48 for NBA
    period: str = ""            # "1H", "2H", "Q1"-"Q4", "OT"
    # Football-specific
    home_shots: int = 0
    away_shots: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0
    home_corners: int = 0
    away_corners: int = 0
    home_red_cards: int = 0
    away_red_cards: int = 0
    # NBA-specific
    home_q_scores: list[int] = field(default_factory=list)
    away_q_scores: list[int] = field(default_factory=list)
    game_clock: str = ""        # "5:32" remaining in period
    possession: str = ""        # "home" or "away"
    # Timing
    timestamp: float = 0.0
    last_event: str = ""        # "goal", "basket", "foul", etc.

    @property
    def minutes_remaining(self) -> float:
        return max(0.0, self.total_minutes - self.elapsed_minutes)

    @property
    def score_diff(self) -> int:
        """Positive = home leads."""
        return self.home_score - self.away_score

    @property
    def is_live(self) -> bool:
        return self.status in ("live", "1H", "2H", "HT", "ET",
                                "Q1", "Q2", "Q3", "Q4", "OT")


@dataclass
class BookState:
    """Best bid/ask state for one Polymarket token."""
    token_id: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    mid: float = 0.0
    spread: float = 1.0
    last_trade_price: float = 0.0
    last_trade_size: float = 0.0
    timestamp: float = 0.0
    volume_session: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
#  ESPN Football Feed
# ═══════════════════════════════════════════════════════════════════════

# Regex to parse ESPN displayClock like "45'", "90'+5'", "120'+1'"
_CLOCK_RE = re.compile(r"(\d+)'(?:\+(\d+)')?")


def _parse_espn_clock(display_clock: str) -> float:
    """Convert ESPN displayClock string to elapsed minutes."""
    m = _CLOCK_RE.search(display_clock)
    if not m:
        return 0.0
    base = int(m.group(1))
    added = int(m.group(2)) if m.group(2) else 0
    return float(base + added)


def _espn_status_to_internal(status_type: dict, period: int) -> tuple[str, str]:
    """Map ESPN status to internal (status, period_str).

    Returns (status, period_label) where status is one of:
    "NS", "1H", "2H", "HT", "ET", "FT", "AET", "PEN"
    """
    state = status_type.get("state", "")       # "pre", "in", "post"
    short = status_type.get("shortDetail", "")  # "FT", "HT", "AET", etc.
    name = status_type.get("name", "")

    if state == "pre":
        return "NS", ""

    if state == "post":
        if "AET" in short:
            return "FT", "AET"
        return "FT", "FT"

    # state == "in"
    if "HT" in short or "Halftime" in short:
        return "HT", "HT"

    if period <= 1:
        return "1H", "1H"
    elif period == 2:
        return "2H", "2H"
    else:
        return "ET", "ET"


class FootballFeed:
    """Polls ESPN scoreboard for live football scores across configured leagues."""

    def __init__(self):
        self.games: dict[str, GameState] = {}  # event_id → GameState
        self._last_heartbeat: float = 0.0
        # Print startup banner
        log.info("FOOTBALL_SOURCE = ESPN")
        log.info("LEAGUES_LOADED = %d", len(ESPN_LEAGUES))
        log.info("FOOTBALL_FEED_READY")

    async def fetch_todays_fixtures(
        self, session: aiohttp.ClientSession, date_str: str,
    ) -> list[dict]:
        """Fetch all fixtures for a given date across ESPN leagues.

        Args:
            session: aiohttp session
            date_str: "YYYY-MM-DD"

        Returns:
            List of raw ESPN event dicts (for logging/debug).
        """
        # ESPN wants "YYYYMMDD" as dates param
        date_compact = date_str.replace("-", "")
        all_events: list[dict] = []
        total_fixtures = 0

        for league_code in ESPN_LEAGUES:
            url = f"{ESPN_FOOTBALL_BASE}/{league_code}/scoreboard"
            params = {"dates": date_compact}

            try:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        log.warning("FOOTBALL_FEED_ERR | %s | HTTP %d", league_code, resp.status)
                        continue
                    data = await resp.json()

                events = data.get("events", [])
                league_name = league_code

                for evt in events:
                    comps = evt.get("competitions", [])
                    if not comps:
                        continue
                    comp = comps[0]
                    competitors = comp.get("competitors", [])
                    if len(competitors) < 2:
                        continue

                    # Identify home/away
                    home = away = None
                    for c in competitors:
                        if c.get("homeAway") == "home":
                            home = c
                        else:
                            away = c
                    if not home or not away:
                        continue

                    event_id = str(evt.get("id", ""))
                    home_name = home.get("team", {}).get("displayName", "")
                    away_name = away.get("team", {}).get("displayName", "")

                    gs = GameState(
                        game_id=event_id,
                        sport="football",
                        league=league_name,
                        status="NS",
                        home_team=home_name,
                        away_team=away_name,
                        home_score=0,
                        away_score=0,
                        elapsed_minutes=0.0,
                        total_minutes=90.0,
                        period="",
                        timestamp=time.time(),
                    )
                    self.games[event_id] = gs
                    total_fixtures += 1

                all_events.extend(events)

                if events:
                    log.info("FOOTBALL_FEED_OK | %s | fixtures=%d", league_code, len(events))

            except Exception as e:
                log.error("FOOTBALL_FEED_ERR | %s | %s", league_code, e)

            # Brief pause between leagues to be polite
            await asyncio.sleep(0.3)

        log.info("found %d football fixtures across all leagues", total_fixtures)
        return all_events

    async def fetch_live_scores(
        self, session: aiohttp.ClientSession,
    ) -> dict[str, GameState]:
        """Fetch live scores from ESPN for all configured leagues."""
        now = time.time()
        total_live = 0

        for league_code in ESPN_LEAGUES:
            url = f"{ESPN_FOOTBALL_BASE}/{league_code}/scoreboard"

            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        log.warning("FOOTBALL_FEED_ERR | %s | HTTP %d", league_code, resp.status)
                        continue
                    data = await resp.json()
            except Exception as e:
                log.error("FOOTBALL_FEED_ERR | %s | %s", league_code, e)
                continue

            events = data.get("events", [])

            for evt in events:
                evt_status = evt.get("status", {})
                status_type = evt_status.get("type", {})
                state = status_type.get("state", "")

                # We care about live ("in") and just-finished ("post") games
                if state not in ("in", "post"):
                    continue

                comps = evt.get("competitions", [])
                if not comps:
                    continue
                comp = comps[0]
                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue

                home = away = None
                home_id = away_id = ""
                for c in competitors:
                    if c.get("homeAway") == "home":
                        home = c
                        home_id = c.get("id", "")
                    else:
                        away = c
                        away_id = c.get("id", "")
                if not home or not away:
                    continue

                event_id = str(evt.get("id", ""))
                display_clock = evt_status.get("displayClock", "0'")
                period_num = evt_status.get("period", 1)

                internal_status, period_label = _espn_status_to_internal(
                    status_type, period_num,
                )

                # Parse elapsed minutes
                elapsed = _parse_espn_clock(display_clock)

                # Clamp based on status
                if internal_status == "HT":
                    elapsed = 45.0
                elif internal_status == "FT":
                    elapsed = 90.0  # or 120 for AET
                    if period_label == "AET":
                        elapsed = 120.0

                # Determine total_minutes for model
                total_mins = 90.0
                if period_num > 2 or period_label in ("ET", "AET"):
                    total_mins = 120.0

                # Scores
                home_score = int(home.get("score", "0") or "0")
                away_score = int(away.get("score", "0") or "0")

                # Red cards from details
                home_reds = 0
                away_reds = 0
                details = comp.get("details", [])
                for detail in details:
                    if detail.get("redCard"):
                        detail_team_id = detail.get("team", {}).get("id", "")
                        if detail_team_id == home_id:
                            home_reds += 1
                        elif detail_team_id == away_id:
                            away_reds += 1

                # Last event
                last_event = ""
                if details:
                    last_detail = details[-1]
                    last_event = last_detail.get("type", {}).get("text", "")

                home_name = home.get("team", {}).get("displayName", "")
                away_name = away.get("team", {}).get("displayName", "")

                gs = GameState(
                    game_id=event_id,
                    sport="football",
                    league=league_code,
                    status=internal_status,
                    home_team=home_name,
                    away_team=away_name,
                    home_score=home_score,
                    away_score=away_score,
                    elapsed_minutes=elapsed,
                    total_minutes=total_mins,
                    period=period_label,
                    home_red_cards=home_reds,
                    away_red_cards=away_reds,
                    timestamp=now,
                    last_event=last_event,
                )

                self.games[event_id] = gs

                if state == "in":
                    total_live += 1
                    log.info(
                        "FOOTBALL_LIVE | %s vs %s | min=%d | score=%d-%d | reds=%d-%d",
                        home_name, away_name, int(elapsed),
                        home_score, away_score,
                        home_reds, away_reds,
                    )

            await asyncio.sleep(0.2)

        # Heartbeat every 5 minutes
        if now - self._last_heartbeat >= 300:
            total_tracked = sum(1 for g in self.games.values() if g.status != "FT")
            log.info(
                "FOOTBALL_HEARTBEAT | total_live=%d | total_tracked=%d",
                total_live, total_tracked,
            )
            self._last_heartbeat = now

        return self.games


# ═══════════════════════════════════════════════════════════════════════
#  NBA Feed
# ═══════════════════════════════════════════════════════════════════════

class NBAFeed:
    """Polls NBA.com CDN for live scores."""

    def __init__(self):
        self.games: dict[str, GameState] = {}  # game_id → GameState

    async def fetch_live_scores(self, session: aiohttp.ClientSession) -> dict[str, GameState]:
        """Fetch today's NBA scoreboard."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.nba.com/",
            }
            async with session.get(
                NBA_SCOREBOARD_URL,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    log.warning("NBA scoreboard returned %d", resp.status)
                    return self.games
                data = await resp.json(content_type=None)
        except Exception as e:
            log.error("NBA scoreboard error: %s", e)
            return self.games

        now = time.time()
        scoreboard = data.get("scoreboard", {})
        games = scoreboard.get("games", [])

        for game in games:
            game_id = game.get("gameId", "")
            status = game.get("gameStatus", 1)  # 1=scheduled, 2=live, 3=finished
            period = game.get("period", 0)
            game_clock = game.get("gameClock", "")

            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})

            # Parse game clock: "PT05M32.00S" → minutes elapsed in period
            clock_minutes = 12.0  # default full period remaining
            if game_clock and game_clock.startswith("PT"):
                try:
                    parts = game_clock[2:].replace("S", "").split("M")
                    mins = float(parts[0]) if parts[0] else 0
                    secs = float(parts[1]) if len(parts) > 1 and parts[1] else 0
                    clock_minutes = mins + secs / 60.0
                except (ValueError, IndexError):
                    pass

            # Elapsed = completed periods * 12 + (12 - remaining in current period)
            if status == 2:  # live
                elapsed = (period - 1) * 12.0 + (12.0 - clock_minutes)
                status_str = f"Q{period}"
            elif status == 3:
                elapsed = 48.0
                status_str = "finished"
            else:
                elapsed = 0.0
                status_str = "scheduled"

            # Quarter scores
            home_q = []
            away_q = []
            for p in home.get("periods", []):
                home_q.append(p.get("score", 0))
            for p in away.get("periods", []):
                away_q.append(p.get("score", 0))

            gs = GameState(
                game_id=game_id,
                sport="nba",
                league="NBA",
                status=status_str,
                home_team=home.get("teamName", ""),
                away_team=away.get("teamName", ""),
                home_score=home.get("score", 0),
                away_score=away.get("score", 0),
                elapsed_minutes=elapsed,
                total_minutes=48.0,
                period=f"Q{period}" if period > 0 else "",
                home_q_scores=home_q,
                away_q_scores=away_q,
                game_clock=game_clock,
                timestamp=now,
            )

            self.games[game_id] = gs

        return self.games


# ═══════════════════════════════════════════════════════════════════════
#  Polymarket WebSocket Feed (lightweight, sports-focused)
# ═══════════════════════════════════════════════════════════════════════

class PolymarketFeed:
    """
    Connects to Polymarket CLOB WebSocket, subscribes to sports
    market token IDs, and maintains best bid/ask state.
    """

    def __init__(self):
        self.books: dict[str, BookState] = {}  # token_id → BookState
        self.trades: list[dict] = []           # recent trades buffer
        self._token_ids: list[str] = []
        self._subscribed_set: set[str] = set() # fast membership check
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._shutdown = False
        self._message_count = 0
        self._trade_callbacks: list = []
        # ── Diagnostic counters ──────────────────────────────────
        self._raw_log_budget = 50           # messages to log raw after connect
        self._event_type_counts: Counter = Counter()
        self._unique_asset_ids: set[str] = set()
        self._mismatch_count = 0
        self._mismatch_samples: list[str] = []
        self._last_book_update_ts: float = 0.0
        self._last_stats_ts: float = 0.0
        self._connect_count = 0

    def set_tokens(self, token_ids: list[str]):
        """Update the set of token IDs to subscribe to.
        If WS is already connected, immediately re-subscribe."""
        old_set = self._subscribed_set
        self._token_ids = token_ids
        self._subscribed_set = set(token_ids)
        for tid in token_ids:
            if tid not in self.books:
                self.books[tid] = BookState(token_id=tid)
        log.info("polymarket feed configured for %d tokens", len(token_ids))

        # Re-subscribe immediately if WS is already live
        if self._connected and self._ws and old_set != self._subscribed_set:
            diff_new = self._subscribed_set - old_set
            diff_removed = old_set - self._subscribed_set
            log.info("token set changed while WS live: +%d new, -%d removed → re-subscribing",
                     len(diff_new), len(diff_removed))
            asyncio.ensure_future(self._subscribe())

    def on_trade(self, callback):
        """Register callback for trade events."""
        self._trade_callbacks.append(callback)

    async def _subscribe(self):
        """Subscribe to all configured token IDs."""
        if not self._ws or not self._token_ids:
            log.warning("_subscribe called but ws=%s, tokens=%d",
                        bool(self._ws), len(self._token_ids))
            return

        # Subscribe in batches to avoid message size limits
        batch_size = 50
        for i in range(0, len(self._token_ids), batch_size):
            batch = self._token_ids[i:i + batch_size]
            msg = {
                "type": "market",
                "assets_ids": batch,
            }
            payload = json.dumps(msg)
            await self._ws.send(payload)

            # Log first and last batch payloads for verification
            if i == 0:
                sample = batch[:3]
                log.info("SUB payload sample (batch 1): type=market, first_3_tokens=%s", sample)
            log.info("subscribed to tokens %d-%d of %d",
                     i + 1, min(i + batch_size, len(self._token_ids)),
                     len(self._token_ids))
            await asyncio.sleep(0.1)

        log.info("subscription complete — waiting for server ACK/first book message")

    async def _handle_message(self, raw: str):
        """Process incoming WS message."""
        self._message_count += 1
        now = time.time()

        # ── Controlled raw logging ───────────────────────────────
        if self._raw_log_budget > 0:
            self._raw_log_budget -= 1
            # Truncate to 500 chars to avoid log flooding
            snippet = raw[:500]
            log.info("RAW_WS [%d remaining]: %s", self._raw_log_budget, snippet)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("WS JSON decode failed: %s", raw[:200])
            return

        messages = data if isinstance(data, list) else [data]

        for msg in messages:
            event_type = msg.get("event_type", "")
            asset_id = msg.get("asset_id", "")

            # ── Handle nested price_changes wrapper ──────────────
            # Some messages arrive as {"market":"...", "price_changes":[...]}
            # without a top-level event_type. Unpack and process each sub-item.
            if not event_type and "price_changes" in msg:
                for pc in msg.get("price_changes", []):
                    pc_asset = pc.get("asset_id", "")
                    self._event_type_counts["price_change"] += 1
                    if pc_asset:
                        self._unique_asset_ids.add(pc_asset)
                        if self._subscribed_set and pc_asset not in self._subscribed_set:
                            self._mismatch_count += 1
                            if len(self._mismatch_samples) < 5:
                                self._mismatch_samples.append(pc_asset)
                    self._handle_price_change(pc, pc_asset, now)
                continue

            # ── Event type counter ───────────────────────────────
            self._event_type_counts[event_type or "_no_event_type"] += 1

            # ── Token mismatch detector ──────────────────────────
            if asset_id:
                self._unique_asset_ids.add(asset_id)
                if self._subscribed_set and asset_id not in self._subscribed_set:
                    self._mismatch_count += 1
                    if len(self._mismatch_samples) < 5:
                        self._mismatch_samples.append(asset_id)

            if event_type == "book":
                self._handle_book(msg, asset_id, now)
            elif event_type == "tick":
                await self._handle_tick(msg, asset_id, now)
            elif event_type == "last_trade_price":
                self._handle_last_trade(msg, asset_id, now)
            elif event_type == "price_change":
                self._handle_price_change(msg, asset_id, now)

        # ── 60-second diagnostic stats ───────────────────────────
        if now - self._last_stats_ts >= 60.0:
            self._last_stats_ts = now
            book_age = now - self._last_book_update_ts if self._last_book_update_ts > 0 else -1
            log.info(
                "WS_DIAG | msgs_total=%d | event_types=%s | "
                "unique_assets=%d | subscribed=%d | "
                "mismatch=%d | book_age=%.1fs",
                self._message_count,
                dict(self._event_type_counts),
                len(self._unique_asset_ids),
                len(self._subscribed_set),
                self._mismatch_count,
                book_age,
            )
            if self._mismatch_samples:
                log.info("WS_DIAG mismatch samples: %s", self._mismatch_samples[:5])
            # Check for book staleness
            if book_age > 120 and self._message_count > 100:
                log.warning(
                    "BOOK STALE: no book update in %.0fs despite %d messages — "
                    "possible schema mismatch or subscription failure",
                    book_age, self._message_count
                )

    def _handle_book(self, msg: dict, asset_id: str, ts: float):
        """Process book snapshot — update BBO."""
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])

        if asset_id not in self.books:
            self.books[asset_id] = BookState(token_id=asset_id)

        book = self.books[asset_id]
        book.timestamp = ts
        self._last_book_update_ts = ts  # staleness tracker

        # Reset BOTH sides from the snapshot — if a side is empty,
        # zero it out so we don't carry stale data from a previous snapshot.
        if bids:
            best = max(bids, key=lambda x: float(x.get("price", 0)))
            book.best_bid = float(best.get("price", 0))
            book.bid_size = float(best.get("size", 0))
        else:
            book.best_bid = 0.0
            book.bid_size = 0.0

        if asks:
            best = min(asks, key=lambda x: float(x.get("price", 1)))
            book.best_ask = float(best.get("price", 1))
            book.ask_size = float(best.get("size", 0))
        else:
            book.best_ask = 0.0
            book.ask_size = 0.0

        self._update_mid(book)

    async def _handle_tick(self, msg: dict, asset_id: str, ts: float):
        """Process trade tick."""
        price = msg.get("price")
        size = msg.get("size")

        if price is not None and asset_id in self.books:
            book = self.books[asset_id]
            book.last_trade_price = float(price)
            book.last_trade_size = float(size) if size else 0.0
            book.timestamp = ts

            trade = {
                "timestamp": ts,
                "token_id": asset_id,
                "price": float(price),
                "size": float(size) if size else 0.0,
                "side": msg.get("side", ""),
            }
            self.trades.append(trade)

            # Keep trade buffer manageable
            if len(self.trades) > 10000:
                self.trades = self.trades[-5000:]

            for cb in self._trade_callbacks:
                try:
                    await cb(trade)
                except Exception as e:
                    log.warning("trade callback error: %s", e)

    def _handle_last_trade(self, msg: dict, asset_id: str, ts: float):
        """Process last_trade_price update."""
        price = msg.get("price")
        if price is not None and asset_id in self.books:
            self.books[asset_id].last_trade_price = float(price)
            self.books[asset_id].timestamp = ts

    def _handle_price_change(self, msg: dict, asset_id: str, ts: float):
        """Process price change — update BBO from best_bid/best_ask fields.

        price_change events are the PRIMARY live data source (~200x more
        frequent than book snapshots). Each message contains best_bid/best_ask
        which must be used to keep BookState current.
        """
        if not asset_id:
            return

        if asset_id not in self.books:
            self.books[asset_id] = BookState(token_id=asset_id)

        book = self.books[asset_id]
        book.timestamp = ts

        price = msg.get("price")
        if price is not None:
            book.last_trade_price = float(price)

        # Update BBO from price_change fields
        best_bid = msg.get("best_bid")
        best_ask = msg.get("best_ask")
        if best_bid is not None:
            book.best_bid = float(best_bid)
        if best_ask is not None:
            book.best_ask = float(best_ask)

        size = msg.get("size")
        side = msg.get("side", "")
        if size is not None:
            if side == "BUY":
                book.bid_size = float(size)
            elif side == "SELL":
                book.ask_size = float(size)

        self._update_mid(book)

    @staticmethod
    def _update_mid(book: BookState):
        """Recalculate mid/spread from current BBO."""
        if book.best_bid > 0 and book.best_ask > 0:
            book.mid = (book.best_bid + book.best_ask) / 2
            book.spread = book.best_ask - book.best_bid
        elif book.best_ask > 0:
            # One-sided: only asks (team heavily favored). Use ask as ceiling.
            book.mid = book.best_ask
            book.spread = 1.0
        elif book.best_bid > 0:
            # One-sided: only bids. Use bid as floor.
            book.mid = book.best_bid
            book.spread = 1.0
        # else: both zero, don't update mid

    async def run(self):
        """Main WS loop with reconnection."""
        log.info("polymarket WS feed starting")
        backoff = 1.0

        while not self._shutdown:
            try:
                async with websockets.connect(
                    POLYMARKET_WS_URL,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=5,
                    max_size=2**22,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._connect_count += 1
                    backoff = 1.0

                    # Set raw log budget: 50 on first connect, 10 on reconnect
                    if self._connect_count == 1:
                        self._raw_log_budget = 50
                        log.info("connected to polymarket WS (first connect — logging 50 raw messages)")
                    else:
                        self._raw_log_budget = 10
                        log.info("reconnected to polymarket WS (connect #%d — logging 10 raw messages)",
                                 self._connect_count)

                    if self._token_ids:
                        await self._subscribe()
                    else:
                        log.warning("WS connected but NO token IDs to subscribe")

                    while not self._shutdown:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                            await self._handle_message(raw)
                        except asyncio.TimeoutError:
                            try:
                                pong = await ws.ping()
                                await asyncio.wait_for(pong, timeout=10)
                            except Exception:
                                log.warning("ping failed, reconnecting")
                                break

            except websockets.exceptions.ConnectionClosed as e:
                log.warning("WS closed: %s", e)
            except Exception as e:
                log.error("WS error: %s", e)
            finally:
                self._connected = False
                self._ws = None

            if not self._shutdown:
                log.info("reconnecting in %.1fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def shutdown(self):
        self._shutdown = True
        if self._ws:
            await self._ws.close()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def message_count(self) -> int:
        return self._message_count
