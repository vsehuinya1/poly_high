"""
Tennis data feed abstraction + Flashscore adapter.

Design:
    TennisDataFeed — abstract base class defining the feed interface.
    FlashscoreFeed — concrete adapter that polls Flashscore mobile API
                     for live point-by-point tennis data.

The adapter is fully isolated from engine/strategy logic. It only
produces TennisPointEvent objects. The engine layer converts those
into TennisState transitions.
"""
from __future__ import annotations

import abc
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from tennis.state import TennisPointEvent

log = logging.getLogger("tennis.feeds")


# ═══════════════════════════════════════════════════════════════════════
#  Abstract Feed Interface
# ═══════════════════════════════════════════════════════════════════════

class TennisDataFeed(abc.ABC):
    """Abstract base class for tennis live data feeds.

    Any concrete adapter must implement these four methods.
    The engine interacts only through this interface.
    """

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        ...

    @abc.abstractmethod
    async def subscribe(self, match_id: str) -> None:
        """Subscribe to point-by-point updates for a specific match."""
        ...

    @abc.abstractmethod
    def parse_point_update(self, raw_event: dict) -> Optional[TennisPointEvent]:
        """Parse a raw event dict into a TennisPointEvent.

        Returns None if the event is not a valid point update
        (e.g., a heartbeat, status change, etc.).
        """
        ...

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return True if the feed is healthy and receiving data."""
        ...


# ═══════════════════════════════════════════════════════════════════════
#  Feed Health Tracker
# ═══════════════════════════════════════════════════════════════════════

class FeedHealthTracker:
    """Tracks feed health metrics: last update time, stall count, etc."""

    def __init__(self, stall_threshold_s: float = 60.0,
                 health_log_interval_s: float = 60.0):
        self.stall_threshold_s = stall_threshold_s
        self.health_log_interval_s = health_log_interval_s
        self.last_update_time: float = 0.0
        self.total_events: int = 0
        self.stall_count: int = 0
        self.restart_count: int = 0
        self._last_health_log: float = 0.0

    def record_event(self) -> None:
        self.last_update_time = time.time()
        self.total_events += 1

    def is_stalled(self) -> bool:
        """True if no update received for longer than stall threshold."""
        if self.last_update_time == 0:
            return False  # never received anything yet
        return (time.time() - self.last_update_time) > self.stall_threshold_s

    def record_stall(self) -> None:
        self.stall_count += 1

    def record_restart(self) -> None:
        self.restart_count += 1

    def log_health(self) -> None:
        """Log health metrics periodically."""
        now = time.time()
        if now - self._last_health_log < self.health_log_interval_s:
            return
        self._last_health_log = now

        age = now - self.last_update_time if self.last_update_time > 0 else -1
        log.info(
            "FEED_HEALTH | events=%d | last_update_age=%.1fs | "
            "stalls=%d | restarts=%d",
            self.total_events, age, self.stall_count, self.restart_count,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Flashscore Mobile API Adapter
# ═══════════════════════════════════════════════════════════════════════

# Point score mapping: Flashscore sends integer codes
_POINT_MAP = {
    "0": "0", "15": "15", "30": "30", "40": "40",
    "A": "AD", "AD": "AD",
    # Flashscore sometimes sends "50" for advantage
    "50": "AD",
}


class FlashscoreFeed(TennisDataFeed):
    """Concrete adapter for Flashscore mobile API.

    Polls the Flashscore mobile JSON endpoints for live point-by-point
    tennis match data. This is a polling-based adapter (not WebSocket)
    since the Flashscore mobile API does not expose a streaming endpoint.

    Connection flow:
        1. connect() → creates aiohttp session.
        2. subscribe(match_id) → registers match for polling.
        3. Poll loop fetches /x/feed/d_su_{match_id} every 2-3s.
        4. parse_point_update() converts Flashscore JSON → TennisPointEvent.
        5. health_check() verifies session + last update age.

    Auto-restart:
        If the feed stalls (no update for 60s during a live match),
        the poll loop reconnects with exponential backoff.

    Note:
        Flashscore endpoints may change without notice.
        This adapter should be treated as fragile and tested
        before every deployment. The abstract TennisDataFeed
        interface makes it trivial to swap in a different source.
    """

    # Mobile API base URL
    BASE_URL = "https://local-global.flashscore.ninja/2/x/feed"

    # Headers to mimic Flashscore mobile app
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 12) FlashScore/3.13",
        "Accept": "application/json",
        "X-Fsign": "SW9D1eZo",      # Flashscore session token (rotates)
    }

    def __init__(self, poll_interval_s: float = 3.0,
                 stall_threshold_s: float = 60.0):
        self.poll_interval_s = poll_interval_s
        self._session = None
        self._subscriptions: set[str] = set()
        self._health = FeedHealthTracker(stall_threshold_s=stall_threshold_s)
        self._running = False
        self._last_scores: dict[str, dict] = {}  # match_id → last parsed score
        self._backoff_s = 1.0
        self._max_backoff_s = 30.0

    async def connect(self) -> None:
        """Create aiohttp session for polling."""
        try:
            import aiohttp
        except ImportError:
            log.error("aiohttp not installed — required for FlashscoreFeed")
            raise

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self._HEADERS,
                timeout=aiohttp.ClientTimeout(total=10),
            )
        self._running = True
        self._backoff_s = 1.0
        log.info("FlashscoreFeed connected")

    async def subscribe(self, match_id: str) -> None:
        """Register a match for polling."""
        self._subscriptions.add(match_id)
        log.info("Subscribed to match: %s (total=%d)",
                 match_id, len(self._subscriptions))

    def parse_point_update(self, raw_event: dict) -> Optional[TennisPointEvent]:
        """Parse Flashscore event JSON into TennisPointEvent.

        Expected raw_event structure (Flashscore mobile format):
        {
            "match_id": "ABC123",
            "home_sets": 1, "away_sets": 0,
            "home_games": 3, "away_games": 2,
            "home_points": "30", "away_points": "15",
            "server": "home",  // or "away"
            "point_winner": "home",
            "is_tiebreak": false,
            "timestamp": 1709500000.0
        }

        Returns None if event is not parseable.
        """
        try:
            match_id = raw_event.get("match_id", "")
            if not match_id:
                return None

            # Map point scores
            hp = _POINT_MAP.get(str(raw_event.get("home_points", "0")), "0")
            ap = _POINT_MAP.get(str(raw_event.get("away_points", "0")), "0")

            # Determine server
            server_raw = raw_event.get("server", "")
            # Server ID will be resolved by the engine layer to actual player IDs
            server_id = f"{match_id}_{server_raw}"

            # Point winner
            winner_raw = raw_event.get("point_winner", "")
            winner_id = f"{match_id}_{winner_raw}" if winner_raw else ""

            return TennisPointEvent(
                match_id=match_id,
                point_winner_id=winner_id,
                new_sets_a=int(raw_event.get("home_sets", 0)),
                new_sets_b=int(raw_event.get("away_sets", 0)),
                new_games_a=int(raw_event.get("home_games", 0)),
                new_games_b=int(raw_event.get("away_games", 0)),
                new_point_a=hp,
                new_point_b=ap,
                new_server_id=server_id,
                is_tiebreak=bool(raw_event.get("is_tiebreak", False)),
                timestamp=float(raw_event.get("timestamp", time.time())),
            )
        except (KeyError, ValueError, TypeError) as e:
            log.warning("Failed to parse point update: %s | raw=%s", e, raw_event)
            return None

    async def health_check(self) -> bool:
        """Return True if feed is healthy."""
        if self._session is None or self._session.closed:
            return False
        if self._health.is_stalled():
            return False
        return True

    # ── Polling Loop ──────────────────────────────────────────────

    async def poll_loop(self) -> None:
        """Main polling loop — runs until shutdown.

        For each subscribed match, fetches the latest score state
        from Flashscore and yields TennisPointEvent objects when
        the score changes.

        Auto-restarts on failure with exponential backoff.
        """
        log.info("FlashscoreFeed poll_loop started | matches=%d",
                 len(self._subscriptions))

        while self._running:
            try:
                for match_id in list(self._subscriptions):
                    event = await self._fetch_match(match_id)
                    if event:
                        self._health.record_event()
                        self._backoff_s = 1.0  # reset backoff on success

                # Log health periodically
                self._health.log_health()

                # Check for stalls
                if self._health.is_stalled() and self._subscriptions:
                    self._health.record_stall()
                    log.warning("FEED_STALL detected — will attempt reconnect")
                    await self._reconnect()

                await asyncio.sleep(self.poll_interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("FlashscoreFeed poll error: %s", e, exc_info=True)
                await asyncio.sleep(self._backoff_s)
                self._backoff_s = min(self._backoff_s * 2, self._max_backoff_s)

    async def _fetch_match(self, match_id: str) -> Optional[TennisPointEvent]:
        """Fetch latest score for a match from Flashscore.

        Returns TennisPointEvent if score changed, else None.
        """
        if not self._session or self._session.closed:
            return None

        url = f"{self.BASE_URL}/d_su_{match_id}"
        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    log.debug("Flashscore returned %d for %s", resp.status, match_id)
                    return None

                data = await resp.json(content_type=None)

                # Check if score changed
                last = self._last_scores.get(match_id, {})
                current_key = (
                    data.get("home_sets"), data.get("away_sets"),
                    data.get("home_games"), data.get("away_games"),
                    data.get("home_points"), data.get("away_points"),
                )
                if current_key == last.get("_key"):
                    return None  # no change

                # Score changed — parse and return
                data["match_id"] = match_id
                event = self.parse_point_update(data)
                if event:
                    self._last_scores[match_id] = {**data, "_key": current_key}
                return event

        except Exception as e:
            log.debug("Flashscore fetch error for %s: %s", match_id, e)
            return None

    async def _reconnect(self) -> None:
        """Close and re-establish the aiohttp session."""
        self._health.record_restart()
        log.info("FlashscoreFeed reconnecting (restart #%d, backoff=%.1fs)",
                 self._health.restart_count, self._backoff_s)

        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception:
            pass

        await asyncio.sleep(self._backoff_s)
        self._backoff_s = min(self._backoff_s * 2, self._max_backoff_s)
        await self.connect()

    # ── Shutdown ──────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Gracefully shut down the feed."""
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
        log.info("FlashscoreFeed shut down")
