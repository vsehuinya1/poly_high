"""
Flashscore live tennis score feed.

Polls Flashscore's pipe-delimited feed for live tennis match data.
Parses match IDs, player names, set/game scores, and serving status.
Uses tennis/matching.py to map Flashscore players to Polymarket markets.

Flashscore field codes (pipe-delimited, ¬-separated):
    AA÷ = match ID              AE÷ = player 1 name
    AF÷ = player 2 name         AB÷ = status code
    BA÷/BB÷ = set 1 (h/a)      BC÷/BD÷ = set 2
    BE÷/BF÷ = set 3             BG÷/BH÷ = set 4
    BI÷/BJ÷ = set 5             AG÷/AH÷ = current game score
    AO÷ = serving (1=home, 2=away)
    AD÷ = unix timestamp

Status codes:
    1 = not started   2 = in progress (set 1)
    3 = in progress (set 2)   5 = set 3   7 = finished
    9 = walkover/retired
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("tennis.livefeed")

# Flashscore endpoints
_FS_TENNIS_URL = "https://local-global.flashscore.ninja/2/x/feed/f_2_0_3_en-gb_1"
_FS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 12) FlashScore/3.13",
    "Accept": "*/*",
    "X-Fsign": "SW9D1eZo",
}

# Status code mapping
_LIVE_STATUSES = {"2", "3", "5", "6", "7"}  # in-progress or just finished
_FINISHED_STATUSES = {"7", "9"}


@dataclass
class FlashscoreMatch:
    """Parsed match state from Flashscore feed."""
    match_id: str
    player_a: str           # home player
    player_b: str           # away player
    status: str             # raw status code
    is_live: bool = False
    is_finished: bool = False
    sets_a: int = 0         # sets won by player A
    sets_b: int = 0         # sets won by player B
    games_a: int = 0        # games in current set for A
    games_b: int = 0        # games in current set for B
    point_a: str = "0"      # current game point for A
    point_b: str = "0"      # current game point for B
    serving: str = ""       # "a" or "b" — who is serving
    set_scores: list = field(default_factory=list)  # [(a1,b1), (a2,b2), ...]
    timestamp: float = 0.0


def parse_flashscore_feed(raw_data: str) -> list[FlashscoreMatch]:
    """Parse Flashscore pipe-delimited feed into FlashscoreMatch objects.

    The feed format uses ¬ (xac) as field separator and ~AA÷ as match delimiter.
    Each field is a 2-char key followed by ÷ then the value.
    """
    matches = []
    segments = raw_data.split("~AA÷")

    for seg in segments[1:]:  # skip preamble before first match
        fields = {}
        # The match ID is the first thing before any ¬
        parts = seg.split("\xac")
        if parts:
            fields["AA"] = parts[0]  # match ID

        for part in parts[1:]:
            # Each part is like "AE÷Altmaier D."
            sep_idx = part.find("÷")
            if sep_idx >= 1:
                key = part[:sep_idx]
                val = part[sep_idx + 1:]
                fields[key] = val

        match_id = fields.get("AA", "")
        if not match_id:
            continue

        player_a = fields.get("AE", "").strip()
        player_b = fields.get("AF", "").strip()
        status = fields.get("AB", "0")

        if not player_a or not player_b:
            continue

        # Parse set scores
        set_pairs = [
            (fields.get("BA", ""), fields.get("BB", "")),
            (fields.get("BC", ""), fields.get("BD", "")),
            (fields.get("BE", ""), fields.get("BF", "")),
            (fields.get("BG", ""), fields.get("BH", "")),
            (fields.get("BI", ""), fields.get("BJ", "")),
        ]

        set_scores = []
        sets_a = 0
        sets_b = 0
        current_set_games_a = 0
        current_set_games_b = 0

        for i, (sa, sb) in enumerate(set_pairs):
            if sa and sb:
                try:
                    ga = int(sa)
                    gb = int(sb)
                    set_scores.append((ga, gb))

                    # Determine if this set is finished
                    # A set is won at 6 (with 2 game lead) or 7 (tiebreak/advantage)
                    set_won_by_a = (ga >= 6 and ga - gb >= 2) or ga == 7
                    set_won_by_b = (gb >= 6 and gb - ga >= 2) or gb == 7

                    if set_won_by_a:
                        sets_a += 1
                    elif set_won_by_b:
                        sets_b += 1
                    else:
                        # Current in-progress set
                        current_set_games_a = ga
                        current_set_games_b = gb
                except (ValueError, TypeError):
                    pass

        # Current game point score
        point_a = fields.get("AG", "0") or "0"
        point_b = fields.get("AH", "0") or "0"

        # Serving indicator: AO÷1 = home serving, AO÷2 = away serving
        serving_raw = fields.get("AO", "")
        serving = ""
        if serving_raw == "1":
            serving = "a"
        elif serving_raw == "2":
            serving = "b"

        is_live = status in _LIVE_STATUSES
        is_finished = status in _FINISHED_STATUSES

        # Timestamp
        ts_raw = fields.get("AD", "0")
        try:
            ts = float(ts_raw) if ts_raw else 0.0
        except (ValueError, TypeError):
            ts = 0.0

        m = FlashscoreMatch(
            match_id=match_id,
            player_a=player_a,
            player_b=player_b,
            status=status,
            is_live=is_live,
            is_finished=is_finished,
            sets_a=sets_a,
            sets_b=sets_b,
            games_a=current_set_games_a,
            games_b=current_set_games_b,
            point_a=point_a,
            point_b=point_b,
            serving=serving,
            set_scores=set_scores,
            timestamp=ts,
        )
        matches.append(m)

    return matches


class TennisScoreFeed:
    """Polls Flashscore for live tennis match scores.

    This feed:
    1. Fetches all tennis matches from Flashscore every poll_interval_s seconds
    2. Parses player names and scores from the pipe-delimited format
    3. Stores the latest state for each Flashscore match ID
    4. Provides a lookup method to find matches by player name

    Usage in orchestrator:
        feed = TennisScoreFeed(poll_interval_s=5.0)
        await feed.start()     # starts background polling
        match = feed.find_match("Djokovic", "Sinner")
        if match and match.is_live:
            # Update TennisState from match data
            ...
    """

    def __init__(self, poll_interval_s: float = 5.0):
        self.poll_interval_s = poll_interval_s
        self._matches: dict[str, FlashscoreMatch] = {}  # fs_id → match
        self._running = False
        self._session = None
        self._last_poll: float = 0.0
        self._poll_count: int = 0
        self._error_count: int = 0

    async def start(self) -> None:
        """Start the background polling loop."""
        import aiohttp
        self._session = aiohttp.ClientSession(
            headers=_FS_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        self._running = True
        log.info("TennisScoreFeed started (poll=%.1fs)", self.poll_interval_s)

    async def poll_once(self) -> int:
        """Fetch and parse one round of Flashscore data.

        Returns number of matches parsed.
        """
        if not self._session or self._session.closed:
            return 0

        try:
            async with self._session.get(_FS_TENNIS_URL) as resp:
                if resp.status != 200:
                    log.warning("Flashscore returned %d", resp.status)
                    self._error_count += 1
                    return 0

                raw = await resp.text(encoding="utf-8", errors="replace")

            parsed = parse_flashscore_feed(raw)
            for m in parsed:
                self._matches[m.match_id] = m

            self._poll_count += 1
            self._last_poll = time.time()

            live_count = sum(1 for m in parsed if m.is_live)
            if self._poll_count % 60 == 1:  # Log every ~5 minutes at 5s interval
                log.info("FLASHSCORE | total=%d | live=%d | polls=%d | errors=%d",
                         len(parsed), live_count, self._poll_count, self._error_count)

            return len(parsed)

        except Exception as e:
            self._error_count += 1
            if self._error_count % 10 == 1:
                log.error("Flashscore poll error (#%d): %s", self._error_count, e)
            return 0

    async def poll_loop(self) -> None:
        """Background polling loop. Call as asyncio task."""
        while self._running:
            await self.poll_once()
            await asyncio.sleep(self.poll_interval_s)

    def find_match_by_players(self, player_a: str, player_b: str,
                                threshold: float = 0.80) -> Optional[FlashscoreMatch]:
        """Find a Flashscore match by player names using fuzzy matching.

        Args:
            player_a: First player name (from Polymarket).
            player_b: Second player name (from Polymarket).
            threshold: Minimum match score to accept.

        Returns:
            The best matching FlashscoreMatch, or None.
        """
        from tennis.matching import tennis_name_match_score

        best_match = None
        best_score = 0.0

        for fs_match in self._matches.values():
            # Try direct order
            score_direct = (
                tennis_name_match_score(player_a, fs_match.player_a) +
                tennis_name_match_score(player_b, fs_match.player_b)
            ) / 2.0

            # Try reversed
            score_reversed = (
                tennis_name_match_score(player_a, fs_match.player_b) +
                tennis_name_match_score(player_b, fs_match.player_a)
            ) / 2.0

            score = max(score_direct, score_reversed)
            if score > best_score:
                best_score = score
                best_match = fs_match

        if best_score >= threshold and best_match:
            return best_match

        return None

    @property
    def live_matches(self) -> list[FlashscoreMatch]:
        """Return all currently live matches."""
        return [m for m in self._matches.values() if m.is_live]

    @property
    def total_matches(self) -> int:
        return len(self._matches)

    async def shutdown(self) -> None:
        """Shut down the feed."""
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
        log.info("TennisScoreFeed shut down (polls=%d, errors=%d)",
                 self._poll_count, self._error_count)
