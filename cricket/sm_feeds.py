"""
Sportmonks v2 Cricket Feed — Fixture-Level Scoreboard Poller.

Polls individual fixtures via:
    GET /v2.0/fixtures/{fixture_id}?include=scoreboards

NOT livescores (delayed/inconsistent).

v1.0.1 — 2026-04-25
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import aiohttp

log = logging.getLogger("cricket.sm_feeds")


@dataclass
class ScoreboardSnapshot:
    """Parsed scoreboard state from a single Sportmonks poll."""
    fixture_id: int
    timestamp: float              # local poll time
    status: str                   # "1st Innings", "2nd Innings", "Finished", etc.
    innings: int                  # 1 or 2 (derived from status / scoreboard key)

    # Batting team state (current innings)
    batting_team_id: int = 0
    runs: int = 0
    wickets: int = 0
    overs: float = 0.0

    # First innings total (for 2nd innings chase context)
    first_innings_total: int = 0
    first_innings_wickets: int = 0
    first_innings_overs: float = 0.0

    # Home / away team IDs for mapping
    home_team_id: int = 0
    away_team_id: int = 0

    # Raw status string
    raw_status: str = ""

    @property
    def balls_bowled(self) -> int:
        """Convert overs (e.g. 12.3) to total balls."""
        full = int(self.overs)
        partial = round((self.overs - full) * 10)
        return full * 6 + partial

    @property
    def run_rate(self) -> float:
        """Current run rate."""
        if self.overs <= 0:
            return 0.0
        return self.runs / self.overs

    @property
    def target(self) -> int:
        """Chase target (first innings total + 1)."""
        return self.first_innings_total + 1 if self.first_innings_total > 0 else 0

    @property
    def runs_remaining(self) -> int:
        """Runs still needed to win (2nd innings only)."""
        if self.innings != 2 or self.target <= 0:
            return 0
        return max(0, self.target - self.runs)

    @property
    def overs_remaining(self) -> float:
        """Overs left (T20 = 20 overs max)."""
        return max(0.0, 20.0 - self.overs)

    @property
    def required_run_rate(self) -> float:
        """Required run rate to win (2nd innings only)."""
        if self.overs_remaining <= 0 or self.runs_remaining <= 0:
            return 0.0
        return self.runs_remaining / self.overs_remaining

    @property
    def wickets_in_hand(self) -> int:
        """Remaining wickets (10 - fallen)."""
        return max(0, 10 - self.wickets)

    @property
    def is_live(self) -> bool:
        """Check if match is currently in play."""
        s = self.raw_status.lower()
        return any(k in s for k in ["1st innings", "2nd innings", "innings break"])


class SmCricketFeed:
    """Async Sportmonks v2 fixture poller.

    Polls each mapped fixture individually for maximum freshness.
    """

    BASE_URL = "https://cricket.sportmonks.com/api/v2.0"

    def __init__(self, api_token: str, poll_interval_s: float = 5.0):
        self.api_token = api_token
        self.poll_interval_s = poll_interval_s
        self._last_poll: dict[int, float] = {}  # fixture_id → last poll ts
        self._poll_count = 0

    async def poll_fixture(
        self,
        session: aiohttp.ClientSession,
        fixture_id: int,
    ) -> ScoreboardSnapshot | None:
        """Poll a single fixture's scoreboard.

        Returns None if API fails or fixture not found.
        Respects poll interval to avoid hammering API.
        """
        now = time.time()
        last = self._last_poll.get(fixture_id, 0)
        if now - last < self.poll_interval_s:
            return None  # too soon

        self._last_poll[fixture_id] = now

        try:
            url = f"{self.BASE_URL}/fixtures/{fixture_id}"
            params = {
                "api_token": self.api_token,
                "include": "scoreboards",
            }
            async with session.get(
                url, params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    log.warning(
                        "CRICKET_SM_POLL_FAIL | fixture=%d | status=%d",
                        fixture_id, resp.status,
                    )
                    return None
                data = await resp.json()

            self._poll_count += 1
            return self._parse_fixture(data.get("data", {}), now)

        except Exception as e:
            log.warning("CRICKET_SM_POLL_ERR | fixture=%d | %s", fixture_id, e)
            return None

    def _parse_fixture(self, fix: dict, ts: float) -> ScoreboardSnapshot | None:
        """Parse fixture JSON into ScoreboardSnapshot."""
        if not fix:
            return None

        fixture_id = fix.get("id", 0)
        status = fix.get("status", "")
        home_id = fix.get("localteam_id", 0)
        away_id = fix.get("visitorteam_id", 0)

        scoreboards = fix.get("scoreboards", [])
        if not scoreboards:
            return None

        # Parse all scoreboard entries — separate by innings (S1, S2)
        innings_data: dict[str, dict] = {}  # "S1" / "S2" → {team_id, runs, wickets, overs}
        for sb in scoreboards:
            if sb.get("type") != "total":
                continue
            key = sb.get("scoreboard", "S1")  # "S1" or "S2"
            innings_data[key] = {
                "team_id": sb.get("team_id", 0),
                "runs": sb.get("total", 0),
                "wickets": sb.get("wickets", 0),
                "overs": sb.get("overs", 0),
            }

        if not innings_data:
            return None

        # Determine current innings
        has_s2 = "S2" in innings_data
        current_key = "S2" if has_s2 else "S1"
        current = innings_data[current_key]
        innings_num = 2 if has_s2 else 1

        # First innings data (for 2nd innings context)
        first = innings_data.get("S1", {})

        snapshot = ScoreboardSnapshot(
            fixture_id=fixture_id,
            timestamp=ts,
            status=status,
            innings=innings_num,
            batting_team_id=current.get("team_id", 0),
            runs=current.get("runs", 0),
            wickets=current.get("wickets", 0),
            overs=float(current.get("overs", 0)),
            first_innings_total=first.get("runs", 0) if has_s2 else 0,
            first_innings_wickets=first.get("wickets", 0) if has_s2 else 0,
            first_innings_overs=float(first.get("overs", 0)) if has_s2 else 0.0,
            home_team_id=home_id,
            away_team_id=away_id,
            raw_status=status,
        )

        log.info(
            "CRICKET_SM_POLL | fixture=%d | status=%s | inn=%d | "
            "%d/%d (%.1f ov) | target=%d | RRR=%.1f",
            fixture_id, status, innings_num,
            snapshot.runs, snapshot.wickets, snapshot.overs,
            snapshot.target, snapshot.required_run_rate,
        )

        return snapshot

    @property
    def stats_line(self) -> str:
        """Status string for periodic logging."""
        return f"SM polls={self._poll_count}"
