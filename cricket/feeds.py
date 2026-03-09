"""
Cricket live score feed — ESPN Cricinfo.

Polls ESPN's cricket scoreboard API for live match data.
Same architecture as football/NBA feeds in sports/feeds.py.

Endpoints:
    https://site.api.espn.com/apis/site/v2/sports/cricket/{league}/scoreboard

Supported leagues:
    - icc-mens-t20-world-cup
    - indian-premier-league
    - big-bash-league
    - t20-blast
    - icc-cricket-world-cup
    - 8604 (generic T20I schedule)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from cricket.state import CricketState, InningsPhase

log = logging.getLogger("cricket.feeds")


# ═══════════════════════════════════════════════════════════════════════
#  ESPN Cricket Configuration
# ═══════════════════════════════════════════════════════════════════════

ESPN_CRICKET_BASE = "https://site.api.espn.com/apis/site/v2/sports/cricket"

# Leagues to poll — covers all major T20/ODI tournaments
ESPN_CRICKET_LEAGUES = [
    "icc-mens-t20-world-cup",
    "indian-premier-league",
    "big-bash-league",
    "t20-blast",
    "icc-cricket-world-cup",
    "icc-world-test-championship",
    # SA20, CPL, PSL, etc. can be added
]

# Dynamic discovery endpoint — returns ALL active cricket globally
ESPN_HEADER_URL = "https://site.api.espn.com/apis/v2/scoreboard/header"

CRICKET_POLL_INTERVAL_S = 30  # poll every 30 seconds


# ═══════════════════════════════════════════════════════════════════════
#  ESPN Data Parser
# ═══════════════════════════════════════════════════════════════════════

def _parse_innings_data(event_data: dict) -> dict:
    """Parse innings details from ESPN competition data.

    Returns dict with:
        batting_team, bowling_team, innings,
        runs, wickets, overs, balls,
        run_rate, required_run_rate, target,
        recent_over_runs, recent_wickets
    """
    result = {
        "innings": 1,
        "runs": 0,
        "wickets": 0,
        "overs": 0.0,
        "balls": 0,
        "run_rate": 0.0,
        "required_run_rate": 0.0,
        "target_score": 0,
        "first_innings_total": 0,
        "batting_team": "",
        "bowling_team": "",
        "recent_over_runs": (),
        "recent_wickets": (),
    }

    comps = event_data.get("competitions", [])
    if not comps:
        return result
    comp = comps[0]

    # Get competitors
    competitors = comp.get("competitors", [])
    if len(competitors) < 2:
        return result

    # Parse score from each competitor
    team_scores = {}
    for c in competitors:
        team_name = c.get("team", {}).get("displayName", "Unknown")
        team_id = c.get("id", "")
        score_str = c.get("score", "0")
        linescores = c.get("linescores", [])

        team_scores[team_id] = {
            "name": team_name,
            "score_str": score_str,
            "linescores": linescores,
            "order": c.get("order", 0),
        }

    # Parse the current innings state from situation/note
    situation = comp.get("situation", {}) or {}
    status_detail = event_data.get("status", {}).get("type", {}).get("detail", "")
    note = comp.get("notes", [{}])

    # Try to extract innings info from situation
    last_batting = situation.get("lastBattingTeam", {}) or {}
    batting_team_id = last_batting.get("id", "")

    # Extract runs/wickets/overs from the score display
    # ESPN format: "180/5 (16.3 ov)"
    for tid, ts in team_scores.items():
        score_text = ts.get("score_str", "")
        if "/" in score_text:
            # Parse "180/5"
            parts = score_text.split("/")
            try:
                runs = int(parts[0])
                wickets = int(parts[1].split()[0]) if parts[1] else 0
            except (ValueError, IndexError):
                runs = 0
                wickets = 0

            # If this team is currently batting, use their score
            if tid == batting_team_id or (not batting_team_id):
                result["runs"] = runs
                result["wickets"] = wickets
                result["batting_team"] = ts["name"]

    # Find bowling team
    for tid, ts in team_scores.items():
        if ts["name"] != result["batting_team"]:
            result["bowling_team"] = ts["name"]
            break

    # Parse overs from status detail (e.g., "India 120/3 (14.2 ov)")
    if "ov)" in status_detail:
        try:
            ov_part = status_detail.split("(")[1].split("ov)")[0].strip()
            result["overs"] = float(ov_part)
            # Convert overs to balls: 14.3 → 14*6 + 3 = 87
            over_int = int(result["overs"])
            balls_part = round((result["overs"] - over_int) * 10)
            result["balls"] = over_int * 6 + balls_part
        except (IndexError, ValueError):
            pass

    # Determine innings number
    # If both teams have linescores, 2nd innings is in progress
    teams_with_scores = sum(1 for ts in team_scores.values()
                           if ts.get("score_str", "0") not in ("0", ""))
    if teams_with_scores >= 2:
        result["innings"] = 2
        # First innings total from the other team
        for tid, ts in team_scores.items():
            if ts["name"] != result["batting_team"]:
                try:
                    first_score = ts["score_str"].split("/")[0]
                    result["first_innings_total"] = int(first_score)
                    result["target_score"] = result["first_innings_total"] + 1
                except (ValueError, IndexError):
                    pass

    # Calculate run rate
    if result["overs"] > 0:
        result["run_rate"] = round(result["runs"] / result["overs"], 2)

    # Calculate required run rate (2nd innings only)
    if result["innings"] == 2 and result["target_score"] > 0:
        overs_remaining = 20.0 - result["overs"]  # T20 default
        if overs_remaining > 0:
            runs_needed = result["target_score"] - result["runs"]
            result["required_run_rate"] = round(runs_needed / overs_remaining, 2)

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Cricket Feed
# ═══════════════════════════════════════════════════════════════════════

class CricketFeed:
    """ESPN Cricinfo live score feed for cricket matches.

    Polls ESPN's scoreboard API across multiple leagues.
    Produces CricketState snapshots for each live match.
    """

    def __init__(self):
        self.games: dict[str, CricketState] = {}
        self._last_poll_ts = 0.0
        self._poll_count = 0

    async def fetch_live_scores(
        self,
        session: aiohttp.ClientSession,
    ) -> dict[str, CricketState]:
        """Fetch live cricket scores from ESPN.

        Polls all configured leagues and updates self.games
        with CricketState snapshots for each live/recent match.

        Returns:
            Dict of game_id → CricketState for live matches.
        """
        now = time.time()
        total_live = 0

        for league_code in ESPN_CRICKET_LEAGUES:
            url = f"{ESPN_CRICKET_BASE}/{league_code}/scoreboard"

            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        log.debug("CRICKET_FEED_ERR | %s | HTTP %d",
                                  league_code, resp.status)
                        continue
                    data = await resp.json()
            except Exception as e:
                log.error("CRICKET_FEED_ERR | %s | %s", league_code, e)
                continue

            events = data.get("events", [])

            for evt in events:
                evt_status = evt.get("status", {})
                status_type = evt_status.get("type", {})
                state = status_type.get("state", "")

                # We care about live ("in") and just-finished ("post")
                if state not in ("in", "post"):
                    continue

                event_id = str(evt.get("id", ""))
                event_name = evt.get("name", "")

                # Parse innings data
                innings = _parse_innings_data(evt)

                # Determine match format from league
                match_format = "t20"  # default
                league_lower = league_code.lower()
                if "test" in league_lower:
                    match_format = "test"
                elif "world-cup" in league_lower and "t20" not in league_lower:
                    match_format = "odi"

                # Determine teams from competitors
                comps = evt.get("competitions", [{}])
                comp = comps[0] if comps else {}
                competitors = comp.get("competitors", [])

                home_team = ""
                away_team = ""
                for c in competitors:
                    team_name = c.get("team", {}).get("displayName", "")
                    if c.get("homeAway") == "home":
                        home_team = team_name
                    else:
                        away_team = team_name

                # Build venue
                venue_obj = comp.get("venue", {})
                venue = venue_obj.get("fullName", venue_obj.get("shortName", ""))

                # Map status
                if state == "post":
                    match_status = "finished"
                else:
                    match_status = "live"

                # Determine overs_remaining based on format
                total_overs = 20.0 if match_format == "t20" else (50.0 if match_format == "odi" else 450.0)

                cs = CricketState(
                    match_id=event_id,
                    format=match_format,
                    team_a=home_team,
                    team_b=away_team,
                    batting_team=innings["batting_team"] or home_team,
                    bowling_team=innings["bowling_team"] or away_team,
                    innings=innings["innings"],
                    runs=innings["runs"],
                    wickets=innings["wickets"],
                    overs=innings["overs"],
                    balls=innings["balls"],
                    run_rate=innings["run_rate"],
                    required_run_rate=innings["required_run_rate"],
                    target_score=innings["target_score"],
                    first_innings_total=innings["first_innings_total"],
                    recent_over_runs=innings.get("recent_over_runs", ()),
                    recent_wickets=innings.get("recent_wickets", ()),
                    venue=venue,
                    timestamp=now,
                    status=match_status,
                )

                self.games[event_id] = cs

                if match_status == "live":
                    total_live += 1
                    log.info(
                        "CRICKET_LIVE | %s | %s | %s",
                        league_code, event_name, cs,
                    )

            # Brief pause between leagues
            await asyncio.sleep(0.3)

        # ── Dynamic discovery via ESPN header API ─────────────────
        # This catches ALL active cricket globally, including bilateral
        # series, qualifiers, and domestic leagues not in our static list.
        try:
            async with session.get(
                ESPN_HEADER_URL,
                params={"sport": "cricket", "limit": "50"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    header = await resp.json()
                    for sport_block in header.get("sports", []):
                        for league in sport_block.get("leagues", []):
                            league_name = league.get("name", "")
                            league_id = league.get("id", "")
                            for hdr_evt in league.get("events", []):
                                hdr_state = hdr_evt.get("status", "")
                                # Only care about live and upcoming
                                if hdr_state not in ("in", "pre"):
                                    continue

                                hdr_id = str(hdr_evt.get("id", ""))
                                if hdr_id in self.games:
                                    continue  # already have from league poll

                                # Parse teams from header event
                                competitors = hdr_evt.get("competitors", [])
                                home_team = ""
                                away_team = ""
                                for c in competitors:
                                    if c.get("homeAway") == "home":
                                        home_team = c.get("displayName", "")
                                    else:
                                        away_team = c.get("displayName", "")

                                if not home_team and not away_team:
                                    # Fallback: just use first two competitor names
                                    if len(competitors) >= 2:
                                        home_team = competitors[0].get("displayName", "")
                                        away_team = competitors[1].get("displayName", "")

                                # Determine format from league name
                                match_format = "t20"  # default
                                ln = league_name.lower()
                                if "test" in ln:
                                    match_format = "test"
                                elif "odi" in ln or ("world cup" in ln and "t20" not in ln):
                                    match_format = "odi"

                                # Build minimal CricketState from header data
                                cs = CricketState(
                                    match_id=hdr_id,
                                    format=match_format,
                                    team_a=home_team,
                                    team_b=away_team,
                                    batting_team=home_team,
                                    bowling_team=away_team,
                                    innings=1,
                                    runs=0,
                                    wickets=0,
                                    overs=0.0,
                                    balls=0,
                                    run_rate=0.0,
                                    required_run_rate=0.0,
                                    target_score=0,
                                    first_innings_total=0,
                                    recent_over_runs=(),
                                    recent_wickets=(),
                                    venue="",
                                    timestamp=now,
                                    status="live" if hdr_state == "in" else "finished",
                                )
                                self.games[hdr_id] = cs

                                if hdr_state == "in":
                                    total_live += 1
                                log.info(
                                    "CRICKET_HEADER_FOUND | %s | %s vs %s | league=%s | state=%s",
                                    hdr_id, home_team, away_team, league_name, hdr_state,
                                )

                                # Also fetch full scoreboard for this league
                                # if we haven't already
                                if league_id and league_id not in ESPN_CRICKET_LEAGUES:
                                    try:
                                        full_url = f"{ESPN_CRICKET_BASE}/{league_id}/scoreboard"
                                        async with session.get(
                                            full_url,
                                            timeout=aiohttp.ClientTimeout(total=10),
                                        ) as full_resp:
                                            if full_resp.status == 200:
                                                full_data = await full_resp.json()
                                                for evt in full_data.get("events", []):
                                                    eid = str(evt.get("id", ""))
                                                    evt_state = evt.get("status", {}).get("type", {}).get("state", "")
                                                    if evt_state not in ("in", "post"):
                                                        continue
                                                    if eid in self.games:
                                                        # Update with richer data
                                                        innings = _parse_innings_data(evt)
                                                        existing = self.games[eid]
                                                        existing.runs = innings["runs"]
                                                        existing.wickets = innings["wickets"]
                                                        existing.overs = innings["overs"]
                                                        existing.balls = innings["balls"]
                                                        existing.run_rate = innings["run_rate"]
                                                        existing.required_run_rate = innings["required_run_rate"]
                                                        existing.target_score = innings["target_score"]
                                                        existing.first_innings_total = innings["first_innings_total"]
                                                        existing.batting_team = innings["batting_team"] or existing.team_a
                                                        existing.bowling_team = innings["bowling_team"] or existing.team_b
                                                        existing.innings = innings["innings"]
                                                        existing.timestamp = now
                                                        log.info("CRICKET_FULL_UPDATE | %s | %s", eid, existing)
                                    except Exception as e:
                                        log.debug("CRICKET_FULL_FETCH_ERR | league=%s | %s", league_id, e)
                                    await asyncio.sleep(0.3)
                else:
                    log.debug("CRICKET_HEADER_ERR | HTTP %d", resp.status)
        except Exception as e:
            log.warning("CRICKET_HEADER_ERR | %s", e)

        self._poll_count += 1
        self._last_poll_ts = now

        if total_live > 0 or self._poll_count % 20 == 0:
            log.info("CRICKET_FEED | live=%d total=%d poll=%d",
                     total_live, len(self.games), self._poll_count)

        if total_live == 0 and len(self.games) == 0 and self._poll_count % 20 == 0:
            log.warning("CRICKET_FEED_WARN | zero live matches across all endpoints")

        return {gid: gs for gid, gs in self.games.items()}

    @property
    def live_games(self) -> dict[str, CricketState]:
        """Return only currently live games."""
        return {gid: gs for gid, gs in self.games.items()}


# ═══════════════════════════════════════════════════════════════════════
#  Cricket CSV Logger — for post-match analysis
# ═══════════════════════════════════════════════════════════════════════

import csv
import os
from pathlib import Path
from cricket.strategy import CricketSignal


class CricketCSVLogger:
    """Logs cricket signals and state snapshots to CSV for post-match analysis."""

    def __init__(self, data_dir: str = "sports_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._signal_file: Optional[object] = None
        self._signal_writer: Optional[csv.writer] = None
        self._state_file: Optional[object] = None
        self._state_writer: Optional[csv.writer] = None
        self._initialized = False

    def _init_files(self, date_str: str):
        """Initialize CSV files for the day."""
        if self._initialized:
            return

        # Signal log
        sig_path = self.data_dir / f"cricket_signals_{date_str}.csv"
        sig_exists = sig_path.exists()
        self._signal_file = open(sig_path, "a", newline="")
        self._signal_writer = csv.writer(self._signal_file)
        if not sig_exists:
            self._signal_writer.writerow([
                "timestamp", "match_id", "signal_type", "edge",
                "fair_price", "market_price", "direction",
                "runs", "wickets", "overs", "run_rate", "required_rr",
                "rolling_rr_3", "resource_pct", "momentum_factor",
                "latency_ms", "is_tradeable",
                "pitch_bias", "anchor_trap",
            ])

        # State log (every poll)
        state_path = self.data_dir / f"cricket_states_{date_str}.csv"
        state_exists = state_path.exists()
        self._state_file = open(state_path, "a", newline="")
        self._state_writer = csv.writer(self._state_file)
        if not state_exists:
            self._state_writer.writerow([
                "timestamp", "match_id", "innings", "batting_team",
                "runs", "wickets", "overs", "run_rate", "required_rr",
                "target", "phase", "market_price",
            ])

        self._initialized = True

    def log_signal(self, signal: CricketSignal):
        """Log a signal to CSV."""
        from datetime import datetime
        date_str = datetime.utcnow().strftime("%Y%m%d")
        self._init_files(date_str)

        s = signal.state_snapshot
        m = signal.model_output
        self._signal_writer.writerow([
            f"{signal.timestamp:.3f}",
            signal.match_id,
            signal.signal_type,
            f"{signal.edge:.4f}",
            f"{signal.fair_price:.4f}",
            f"{signal.market_price:.4f}",
            signal.direction,
            s.runs, s.wickets, s.overs,
            f"{s.run_rate:.2f}",
            f"{s.required_run_rate:.2f}",
            f"{s.rolling_run_rate_3:.2f}",
            f"{m.resource_pct:.1f}",
            f"{m.momentum_factor:.3f}",
            f"{signal.latency_ms:.0f}",
            signal.is_tradeable,
            signal.pitch_bias_active,
            signal.anchor_trap_active,
        ])
        self._signal_file.flush()

    def log_state(self, state: CricketState, market_price: float = 0.0):
        """Log a state snapshot to CSV."""
        from datetime import datetime
        date_str = datetime.utcnow().strftime("%Y%m%d")
        self._init_files(date_str)

        self._state_writer.writerow([
            f"{state.timestamp:.3f}",
            state.match_id,
            state.innings,
            state.batting_team,
            state.runs,
            state.wickets,
            state.overs,
            f"{state.run_rate:.2f}",
            f"{state.required_run_rate:.2f}",
            state.target_score,
            state.phase.value,
            f"{market_price:.4f}",
        ])
        self._state_file.flush()

    def close(self):
        if self._signal_file:
            self._signal_file.close()
        if self._state_file:
            self._state_file.close()
