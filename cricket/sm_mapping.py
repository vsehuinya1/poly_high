"""
Cricket Fixture → Polymarket Token Mapping.

v2.0 — Auto-Discovery Mode (2026-04-30)

Instead of hard-coding each match, this module:
  1. Fetches ALL upcoming IPL fixtures from Sportmonks at startup
  2. Matches team names from Polymarket discovery to Sportmonks fixtures
  3. Automatically builds the mapping — no manual updates ever needed

The fallback static map is kept for safety but should never be needed.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger("cricket.sm_mapping")


@dataclass
class FixtureMapping:
    """Maps a Sportmonks fixture to Polymarket tokens."""
    fixture_id: int
    home_team: str
    away_team: str
    poly_token_yes: str    # YES token (home team wins)
    poly_token_no: str     # NO token (away team wins / complement)
    poly_market_title: str = ""


# ═════════════════════════════════════════════════════════════════════
#  IPL Team Name Normalization
#
#  Maps various forms of team names to a canonical short key.
#  Used for fuzzy matching between Sportmonks and Polymarket.
# ═════════════════════════════════════════════════════════════════════

_TEAM_ALIASES: dict[str, str] = {
    "mumbai indians": "MI",
    "mumbai": "MI",
    "mi": "MI",
    "chennai super kings": "CSK",
    "chennai": "CSK",
    "csk": "CSK",
    "royal challengers bengaluru": "RCB",
    "royal challengers bangalore": "RCB",
    "bengaluru": "RCB",
    "bangalore": "RCB",
    "rcb": "RCB",
    "kolkata knight riders": "KKR",
    "kolkata": "KKR",
    "kkr": "KKR",
    "sunrisers hyderabad": "SRH",
    "hyderabad": "SRH",
    "srh": "SRH",
    "rajasthan royals": "RR",
    "rajasthan": "RR",
    "rr": "RR",
    "delhi capitals": "DC",
    "delhi": "DC",
    "dc": "DC",
    "punjab kings": "PBKS",
    "punjab": "PBKS",
    "pbks": "PBKS",
    "lucknow super giants": "LSG",
    "lucknow": "LSG",
    "lsg": "LSG",
    "gujarat titans": "GT",
    "gujarat": "GT",
    "gt": "GT",
}


def _normalize_team(name: str) -> str:
    """Normalize a team name to a canonical short key."""
    n = name.strip().lower()
    return _TEAM_ALIASES.get(n, n.upper())


def _extract_teams_from_title(title: str) -> tuple[str, str]:
    """Extract two team keys from a Polymarket or Sportmonks title.

    'Indian Premier League: Gujarat Titans vs Royal Challengers Bengaluru'
    → ('GT', 'RCB')
    """
    t = title.lower()
    # Strip league prefix
    if ":" in t:
        t = t.rsplit(":", 1)[-1].strip()
    # Strip prop suffix
    if " - " in t:
        t = t.split(" - ", 1)[0].strip()

    for sep in [" vs. ", " vs "]:
        if sep in t:
            a, b = t.split(sep, 1)
            return _normalize_team(a.strip()), _normalize_team(b.strip())
    return ("?", "?")


# ═════════════════════════════════════════════════════════════════════
#  Runtime State
# ═════════════════════════════════════════════════════════════════════

# Sportmonks fixtures indexed by match key ("GT_vs_RCB" → fixture data)
_sm_fixtures: dict[str, dict] = {}

# Active mappings (fixture_id → FixtureMapping)
IPL_FIXTURE_MAP: dict[int, FixtureMapping] = {}


def load_sportmonks_fixtures(fixtures_data: list[dict]) -> int:
    """Load Sportmonks fixture list into memory for auto-matching.

    Called at startup with the result of the Sportmonks API query.
    Returns the number of fixtures loaded.
    """
    global _sm_fixtures
    _sm_fixtures.clear()

    for f in fixtures_data:
        fid = f.get("id", 0)
        lt = f.get("localteam", {}).get("name", "")
        vt = f.get("visitorteam", {}).get("name", "")
        status = f.get("status", "")
        starting_at = f.get("starting_at", "")

        if not lt or not vt:
            continue

        home_key = _normalize_team(lt)
        away_key = _normalize_team(vt)

        # Index by both orderings so matching works regardless of home/away
        key1 = f"{home_key}_vs_{away_key}"
        key2 = f"{away_key}_vs_{home_key}"

        entry = {
            "fixture_id": fid,
            "home_team": lt,
            "away_team": vt,
            "home_key": home_key,
            "away_key": away_key,
            "status": status,
            "starting_at": starting_at,
        }

        # For duplicate keys (same matchup played twice), keep the one
        # closest to now that isn't Finished
        for key in [key1, key2]:
            existing = _sm_fixtures.get(key)
            if existing is None:
                _sm_fixtures[key] = entry
            elif existing["status"] == "Finished" and status != "Finished":
                _sm_fixtures[key] = entry
            elif (existing["status"] != "Finished" and status != "Finished"
                  and starting_at > existing["starting_at"]):
                # Keep the later one (next upcoming match)
                pass  # existing is earlier, keep it
            elif (existing["status"] != "Finished" and status != "Finished"
                  and starting_at < existing["starting_at"]):
                _sm_fixtures[key] = entry

    log.info("CRICKET_SM_FIXTURES_LOADED | total=%d | keys=%d",
             len(fixtures_data), len(_sm_fixtures))
    return len(_sm_fixtures)


def auto_map_from_polymarket(
    poly_title: str,
    poly_token_yes: str,
    poly_token_no: str,
) -> FixtureMapping | None:
    """Attempt to auto-map a Polymarket market to a Sportmonks fixture.

    Called by the cricket signal loop when it discovers an IPL market.
    Returns a FixtureMapping if a match is found, else None.
    """
    team_a, team_b = _extract_teams_from_title(poly_title)
    if team_a == "?" or team_b == "?":
        log.info("CRICKET_AUTOMAP_FAIL | title=%s | parse_fail", poly_title[:60])
        return None

    key = f"{team_a}_vs_{team_b}"
    entry = _sm_fixtures.get(key)

    if entry is None:
        log.info("CRICKET_AUTOMAP_MISS | title=%s | key=%s | no SM fixture",
                 poly_title[:60], key)
        return None

    fid = entry["fixture_id"]

    # Check if already mapped
    if fid in IPL_FIXTURE_MAP:
        return IPL_FIXTURE_MAP[fid]

    mapping = FixtureMapping(
        fixture_id=fid,
        home_team=entry["home_team"],
        away_team=entry["away_team"],
        poly_token_yes=poly_token_yes,
        poly_token_no=poly_token_no,
        poly_market_title=poly_title,
    )

    IPL_FIXTURE_MAP[fid] = mapping
    log.info(
        "CRICKET_AUTOMAP_OK | fixture=%d | %s vs %s | poly=%s | starts=%s",
        fid, entry["home_team"], entry["away_team"],
        poly_title[:50], entry["starting_at"],
    )
    return mapping


# ═════════════════════════════════════════════════════════════════════
#  Legacy API (backward compatible)
# ═════════════════════════════════════════════════════════════════════

def get_mapping(fixture_id: int) -> FixtureMapping | None:
    """Look up Polymarket tokens for a Sportmonks fixture."""
    return IPL_FIXTURE_MAP.get(fixture_id)


def get_all_fixture_ids() -> list[int]:
    """Return all mapped fixture IDs for polling."""
    return list(IPL_FIXTURE_MAP.keys())


def add_mapping(
    fixture_id: int,
    home_team: str,
    away_team: str,
    poly_token_yes: str,
    poly_token_no: str,
    poly_market_title: str = "",
) -> None:
    """Add or update a fixture mapping at runtime."""
    mapping = FixtureMapping(
        fixture_id=fixture_id,
        home_team=home_team,
        away_team=away_team,
        poly_token_yes=poly_token_yes,
        poly_token_no=poly_token_no,
        poly_market_title=poly_market_title,
    )
    IPL_FIXTURE_MAP[fixture_id] = mapping
    log.info(
        "CRICKET_MAPPING_ADD | fixture=%d | %s vs %s | token_yes=%s...%s",
        fixture_id, home_team, away_team,
        poly_token_yes[:8], poly_token_yes[-4:] if len(poly_token_yes) > 8 else "",
    )
