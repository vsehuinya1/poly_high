"""
Cricket Fixture → Polymarket Token Mapping.

Hard-coded mapping of Sportmonks fixture IDs to Polymarket token IDs.
This ensures we NEVER trade the wrong market.

v1.0.1 — 2026-04-25
"""
from __future__ import annotations

import logging
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
#  IPL 2026 Hard Mapping
#
#  Updated before each match day. Maps Sportmonks fixture IDs
#  to Polymarket CLOB token IDs.
#
#  To find fixture IDs:
#    curl "https://cricket.sportmonks.com/api/v2.0/fixtures?
#          filter[league_id]=1&sort=-starting_at&api_token=..."
#
#  To find token IDs: check sports/discovery.py output logs
#    CRICKET_CANDIDATE | ... | token_id=...
# ═════════════════════════════════════════════════════════════════════

IPL_FIXTURE_MAP: dict[int, FixtureMapping] = {
    # April 27: Delhi Capitals vs Royal Challengers Bengaluru (39th Match)
    69633: FixtureMapping(
        fixture_id=69633,
        home_team="Delhi Capitals",
        away_team="Royal Challengers Bengaluru",
        poly_token_yes="109682511911087779727834345377492663124053255801388007181934149546406630368131",
        poly_token_no="113634929071500980536308211559860233709108886450747146687804489508961101445567",
        poly_market_title="Indian Premier League: Delhi Capitals vs Royal Challengers Bengaluru",
    ),
}


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
