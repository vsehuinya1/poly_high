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
    # April 25: Delhi Capitals vs Punjab Kings (LIVE)
    69629: FixtureMapping(
        fixture_id=69629,
        home_team="Delhi Capitals",
        away_team="Punjab Kings",
        poly_token_yes="50126410203778007142503549338936271387140662701828838552454333525503628755261",
        poly_token_no="18575430438358390541772074127628816002193416137521174798561427510895122834765",
        poly_market_title="Indian Premier League: Delhi Capitals vs Punjab Kings",
    ),
    # April 25: Rajasthan Royals vs Sunrisers Hyderabad (Tonight)
    69630: FixtureMapping(
        fixture_id=69630,
        home_team="Rajasthan Royals",
        away_team="Sunrisers Hyderabad",
        poly_token_yes="60313963775699758188097104423721190530422027186233362742822676617353903244085",
        poly_token_no="71367208209871569480963362449047716278879223456490982543510091929629209420090",
        poly_market_title="Indian Premier League: Rajasthan Royals vs Sunrisers Hyderabad",
    ),
    # April 26: Chennai Super Kings vs Gujarat Titans
    69661: FixtureMapping(
        fixture_id=69661,
        home_team="Chennai Super Kings",
        away_team="Gujarat Titans",
        poly_token_yes="7019700144365428912679939873052023465481022828774222035513408366152052803033",
        poly_token_no="74045649422532066563684830072567085635074538814335742772911880217946270689339",
        poly_market_title="Indian Premier League: Chennai Super Kings vs Gujarat Titans",
    ),
    # April 27: Delhi Capitals vs Royal Challengers Bengaluru
    69639: FixtureMapping(
        fixture_id=69639,
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
