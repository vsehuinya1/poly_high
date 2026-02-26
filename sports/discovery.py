"""
Sports market discovery — finds active NBA + football markets on Polymarket,
extracts clobTokenIds and maps to game metadata.
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiohttp

from sports.config import GAMMA_API_URL, SPORTS_SLUG_PATTERNS

log = logging.getLogger("sports.discovery")


@dataclass
class MarketOutcome:
    """Single outcome within a Polymarket sports market."""
    token_id: str
    outcome_label: str          # e.g. "Milwaukee Bucks", "Yes"
    best_bid: float = 0.0
    best_ask: float = 0.0
    last_price: float = 0.0


@dataclass
class SportMarket:
    """Represents one Polymarket sports market (a single game/match)."""
    event_id: str
    market_id: str
    slug: str
    title: str                  # e.g. "Bucks vs. Pelicans"
    sport: str                  # "nba", "football"
    league: str                 # "EPL", "NBA", "La Liga", etc.
    end_date: str
    outcomes: list[MarketOutcome] = field(default_factory=list)
    # Metadata
    volume_24h: float = 0.0
    liquidity: float = 0.0
    spread: float = 1.0
    competitive: float = 0.0
    # Game mapping
    game_id: str = ""           # API-Football fixture ID or NBA game ID
    home_team: str = ""
    away_team: str = ""

    @property
    def all_token_ids(self) -> list[str]:
        return [o.token_id for o in self.outcomes]


def classify_market(slug: str, title: str) -> tuple[str, str]:
    """Classify a market into (sport, league) from its slug/title."""
    s = slug.lower()
    t = title.lower()

    if "nba" in s:
        return ("nba", "NBA")
    if "nfl" in s:
        return ("nfl", "NFL")
    if "nhl" in s:
        return ("nhl", "NHL")
    if "mlb" in s:
        return ("mlb", "MLB")

    # Football leagues
    if "epl-" in s or "premier-league" in s or "premier league" in t:
        return ("football", "EPL")
    if "la-liga" in s or "la liga" in t:
        return ("football", "La Liga")
    if "bundesliga" in s or "bundesliga" in t:
        return ("football", "Bundesliga")
    if "serie-a" in s or "serie a" in t:
        return ("football", "Serie A")
    if "ligue-1" in s or "ligue 1" in t:
        return ("football", "Ligue 1")
    if "eredivisie" in s or "eredivisie" in t:
        return ("football", "Eredivisie")
    if "championship" in s or "championship" in t:
        return ("football", "Championship")
    if "champions-league" in s or "champions league" in t or "ucl-" in s:
        return ("football", "Champions League")
    if "europa-league" in s or "europa league" in t or "uel-" in s:
        return ("football", "Europa League")
    if "conference-league" in s or "uecl-" in s:
        return ("football", "Conference League")
    if "fifa" in s or "world-cup" in s:
        return ("football", "FIFA World Cup")

    return ("unknown", "Unknown")


def is_single_game_market(event: dict) -> bool:
    """Check if an event is a single game (vs. season-long like 'NBA Champion')."""
    slug = event.get("slug", "").lower()
    title = event.get("title", "").lower()

    # Season-long / non-game markets
    season_patterns = [
        "champion", "winner", "mvp", "award", "playoff",
        "medal", "gold", "relegat", "finish", "standings",
        "most-", "top-scorer", "transfer",
    ]
    for p in season_patterns:
        if p in slug or p in title:
            return False

    # Single games typically have "vs" or "v" in title, or team abbreviations in slug
    if " vs" in title or " v " in title:
        return True

    # Check for date-stamped slug pattern: {league}-{team1}-{team2}-{date}
    date_pattern = r"\d{4}-\d{2}-\d{2}$"
    if re.search(date_pattern, slug):
        return True

    return False


def parse_teams_from_title(title: str) -> tuple[str, str]:
    """Extract home/away team names from a match title."""
    # Common patterns: "Team A vs. Team B", "Team A vs Team B", "Team A v Team B"
    for sep in [" vs. ", " vs ", " v "]:
        if sep in title:
            parts = title.split(sep, 1)
            return (parts[0].strip(), parts[1].strip())
    return ("", "")


def extract_token_ids(market: dict) -> list[tuple[str, str]]:
    """Extract (token_id, outcome_label) pairs from a market."""
    results = []
    raw_ids = market.get("clobTokenIds", "[]")
    if isinstance(raw_ids, str):
        try:
            ids = json.loads(raw_ids)
        except json.JSONDecodeError:
            ids = []
    else:
        ids = raw_ids

    raw_outcomes = market.get("outcomes", "[]")
    if isinstance(raw_outcomes, str):
        try:
            outcomes = json.loads(raw_outcomes)
        except json.JSONDecodeError:
            outcomes = []
    else:
        outcomes = raw_outcomes

    raw_prices = market.get("outcomePrices", "[]")
    if isinstance(raw_prices, str):
        try:
            prices = json.loads(raw_prices)
        except json.JSONDecodeError:
            prices = []
    else:
        prices = raw_prices

    for i, tid in enumerate(ids):
        label = outcomes[i] if i < len(outcomes) else f"Outcome_{i}"
        price = float(prices[i]) if i < len(prices) else 0.0
        results.append((tid, label, price))

    return results


async def discover_sports_markets(session: aiohttp.ClientSession) -> list[SportMarket]:
    """Fetch all active single-game sports markets from Polymarket Gamma API."""
    all_markets: list[SportMarket] = []
    seen_event_ids = set()

    for page in range(20):
        params = {
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "limit": "100",
            "offset": str(page * 100),
        }

        try:
            async with session.get(
                f"{GAMMA_API_URL}/events",
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    log.warning("gamma API returned %d", resp.status)
                    break
                events = await resp.json()
        except Exception as e:
            log.error("gamma API request failed: %s", e)
            break

        if not events:
            break

        for event in events:
            event_id = event.get("id", "")
            if event_id in seen_event_ids:
                continue
            seen_event_ids.add(event_id)

            slug = event.get("slug", "")
            title = event.get("title", "")

            # Check if it's a sports market
            is_sports = any(p in slug.lower() for p in SPORTS_SLUG_PATTERNS)
            if not is_sports:
                continue

            # Only individual games, not season-long markets
            if not is_single_game_market(event):
                continue

            sport, league = classify_market(slug, title)
            if sport not in ("nba", "football"):
                continue

            # Process sub-markets within the event
            sub_markets = event.get("markets", [])
            if not sub_markets:
                continue

            # Use the main moneyline market (first market, or the one without spread/total)
            primary = sub_markets[0]
            market_id = primary.get("id", "")

            home_team, away_team = parse_teams_from_title(title)

            # Only use primary moneyline market tokens — sub-market tokens
            # (spreads, totals, props) don't have valid CLOB orderbooks and
            # cause WS INVALID OPERATION when subscribed to.
            outcomes = []
            for tid, label, price in extract_token_ids(primary):
                outcomes.append(MarketOutcome(
                    token_id=tid,
                    outcome_label=label,
                    last_price=price,
                ))

            sm = SportMarket(
                event_id=event_id,
                market_id=market_id,
                slug=slug,
                title=title,
                sport=sport,
                league=league,
                end_date=event.get("endDate", "")[:10],
                outcomes=outcomes,
                volume_24h=float(event.get("volume24hr", 0) or 0),
                liquidity=float(event.get("liquidity", 0) or 0),
                spread=float(primary.get("spread", 1) or 1),
                competitive=float(event.get("competitive", 0) or 0),
                home_team=home_team,
                away_team=away_team,
            )
            all_markets.append(sm)

        log.info("discovery page %d: %d sports markets so far", page + 1, len(all_markets))

        # If very few results, stop early
        if len(events) < 100:
            break

    log.info("discovered %d single-game sports markets", len(all_markets))
    return all_markets
