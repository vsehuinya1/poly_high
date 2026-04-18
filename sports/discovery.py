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

from sports.config import GAMMA_API_URL, SPORTS_SLUG_PATTERNS, FOOTBALL_ALLOWED_COMPETITIONS

log = logging.getLogger("sports.discovery")


# ── Cricket team-based heuristic detection ────────────────────────────
CRICKET_NATIONS = {
    "india", "new zealand", "australia", "england", "pakistan",
    "south africa", "sri lanka", "bangladesh", "west indies",
    "afghanistan", "netherlands", "zimbabwe", "ireland",
    "scotland", "namibia", "nepal", "usa", "uae", "oman",
    "canada", "papua new guinea", "cayman islands",
    "argentina", "suriname", "mexico", "bermuda", "hong kong",
    "kenya", "qatar", "bahrain", "botswana", "lesotho",
}

# ── IPL franchise names (robust fuzzy matching) ───────────────────────
# All 10 current IPL teams + common abbreviations / short forms
IPL_TEAMS = {
    # Full names
    "mumbai indians", "chennai super kings", "royal challengers bengaluru",
    "royal challengers bangalore", "kolkata knight riders",
    "sunrisers hyderabad", "rajasthan royals", "delhi capitals",
    "punjab kings", "lucknow super giants", "gujarat titans",
    # Short forms / abbreviations
    "mumbai", "chennai", "kolkata", "hyderabad", "rajasthan",
    "delhi", "punjab", "lucknow", "gujarat", "bengaluru", "bangalore",
    # Abbreviation codes
    "mi", "csk", "rcb", "kkr", "srh", "rr", "dc", "pbks", "lsg", "gt",
}

# City→franchise mapping for fuzzy matching
_IPL_CITY_TO_TEAM = {
    "mumbai": "mumbai indians", "chennai": "chennai super kings",
    "bengaluru": "royal challengers bengaluru",
    "bangalore": "royal challengers bangalore",
    "kolkata": "kolkata knight riders", "hyderabad": "sunrisers hyderabad",
    "rajasthan": "rajasthan royals", "jaipur": "rajasthan royals",
    "delhi": "delhi capitals", "punjab": "punjab kings",
    "mohali": "punjab kings", "lucknow": "lucknow super giants",
    "gujarat": "gujarat titans", "ahmedabad": "gujarat titans",
}


def _is_ipl_match(title: str) -> bool:
    """Check if a market title is an IPL match (contains two IPL team names).

    Robust fuzzy matching:
    - Checks full team names
    - Checks city names
    - Case-insensitive, handles 'vs.' and 'vs'
    """
    t = title.lower()
    # Must have "vs" somewhere
    has_vs = False
    for sep in [" vs. ", " vs "]:
        if sep in t:
            has_vs = True
            break
    if not has_vs:
        return False

    # Count how many IPL team references appear
    matches = 0
    matched_teams = []
    for team in IPL_TEAMS:
        if len(team) <= 3:  # skip abbreviations for title matching (too short)
            continue
        if team in t:
            matches += 1
            matched_teams.append(team)
            if matches >= 2:
                log.info("DISCOVERY | IPL candidate: %s | teams=%s",
                         title, matched_teams)
                return True

    # Fallback: check city names
    for city in _IPL_CITY_TO_TEAM:
        if len(city) <= 3:
            continue
        if city in t and city not in [m for m in matched_teams]:
            matches += 1
            matched_teams.append(city)
            if matches >= 2:
                log.info("DISCOVERY | IPL candidate (city): %s | teams=%s",
                         title, matched_teams)
                return True

    return False


def _is_cricket_match(title: str) -> bool:
    """Check if a market title contains two cricket nations (Team A vs Team B)."""
    t = title.lower()
    # Strip common suffixes that might confuse matching
    for suffix in ["(final)", "(semi-final)", "(qualifier)",
                   "(group a)", "(group b)", "(group c)", "(group d)",
                   "(1st t20i)", "(2nd t20i)", "(3rd t20i)",
                   "(1st odi)", "(2nd odi)", "(3rd odi)",
                   "(1st test)", "(2nd test)", "(3rd test)"]:
        t = t.replace(suffix, "")
    t = t.strip()
    # Extract teams from "A vs. B", "A vs B", "A v B"
    for sep in [" vs. ", " vs ", " v "]:
        if sep in t:
            a, b = t.split(sep, 1)
            a, b = a.strip(), b.strip()
            if a in CRICKET_NATIONS and b in CRICKET_NATIONS:
                log.info("DISCOVERY | cricket candidate: %s", title)
                return True
    return False


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
    initial_state: str = "ACTIVE"  # "ACTIVE" or "DEAD" (v2.0)

    @property
    def all_token_ids(self) -> list[str]:
        return [o.token_id for o in self.outcomes]


def classify_market(slug: str, title: str) -> tuple[str, str]:
    """Classify a market into (sport, league) from its slug/title."""
    s = slug.lower()
    t = title.lower()

    if "nba" in s:
        return ("nba", "NBA")
    # NCAA March Madness — treated as 'nba' for model compatibility
    if any(kw in s for kw in ["ncaa", "march-madness", "college-basketball"]):
        return ("nba", "NCAA")
    if any(kw in t for kw in ["march madness", "ncaa", "college basketball",
                               "sweet 16", "elite eight", "final four"]):
        return ("nba", "NCAA")
    if "nfl" in s:
        return ("nfl", "NFL")
    if "nhl" in s:
        return ("nhl", "NHL")
    if "mlb" in s:
        return ("mlb", "MLB")

    # Tennis tours and Grand Slams
    if "atp-" in s or "atp " in t or "atp tour" in t:
        return ("tennis", "ATP")
    if "wta-" in s or "wta " in t or "wta tour" in t:
        return ("tennis", "WTA")
    if "australian-open" in s or "australian open" in t:
        return ("tennis", "Australian Open")
    if "french-open" in s or "roland garros" in t or "french open" in t:
        return ("tennis", "French Open")
    if "wimbledon" in s or "wimbledon" in t:
        return ("tennis", "Wimbledon")
    if "us-open" in s and ("tennis" in s or "tennis" in t):
        return ("tennis", "US Open")
    if "tennis" in s or "tennis" in t:
        return ("tennis", "Tennis")

    # Cricket early guard — must precede football to avoid
    # "Indian Premier League" matching as EPL football
    if "ipl-" in s or "indian premier league" in t:
        return ("cricket", "IPL")

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

    # Polymarket shorthand football codes (v3.8)
    if s.startswith("lal-"):
        return ("football", "La Liga")
    if s.startswith("efa-"):
        return ("football", "EPL")
    if s.startswith("bun-"):
        return ("football", "Bundesliga")
    if s.startswith("fl1-"):
        return ("football", "Ligue 1")
    if s.startswith("sa-"):
        return ("football", "Serie A")
    if s.startswith("ere-"):
        return ("football", "Eredivisie")
    if s.startswith("chm-"):
        return ("football", "Championship")

    # Cricket — Polymarket slug prefix detection (primary path)
    if s.startswith("crint-"):
        return ("cricket", "International")
    if s.startswith("criclcl-"):
        return ("cricket", "Legends League")
    if s.startswith("ipl-"):
        return ("cricket", "IPL")
    # Cricket — keyword detection (fallback)
    if "icc-" in s or "t20-" in s or "odi-" in s:
        return ("cricket", "ICC")
    if "cricket" in s or "cricket" in t:
        return ("cricket", "Cricket")
    # Cricket — franchise league titles
    if any(kw in t for kw in ["legends cricket", "indian premier league",
                               "big bash", "caribbean premier", "psl ",
                               "hundred cricket", "sa20 "]):
        return ("cricket", "Cricket League")
    # Cricket — team-based heuristic (catches "India vs. New Zealand" etc.)
    if _is_cricket_match(title):
        return ("cricket", "International")


    # Football qualifiers / internationals (v4.8)
    if "wcq-" in s or "world-cup-qualif" in s or "world cup qualif" in t:
        return ("football", "World Cup Qualifiers")
    if "ecq-" in s or "euro-qualif" in s or "euro qualif" in t:
        return ("football", "Euro Qualifiers")
    if "afcon" in s or "afcon" in t or "africa-cup" in s or "africa cup" in t:
        return ("football", "AFCON Qualifiers")

    # International football catch-all (v4.8.1)
    _INTL_FOOTBALL_KEYWORDS = [
        "world cup", "qualification", "qualifier", "fifa",
        "uefa", "caf", "afcon", "international", "friendly",
    ]
    if any(kw in t for kw in _INTL_FOOTBALL_KEYWORDS):
        return ("football", "International")

    return ("unknown", "Unknown")

    return ("unknown", "Unknown")


def is_football_friendly(slug: str, title: str) -> bool:
    """Check if a football market is a friendly / exhibition match."""
    s = slug.lower()
    t = title.lower()
    friendly_patterns = [
        "friendly", "friendlies", "exhibition",
        "club-friendly", "international-friendly",
    ]
    for p in friendly_patterns:
        if p in s or p in t:
            return True
    return False


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


def normalize_football_title(title: str) -> str:
    """Strip competition prefix and market suffix for clean team extraction.

    'FIFA World Cup Qualification: Kenya vs Gabon - 1st Half Winner'
    → 'Kenya vs Gabon'
    """
    t = title
    if ":" in t:
        t = t.split(":", 1)[1]
    if " - " in t:
        t = t.split(" - ", 1)[0]
    return t.strip()


def parse_teams_from_title(title: str) -> tuple[str, str]:
    """Extract home/away team names from a match title."""
    # Normalize: strip competition prefix and market suffix
    cleaned = normalize_football_title(title)

    # Common patterns: "Team A vs. Team B", "Team A vs Team B", "Team A v Team B"
    for sep in [" vs. ", " vs ", " v "]:
        if sep in cleaned:
            parts = cleaned.split(sep, 1)
            home, away = parts[0].strip(), parts[1].strip()
            log.info("FOOTBALL_PARSE_SUCCESS | teams=%r", f"{home} vs {away}")
            return (home, away)

    log.info("FOOTBALL_PARSE_FAIL | title=%r", title)
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


async def fetch_cricket_book(
    session: aiohttp.ClientSession,
    token_id: str,
    clob_url: str = "https://clob.polymarket.com",
) -> dict:
    """Fetch REST orderbook for a single token. Returns {bid, ask, spread, mid}."""
    try:
        url = f"{clob_url}/book"
        params = {"token_id": token_id}
        async with session.get(
            url, params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return {"bid": 0, "ask": 0, "spread": 1.0, "mid": 0}
            data = await resp.json()

        bids = data.get("bids", [])
        asks = data.get("asks", [])
        bid = float(bids[0]["price"]) if bids else 0.0
        ask = float(asks[0]["price"]) if asks else 0.0
        spread = ask - bid if ask > bid else 1.0
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        return {"bid": bid, "ask": ask, "spread": spread, "mid": mid}
    except Exception as e:
        log.warning("CRICKET_BOOK_FETCH_FAIL | token=%s | %s", token_id[:12], e)
        return {"bid": 0, "ask": 0, "spread": 1.0, "mid": 0}


def _cricket_match_key(title: str) -> str:
    """Extract a normalized match key for grouping markets by match.

    'Indian Premier League: Mumbai Indians vs Kolkata Knight Riders'
    → 'mumbai indians vs kolkata knight riders'
    """
    t = title.lower()
    # Strip league prefix before last ':'
    if ":" in t:
        t = t.rsplit(":", 1)[-1].strip()
    # Strip prop suffix after ' - '
    if " - " in t:
        t = t.split(" - ", 1)[0].strip()
    return t


async def rank_cricket_markets(
    session: aiohttp.ClientSession,
    cricket_markets: list[SportMarket],
) -> list[SportMarket]:
    """Part 1: Group cricket markets by match, rank, select top 1 per match.

    For each match group:
      1. Fetch REST book for each candidate
      2. Reject dead books (bid ≤ 0.02 AND ask ≥ 0.98, or spread ≥ 0.90)
      3. Rank by: lowest spread → highest liquidity
      4. Select top 1
      5. Log rejections
    """
    if not cricket_markets:
        return []

    # Group by match key
    groups: dict[str, list[SportMarket]] = {}
    for m in cricket_markets:
        key = _cricket_match_key(m.title)
        groups.setdefault(key, []).append(m)

    selected: list[SportMarket] = []

    for match_key, candidates in groups.items():
        log.info("CRICKET_RANK | match=%s | candidates=%d", match_key, len(candidates))

        scored: list[tuple[SportMarket, float, float]] = []  # (market, spread, liquidity)

        for mkt in candidates:
            # Fetch REST book for the first token
            if mkt.outcomes:
                book_data = await fetch_cricket_book(session, mkt.outcomes[0].token_id)
            else:
                book_data = {"bid": 0, "ask": 0, "spread": 1.0, "mid": 0}

            bid = book_data["bid"]
            ask = book_data["ask"]
            rest_spread = book_data["spread"]

            # v2.0: Tag dead books instead of rejecting (STRICT: no spread rejections)
            is_dead_book = (bid <= 0.02 and ask >= 0.98) or rest_spread >= 0.90
            
            scored.append((mkt, rest_spread, mkt.liquidity, is_dead_book))
            log.info(
                "CRICKET_CANDIDATE | %s | spread=%.4f | bid=%.2f "
                "ask=%.2f | liq=$%.0f | is_dead=%s",
                mkt.title[:60], rest_spread, bid, ask, mkt.liquidity, is_dead_book
            )

        if not scored:
            log.warning("CRICKET_RANK_EMPTY | match=%s | no candidates", match_key)
            continue

        # Rank: lowest spread first, then highest liquidity
        scored.sort(key=lambda x: (x[1], -x[2]))

        # Select top 1
        winner, win_spr, win_liq, win_dead = scored[0]
        if win_dead:
            winner.initial_state = "DEAD"
        else:
            winner.initial_state = "ACTIVE"
            
        selected.append(winner)
        log.info(
            "CRICKET_SELECTED | %s | spread=%.4f | liq=$%.0f | initial_state=%s",
            winner.title[:60], win_spr, win_liq, winner.initial_state,
        )

        # Reject others (secondary markets for the same match)
        for mkt, spr, liq, dead in scored[1:]:
            log.info(
                "CRICKET_REJECT_SECONDARY | %s | spread=%.4f | liq=$%.0f",
                mkt.title[:60], spr, liq,
            )

    log.info("CRICKET_RANK_RESULT | selected=%d/%d markets",
             len(selected), len(cricket_markets))
    return selected


async def discover_sports_markets(session: aiohttp.ClientSession) -> list[SportMarket]:
    """Fetch all active single-game sports markets from Polymarket Gamma API."""
    all_markets: list[SportMarket] = []
    cricket_raw: list[SportMarket] = []  # cricket candidates before ranking
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
            if sport not in ("nba", "football", "tennis", "cricket"):
                continue

            # ── Football discovery logging (v4.8.1) ──
            if sport == "football":
                log.info("FOOTBALL_DISCOVERY | title=%r | classified=%r", title, "football")

            # ── Football competition whitelist + friendly block (v4.8) ──
            if sport == "football":
                is_international = league in (
                    "World Cup Qualifiers", "Euro Qualifiers",
                    "AFCON Qualifiers", "Champions League",
                    "Europa League", "FIFA World Cup",
                )

                if is_football_friendly(slug, title):
                    log.info("FOOTBALL_SKIP_FRIENDLY | %s | league=%s", title, league)
                    continue

                if league not in FOOTBALL_ALLOWED_COMPETITIONS:
                    log.info("FOOTBALL_COMP_BLOCKED | %s | league=%s", title, league)
                    continue

                log.info("FOOTBALL_COMP_OK | sport=football | competition=%s | is_international=%s | %s",
                         league, is_international, title)

            # ── Cricket discovery filter (v7.0 — rewritten) ──────────
            if sport == "cricket":
                title_lower = title.lower()
                CRICKET_PROP_KEYWORDS = [
                    "toss", "top batter", "top bowler", "runs",
                    "wickets", "over", "man of the match",
                    "first ball", "sixes", "fours", "boundaries",
                    "highest", "most", "total", "innings",
                ]
                has_vs = " vs " in title_lower or " vs. " in title_lower
                is_prop = any(kw in title_lower for kw in CRICKET_PROP_KEYWORDS)
                if not has_vs or is_prop:
                    log.info("CRICKET_PROP_SKIP | %s | has_vs=%s is_prop=%s",
                             title, has_vs, is_prop)
                    continue

                # ── Binary only (outcomes == 2) ──
                sub_m = event.get("markets", [])
                if sub_m:
                    raw_outcomes = sub_m[0].get("outcomes", "[]")
                    if isinstance(raw_outcomes, str):
                        try:
                            n_outcomes = len(json.loads(raw_outcomes))
                        except Exception:
                            n_outcomes = 0
                    else:
                        n_outcomes = len(raw_outcomes)
                    if n_outcomes != 2:
                        log.info("CRICKET_SKIP_NON_BINARY | %s | outcomes=%d",
                                 title, n_outcomes)
                        continue

                # ── Liquidity gate (≥ 5,000) ──
                # Reduced for v2.0 late-liquidity rescan support
                liq = float(event.get("liquidity", 0) or 0)
                if liq < 5_000:
                    log.info("CRICKET_SKIP_LOW_LIQ | %s | liq=$%.0f", title, liq)
                    continue

                # ── IPL team name match (robust fuzzy) ──
                is_ipl = _is_ipl_match(title)
                is_intl = _is_cricket_match(title)
                if not is_ipl and not is_intl:
                    log.info("CRICKET_SKIP_NO_TEAM_MATCH | %s", title)
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
                end_date=event.get("endDate", "")[:19],  # preserve time for readiness
                outcomes=outcomes,
                volume_24h=float(event.get("volume24hr", 0) or 0),
                liquidity=float(event.get("liquidity", 0) or 0),
                spread=float(primary.get("spread", 1) or 1),
                competitive=float(event.get("competitive", 0) or 0),
                home_team=home_team,
                away_team=away_team,
            )

            # Cricket goes to separate ranking pipeline
            if sport == "cricket":
                cricket_raw.append(sm)
            else:
                all_markets.append(sm)

        log.info("discovery page %d: %d sports markets so far (+ %d cricket candidates)",
                 page + 1, len(all_markets), len(cricket_raw))

        # If very few results, stop early
        if len(events) < 100:
            break

    # ── Cricket: rank and select top 1 per match ─────────────────────
    if cricket_raw:
        selected_cricket = await rank_cricket_markets(session, cricket_raw)
        all_markets.extend(selected_cricket)
        log.info("cricket discovery: %d candidates → %d selected",
                 len(cricket_raw), len(selected_cricket))

    log.info("discovered %d single-game sports markets", len(all_markets))
    return all_markets
