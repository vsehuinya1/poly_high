#!/usr/bin/env python3
"""
Sports Market Orchestrator — main entry point.

Connects game feeds (API-Football, NBA, Tennis) with Polymarket WebSocket,
runs fair value models in real-time, detects edges, paper trades.

Usage:
    python -m sports.main                    # auto-discover and run
    python -m sports.main --discover-only    # just show what's available
    python -m sports.main --date 2026-02-22  # specify date
"""
import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from difflib import SequenceMatcher

import aiohttp

from sports.config import (
    DATA_DIR, LOG_DIR, SCORE_POLL_INTERVAL_S, POLYMARKET_SNAPSHOT_S,
    DISCOVERY_INTERVAL_S,
)
from sports.discovery import discover_sports_markets, SportMarket
from sports.feeds import FootballFeed, NBAFeed, PolymarketFeed, GameState
from sports.engine import SignalEngine, GameMarketLink
from sports.models import invert_1x2_to_lambdas

# Tennis engine imports
from tennis.state import TennisState, PointScore, update_from_point, compute_momentum_delta
from tennis.model import get_win_prob as tennis_get_win_prob
from tennis.strategy import InflectionStrategy, TennisSignal
from tennis.execution import TennisExecutionGuard
from tennis.logger import TennisCSVLogger
from tennis.livefeed import TennisScoreFeed, FlashscoreMatch
from tennis.matching import (
    extract_players_from_title, identify_favorite_from_outcomes,
    normalize_tennis_name, tennis_name_match_score,
)
from sports.config import (
    TENNIS_PANIC_EDGE, TENNIS_REVERSION_EDGE,
    TENNIS_PRICE_CAP, TENNIS_STALENESS_S, TENNIS_COOLDOWN_S,
    TENNIS_FEED_POLL_S,
)

log = logging.getLogger("sports.main")


def setup_logging():
    """Configure logging to console + rotating file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = time.strftime("%Y%m%d")

    fmt = "%(asctime)s [%(name)-18s] %(levelname)-5s  %(message)s"
    datefmt = "%H:%M:%S"

    from logging.handlers import RotatingFileHandler

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(
                LOG_DIR / f"sports_{today}.log",
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
            ),
        ],
    )
    # Quiet some noisy loggers
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


import unicodedata

# Common name variations mapped to canonical versions
TEAM_ALIASES = {
    "red star belgrade": "crvena zvezda",
    "fk crvena zvezda": "crvena zvezda",
    "crvena zvezda": "crvena zvezda",
    "lille osc": "lille",
    "lille": "lille",
    "celta de vigo": "celta vigo",
    "paok salonika": "paok",
    "ferencvarosi": "ferencvaros",
}


def normalize_name(name: str) -> str:
    """Remove accents, prefixes, suffixes and normalize team names."""
    # Convert to lowercase and strip
    name = name.lower().strip()
    
    # Remove accents
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )

    # Check mapping before prefix/suffix stripping
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]

    # Remove common prefixes
    for prefix in ["fk ", "as ", "sc ", "afc ", "rc ", "bc ", "ac "]:
        if name.startswith(prefix):
            name = name[len(prefix):].strip()
    
    # Remove common suffixes
    for suffix in [" fc", " sc", " bk", " cf", " s.k.", " sk", " tc", " pfc", " osc", " ao", " 1945", " tc"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Check mapping again after stripping
    return TEAM_ALIASES.get(name, name)


def fuzzy_match_score(a: str, b: str) -> float:
    """Fuzzy match score between two team names (0-1)."""
    a = normalize_name(a)
    b = normalize_name(b)
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.90
    return SequenceMatcher(None, a, b).ratio()


def match_game_to_market(
    game: GameState,
    markets: list[SportMarket],
    threshold: float = 0.85,
) -> SportMarket | None:
    """Find the best matching Polymarket market for a live game."""
    best_match = None
    best_score = 0.0

    for m in markets:
        if m.sport != game.sport:
            continue

        # Score based on team name similarity - BOTH teams must match
        # Try Game(H) vs Market(H) and Game(A) vs Market(A)
        score1 = (fuzzy_match_score(game.home_team, m.home_team) + 
                  fuzzy_match_score(game.away_team, m.away_team)) / 2.0
        
        # Try Game(H) vs Market(A) and Game(A) vs Market(H)
        score2 = (fuzzy_match_score(game.home_team, m.away_team) + 
                  fuzzy_match_score(game.away_team, m.home_team)) / 2.0
        
        score = max(score1, score2)

        if score > best_score:
            best_score = score
            best_match = m

    if best_score >= threshold and best_match:
        log.info("matched game %s vs %s → %s (score=%.2f)",
                 game.home_team, game.away_team, best_match.title, best_score)
        return best_match

    return None


def build_game_market_link(
    game: GameState,
    market: SportMarket,
) -> GameMarketLink:
    """Create a GameMarketLink from a matched game and market."""
    # Identify home/away token IDs from outcomes
    home_tid = ""
    away_tid = ""
    draw_tid = ""
    all_tids = []

    for outcome in market.outcomes:
        tid = outcome.token_id
        label = outcome.outcome_label.lower()
        all_tids.append(tid)

        # Match outcome labels to home/away/draw
        home_name = game.home_team.lower()
        away_name = game.away_team.lower()

        if any(part in label for part in home_name.split()[-1:]):
            if not home_tid:
                home_tid = tid
        elif any(part in label for part in away_name.split()[-1:]):
            if not away_tid:
                away_tid = tid
        elif "draw" in label or "tie" in label:
            draw_tid = tid
        elif label == "yes":
            # For binary markets, "Yes" is usually the first team
            if not home_tid:
                home_tid = tid
        elif label == "no":
            if not away_tid:
                away_tid = tid

    # Fallback: use first two outcomes as home/away
    if not home_tid and len(market.outcomes) >= 1:
        home_tid = market.outcomes[0].token_id
    if not away_tid and len(market.outcomes) >= 2:
        away_tid = market.outcomes[1].token_id

    # Get pre-game prob from home token's last price in discovery.
    # Only use if game hasn't started yet (is_live=False).
    # If discovered mid-game (e.g. system restart), the market price
    # is already in-play and would poison the anchor → fallback to 0.5.
    pregame_home = 0.5
    pregame_draw = 0.0
    pregame_away = 0.0

    if not game.is_live:
        # Collect prices for all matched outcomes
        for o in market.outcomes:
            if o.token_id == home_tid and 0.05 < o.last_price < 0.95:
                pregame_home = o.last_price
            elif o.token_id == away_tid and 0.05 < o.last_price < 0.95:
                pregame_away = o.last_price
            elif o.token_id == draw_tid and 0.05 < o.last_price < 0.95:
                pregame_draw = o.last_price

        # If draw not explicitly priced, infer it
        if pregame_draw <= 0 and pregame_home > 0 and pregame_away > 0:
            pregame_draw = max(0.0, 1.0 - pregame_home - pregame_away)

        log.info("pre-game anchor for %s: H=%.3f D=%.3f A=%.3f",
                 game.home_team, pregame_home, pregame_draw, pregame_away)
    else:
        log.info("game %s already live — using neutral anchor 0.5", game.home_team)

    return GameMarketLink(
        game_id=game.game_id,
        sport=game.sport,
        league=game.league,
        home_team=game.home_team,
        away_team=game.away_team,
        polymarket_event_id=market.event_id,
        polymarket_title=market.title,
        polymarket_slug=market.slug,
        home_token_id=home_tid,
        away_token_id=away_tid,
        draw_token_id=draw_tid,
        all_token_ids=all_tids,
        pregame_home_prob=pregame_home,
        pregame_draw_prob=pregame_draw,
        pregame_away_prob=pregame_away,
    )


def prewarm_football_lambdas(links: dict[str, "GameMarketLink"]) -> None:
    """Pre-warm λ inversion for all football game-market links.

    MUST be called before starting live polling. Runs grid-search
    inversion (~10s per unique odds triple) and stores results in
    both the global cache and each link object.

    Fails fast if any inversion produces unacceptable SSE.
    """
    football_links = [
        (gid, link) for gid, link in links.items()
        if link.sport == "football"
    ]

    if not football_links:
        log.info("PREWARM | no football links to pre-warm")
        return

    log.info("PREWARM | pre-warming λ for %d football games...", len(football_links))
    warmed = 0
    sse_warn_threshold = 0.01

    for game_id, link in football_links:
        p_h = link.pregame_home_prob
        p_d = link.pregame_draw_prob
        p_a = link.pregame_away_prob

        # Skip if no valid pre-match probs
        if p_h <= 0 and p_d <= 0 and p_a <= 0:
            log.warning(
                "PREWARM | %s — no pre-match probs, using fallback",
                link.polymarket_title,
            )
            # Still invert with fallback so λ is never None
            p_h, p_d, p_a = 0.45, 0.28, 0.27  # neutral default

        lam_h, lam_a, sse = invert_1x2_to_lambdas(p_h, p_d, p_a)
        link.lambda_home = lam_h
        link.lambda_away = lam_a
        warmed += 1

        if sse > sse_warn_threshold:
            log.warning(
                "PREWARM | %s | λh=%.2f λa=%.2f | SSE=%.6f > %.4f WARN",
                link.polymarket_title, lam_h, lam_a, sse, sse_warn_threshold,
            )
        else:
            log.info(
                "INVERSION OK | %s | λh=%.2f λa=%.2f | SSE=%.6f",
                link.polymarket_title, lam_h, lam_a, sse,
            )

    log.info("PREWARM | complete — %d/%d football games warmed",
             warmed, len(football_links))


class SportsOrchestrator:
    """Main async orchestrator for the sports trading system."""

    def __init__(self, target_date: str):
        self.target_date = target_date
        self.football_feed = FootballFeed()
        self.nba_feed = NBAFeed()
        self.poly_feed = PolymarketFeed()
        self.engine = SignalEngine(DATA_DIR)
        self.markets: list[SportMarket] = []
        self.links: dict[str, GameMarketLink] = {}  # game_id → link
        self._shutdown = False
        self._session: aiohttp.ClientSession | None = None

        # ── Tennis Engine ─────────────────────────────────────────
        self.tennis_strategy = InflectionStrategy(
            panic_edge_threshold=TENNIS_PANIC_EDGE,
            reversion_edge_threshold=TENNIS_REVERSION_EDGE,
        )
        self.tennis_guard = TennisExecutionGuard(
            price_cap=TENNIS_PRICE_CAP,
            staleness_s=TENNIS_STALENESS_S,
            cooldown_s=TENNIS_COOLDOWN_S,
        )
        self.tennis_logger = TennisCSVLogger(DATA_DIR)
        self.tennis_score_feed = TennisScoreFeed(poll_interval_s=TENNIS_FEED_POLL_S)
        self.tennis_markets: list[SportMarket] = []  # discovered tennis markets
        self.tennis_links: dict[str, GameMarketLink] = {}  # match_id → link
        self.tennis_states: dict[str, TennisState] = {}  # match_id → latest state
        self._tennis_fs_map: dict[str, str] = {}  # poly_event_id → flashscore_match_id

    async def discover(self) -> list[SportMarket]:
        """Discover active sports markets on Polymarket."""
        log.info("discovering sports markets on Polymarket...")
        async with aiohttp.ClientSession() as session:
            markets = await discover_sports_markets(session)

        self.markets = markets

        # Summary
        by_sport = {}
        for m in markets:
            key = f"{m.sport}/{m.league}"
            by_sport.setdefault(key, []).append(m)

        log.info("=" * 60)
        log.info("DISCOVERED MARKETS")
        log.info("=" * 60)
        for key, items in sorted(by_sport.items()):
            log.info("  %s: %d games", key, len(items))
            for m in items[:5]:
                log.info("    • %s (vol24h=$%.0f, liq=$%.0f)",
                         m.title, m.volume_24h, m.liquidity)
        log.info("  Total: %d single-game markets", len(markets))

        return markets

    async def fetch_fixtures(self):
        """Fetch all game fixtures for today and tomorrow."""
        # We fetch two days to catch late-night NBA games starting after 00:00 UTC
        dates = [
            self.target_date,
            (datetime.strptime(self.target_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        ]

        async with aiohttp.ClientSession() as session:
            for d in dates:
                log.info("fetching football fixtures for %s...", d)
                await self.football_feed.fetch_todays_fixtures(session, d)
                
                # Note: NBA feed fetch_live_scores fetches current/upcoming regardless of date
                # but we call it here to ensure we have data for matching.
                log.info("fetching NBA scoreboard...")
                await self.nba_feed.fetch_live_scores(session)


    async def build_links(self):
        """Match games to Polymarket markets and build monitoring links."""
        log.info("matching games to Polymarket markets...")

        # Match football fixtures
        for game_id, game in self.football_feed.games.items():
            match = match_game_to_market(game, self.markets)
            if match:
                link = build_game_market_link(game, match)
                self.links[game_id] = link
                self.engine.register_link(link)

        # Match NBA games
        async with aiohttp.ClientSession() as session:
            await self.nba_feed.fetch_live_scores(session)
        for game_id, game in self.nba_feed.games.items():
            match = match_game_to_market(game, self.markets)
            if match:
                link = build_game_market_link(game, match)
                self.links[game_id] = link
                self.engine.register_link(link)

        # ── Tennis market links ───────────────────────────────────
        self.tennis_markets = [m for m in self.markets if m.sport == "tennis"]
        for tm in self.tennis_markets:
            if tm.event_id in self.tennis_links:
                continue  # already linked

            # Extract player names from Polymarket title
            player_a_name, player_b_name = extract_players_from_title(tm.title)
            if not player_a_name or not player_b_name:
                log.warning("TENNIS SKIP: could not extract players from '%s'", tm.title)
                continue

            # Match outcomes to players using name matching
            all_tids = [o.token_id for o in tm.outcomes]
            if len(tm.outcomes) >= 2:
                a_tid, b_tid, price_a, price_b = identify_favorite_from_outcomes(
                    tm.outcomes, player_a_name, player_b_name
                )
            elif len(tm.outcomes) == 1:
                a_tid = tm.outcomes[0].token_id
                b_tid = ""
                price_a = tm.outcomes[0].last_price
                price_b = 1.0 - price_a
            else:
                continue

            link = GameMarketLink(
                game_id=tm.event_id,
                sport="tennis",
                league=tm.league,
                home_team=player_a_name,
                away_team=player_b_name,
                polymarket_event_id=tm.event_id,
                polymarket_title=tm.title,
                polymarket_slug=tm.slug,
                home_token_id=a_tid,
                away_token_id=b_tid,
                all_token_ids=all_tids,
                pregame_home_prob=price_a,
                pregame_away_prob=price_b,
            )
            self.tennis_links[tm.event_id] = link

            # Determine who is the pre-game favorite
            fav_id = player_a_name if price_a >= price_b else player_b_name

            # Initialize TennisState with real player names
            self.tennis_states[tm.event_id] = TennisState(
                match_id=tm.event_id,
                player_a_id=player_a_name,
                player_b_id=player_b_name,
                server_id=player_a_name,  # default, updated by feed
                receiver_id=player_b_name,
                pregame_favorite_id=fav_id,
                timestamp=time.time(),
            )

            log.info("TENNIS LINK: %s | %s vs %s | pre=%.3f/%.3f | fav=%s | tokens=%d",
                     tm.title, player_a_name, player_b_name,
                     price_a, price_b, fav_id, len(all_tids))

        log.info("matched %d games + %d tennis matches to Polymarket markets",
                 len(self.links), len(self.tennis_links))

        # Collect token IDs for WS subscription — ONLY matched game tokens.
        # Do NOT subscribe to all discovered market tokens: non-moneyline
        # sub-market tokens (spreads, totals, props) cause WS INVALID OPERATION.
        all_tokens = []
        for link in self.links.values():
            all_tokens.extend(link.all_token_ids)
        # Tennis tokens too
        for link in self.tennis_links.values():
            all_tokens.extend(link.all_token_ids)

        log.info("subscribing to %d token IDs for %d matched games on Polymarket WS",
                 len(all_tokens), len(self.links) + len(self.tennis_links))
        self.poly_feed.set_tokens(all_tokens)

    async def _score_polling_loop(self):
        """Poll live scores for football and NBA."""
        while not self._shutdown:
            try:
                async with aiohttp.ClientSession() as session:
                    while not self._shutdown:
                        # Football
                        try:
                            await self.football_feed.fetch_live_scores(session)
                        except Exception as e:
                            log.error("football feed error: %s", e)

                        # NBA
                        try:
                            await self.nba_feed.fetch_live_scores(session)
                        except Exception as e:
                            log.error("nba feed error: %s", e)

                        await asyncio.sleep(SCORE_POLL_INTERVAL_S)
            except Exception as e:
                log.error("score polling loop error: %s", e)
                await asyncio.sleep(5)

    async def _signal_processing_loop(self):
        """Main signal processing — runs model + edge detection on every tick."""
        while not self._shutdown:
            try:
                live_count = 0
                signal_count = 0

                for game_id, link in list(self.links.items()):
                    game_state = (
                        self.football_feed.games.get(game_id) or
                        self.nba_feed.games.get(game_id)
                    )

                    if not game_state:
                        continue

                    # process_tick handles both live and finished games
                    signals = await self.engine.process_tick(
                        game_state,
                        self.poly_feed.books,
                        link,
                    )
                    if game_state.is_live:
                        live_count += 1
                    signal_count += len(signals)

                if live_count > 0:
                    log.debug("tick: %d live games, %d signals, ws_msgs=%d",
                              live_count, signal_count,
                              self.poly_feed.message_count)

            except Exception as e:
                log.error("signal processing error: %s", e)

            await asyncio.sleep(POLYMARKET_SNAPSHOT_S)

    async def _tennis_score_polling_loop(self):
        """Poll Flashscore for live tennis scores and update TennisState."""
        # Wait for initial discovery + link building
        await asyncio.sleep(5)
        await self.tennis_score_feed.start()
        log.info("Tennis score polling started")

        while not self._shutdown:
            try:
                count = await self.tennis_score_feed.poll_once()

                # For each Polymarket tennis link, find matching Flashscore match
                for poly_id, link in list(self.tennis_links.items()):
                    # Try cached mapping first
                    fs_id = self._tennis_fs_map.get(poly_id)
                    fs_match = None

                    if fs_id:
                        fs_match = self.tennis_score_feed._matches.get(fs_id)
                    
                    if not fs_match:
                        # Fuzzy search by player names
                        fs_match = self.tennis_score_feed.find_match_by_players(
                            link.home_team, link.away_team
                        )
                        if fs_match:
                            self._tennis_fs_map[poly_id] = fs_match.match_id
                            log.info("TENNIS MAP: %s → FS:%s (%s vs %s)",
                                     link.polymarket_title[:40], fs_match.match_id,
                                     fs_match.player_a, fs_match.player_b)

                    if not fs_match or not fs_match.is_live:
                        continue

                    # Update TennisState from Flashscore data
                    old_state = self.tennis_states.get(poly_id)
                    if not old_state:
                        continue

                    # Determine which Flashscore player maps to which Poly player
                    from tennis.matching import tennis_name_match_score
                    score_direct = tennis_name_match_score(link.home_team, fs_match.player_a)
                    score_reversed = tennis_name_match_score(link.home_team, fs_match.player_b)
                    
                    if score_direct >= score_reversed:
                        # Poly A = FS home, Poly B = FS away
                        sets_a, sets_b = fs_match.sets_a, fs_match.sets_b
                        games_a, games_b = fs_match.games_a, fs_match.games_b
                        point_a_raw, point_b_raw = fs_match.point_a, fs_match.point_b
                        server_id = link.home_team if fs_match.serving == "a" else link.away_team
                    else:
                        # Poly A = FS away, Poly B = FS home
                        sets_a, sets_b = fs_match.sets_b, fs_match.sets_a
                        games_a, games_b = fs_match.games_b, fs_match.games_a
                        point_a_raw, point_b_raw = fs_match.point_b, fs_match.point_a
                        server_id = link.home_team if fs_match.serving == "b" else link.away_team

                    # Map point strings to PointScore enum values
                    point_map = {"0": PointScore.ZERO, "15": PointScore.FIFTEEN,
                                 "30": PointScore.THIRTY, "40": PointScore.FORTY,
                                 "A": PointScore.AD, "AD": PointScore.AD,
                                 "50": PointScore.AD}
                    try:
                        pt_a = point_map.get(str(point_a_raw), PointScore.ZERO)
                        pt_b = point_map.get(str(point_b_raw), PointScore.ZERO)
                    except Exception:
                        pt_a, pt_b = PointScore.ZERO, PointScore.ZERO

                    # Detect tiebreak (both at 6 games)
                    is_tiebreak = (games_a == 6 and games_b == 6)

                    receiver_id = link.away_team if server_id == link.home_team else link.home_team

                    new_state = TennisState(
                        match_id=poly_id,
                        sets_a=sets_a,
                        sets_b=sets_b,
                        games_a=games_a,
                        games_b=games_b,
                        point_a=pt_a,
                        point_b=pt_b,
                        is_tiebreak=is_tiebreak,
                        player_a_id=old_state.player_a_id,
                        player_b_id=old_state.player_b_id,
                        server_id=server_id,
                        receiver_id=receiver_id,
                        pregame_favorite_id=old_state.pregame_favorite_id,
                        timestamp=time.time(),
                    )

                    # Only log state changes
                    if (new_state.sets_a != old_state.sets_a or
                        new_state.sets_b != old_state.sets_b or
                        new_state.games_a != old_state.games_a or
                        new_state.games_b != old_state.games_b or
                        new_state.point_a != old_state.point_a or
                        new_state.point_b != old_state.point_b):
                        log.info("TENNIS SCORE: %s | %s [%d-%d] %d-%d (%s-%s) srv=%s",
                                 link.home_team[:15], link.away_team[:15],
                                 sets_a, sets_b, games_a, games_b,
                                 pt_a.value, pt_b.value, server_id[:10])

                    self.tennis_states[poly_id] = new_state

            except Exception as e:
                log.error("tennis score poll error: %s", e)

            await asyncio.sleep(TENNIS_FEED_POLL_S)

    async def _tennis_signal_loop(self):
        """Tennis Strategy B signal processing — runs on every tick."""
        while not self._shutdown:
            try:
                for match_id, link in list(self.tennis_links.items()):
                    state = self.tennis_states.get(match_id)
                    if not state:
                        continue

                    # Only process live matches (state updated by score feed)
                    if state.sets_a == 0 and state.sets_b == 0 and state.games_a == 0 and state.games_b == 0:
                        # State never updated — match not live yet
                        continue

                    # Get current market price for the favorite
                    fav_token = link.home_token_id if state.pregame_favorite_id == link.home_team else link.away_token_id
                    fav_book = self.poly_feed.books.get(fav_token)
                    if not fav_book or fav_book.mid <= 0:
                        continue

                    market_price = fav_book.mid

                    # Run Strategy B evaluation
                    signal = self.tennis_strategy.evaluate(state, market_price)
                    if signal is None:
                        continue

                    # Log signal
                    self.tennis_logger.log_signal(signal)
                    log.info("TENNIS SIGNAL | %s | edge=%+.4f | fair=%.4f | mkt=%.4f",
                             signal.trigger_type, signal.edge, signal.fair_price, market_price)

                    # Check execution guards
                    decision = self.tennis_guard.can_execute(signal, state)
                    if not decision.can_execute:
                        log.info("TENNIS BLOCKED | %s | %s", decision.reason, match_id)
                        continue

                    # Paper trade: log entry
                    self.tennis_guard.record_entry(match_id)
                    self.tennis_logger.log_trade_entry(signal, market_price_at_bp=market_price)
                    log.info("TENNIS PAPER ENTRY | %s | edge=%.4f | mkt=%.4f | %s %d-%d %d-%d",
                             signal.trigger_type, signal.edge, market_price,
                             link.polymarket_title[:30],
                             state.sets_a, state.sets_b, state.games_a, state.games_b)

                    # Telegram alert
                    try:
                        await self.engine.tg.send(
                            f"🎾 <b>Tennis Signal</b>\n"
                            f"Trigger: {signal.trigger_type}\n"
                            f"Edge: {signal.edge:+.4f}\n"
                            f"Fair: {signal.fair_price:.4f} | Mkt: {market_price:.4f}\n"
                            f"Match: {link.polymarket_title}\n"
                            f"Score: {state.sets_a}-{state.sets_b} | {state.games_a}-{state.games_b} | {state.point_a.value}-{state.point_b.value}"
                        )
                    except Exception:
                        pass

            except Exception as e:
                log.error("tennis signal loop error: %s", e)

            await asyncio.sleep(POLYMARKET_SNAPSHOT_S)

    async def _status_printer_loop(self):
        """Periodically print system status + Telegram updates."""
        tg_interval = 0  # send Telegram every 5th iteration (5 min)
        while not self._shutdown:
            try:
                live_football = sum(
                    1 for g in self.football_feed.games.values() if g.is_live
                )
                live_nba = sum(
                    1 for g in self.nba_feed.games.values() if g.is_live
                )
                live_tennis = len(self.tennis_links)

                summary = self.engine.get_summary()

                log.info(
                    "STATUS | Football: %d live | NBA: %d live | Tennis: %d mkts | "
                    "WS: %s (msgs=%d) | Links: %d | "
                    "Trades: %d (wins=%d, PnL=$%.2f)",
                    live_football, live_nba, live_tennis,
                    "OK" if self.poly_feed.is_connected else "DOWN",
                    self.poly_feed.message_count,
                    len(self.links) + len(self.tennis_links),
                    summary.get("total_trades", 0),
                    summary.get("wins", 0),
                    summary.get("daily_pnl", 0.0),
                )

                for game_id, link in self.links.items():
                    game = (
                        self.football_feed.games.get(game_id) or
                        self.nba_feed.games.get(game_id)
                    )
                    if game and game.is_live:
                        home_book = self.poly_feed.books.get(link.home_token_id)
                        away_book = self.poly_feed.books.get(link.away_token_id)
                        h_mid = f"{home_book.mid:.3f}" if home_book and home_book.mid > 0 else "---"
                        a_mid = f"{away_book.mid:.3f}" if away_book and away_book.mid > 0 else "---"
                        log.info(
                            "  LIVE: %s %d-%d %s (%s %s') | poly: H=%s A=%s",
                            game.home_team, game.home_score, game.away_score,
                            game.away_team, game.period, f"{game.elapsed_minutes:.0f}",
                            h_mid, a_mid,
                        )

                # Band stats every 5 minutes
                tg_interval += 1
                if tg_interval >= 5:
                    # Log per-game band rejects
                    for gid, gts in self.engine._game_states.items():
                        if gts.band_rejects > 0:
                            log.info(
                                "BAND_STATS | %s | rejects=%d | gs=%s | gpnl=$%.0f",
                                gid, gts.band_rejects,
                                gts.status.value, gts.pnl,
                            )

                    if live_football > 0 or live_nba > 0:
                        tg_interval = 0
                        await self.engine.tg.notify_status(
                            live_football, live_nba,
                            self.poly_feed.is_connected,
                            self.poly_feed.message_count,
                            len(self.links),
                            summary.get("total_trades", 0),
                            summary.get("daily_pnl", 0.0),
                        )

            except Exception as e:
                log.error("status printer error: %s", e)

            await asyncio.sleep(60)

    async def _rematching_loop(self):
        """Periodically re-discover markets and re-match new games."""
        while not self._shutdown:
            await asyncio.sleep(DISCOVERY_INTERVAL_S)
            try:
                log.info("re-scanning for new markets...")
                await self.discover()
                await self.build_links()
                prewarm_football_lambdas(self.links)
            except Exception as e:
                log.error("re-discovery error: %s", e)

    async def run(self):
        """Main entry point — start all loops."""
        log.info("=" * 60)
        log.info("  SPORTS MARKET SYSTEM STARTING")
        log.info("  Date: %s", self.target_date)
        log.info("  Football source: ESPN (no key required)")
        log.info("  Data dir: %s", DATA_DIR.absolute())
        log.info("=" * 60)

        # Phase 1: Discover markets
        await self.discover()

        # Phase 2: Fetch fixture schedule
        await self.fetch_fixtures()

        # Phase 3: Build game-market links
        await self.build_links()

        # Phase 3.5: Pre-warm football λ values (grid search — must
        # complete before any live polling or signal processing starts)
        prewarm_football_lambdas(self.links)

        if not self.links:
            log.warning("no game-market links found — will continue "
                       "monitoring for new games")

        # Telegram startup notification
        all_tokens = []
        for link in self.links.values():
            all_tokens.extend(link.all_token_ids)
        await self.engine.tg.notify_startup(
            len(self.markets), len(self.links), len(all_tokens)
        )

        # Phase 4: Start all async loops
        tasks = [
            asyncio.create_task(self.poly_feed.run(), name="polymarket_ws"),
            asyncio.create_task(self.poly_feed.run_book_polling(), name="book_rest_polling"),
            asyncio.create_task(self._score_polling_loop(), name="score_polling"),
            asyncio.create_task(self._signal_processing_loop(), name="signal_processing"),
            asyncio.create_task(self._status_printer_loop(), name="status_printer"),
            asyncio.create_task(self._rematching_loop(), name="rematching"),
            # Tennis — score polling + signal processing
            asyncio.create_task(self._tennis_score_polling_loop(), name="tennis_scores"),
            asyncio.create_task(self._tennis_signal_loop(), name="tennis_signals"),
        ]

        # Graceful shutdown handler
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        log.info("all systems running — monitoring for live games")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.info("tasks cancelled")

    async def shutdown(self):
        """Graceful shutdown."""
        log.info("shutting down...")
        self._shutdown = True
        await self.poly_feed.shutdown()
        await self.tennis_score_feed.shutdown()
        self.tennis_logger.close()

        # Print final summary
        summary = self.engine.get_summary()
        log.info("=" * 60)
        log.info("  SESSION SUMMARY")
        log.info("=" * 60)
        for k, v in summary.items():
            if isinstance(v, float):
                log.info("  %s: %.4f", k, v)
            else:
                log.info("  %s: %s", k, v)
        log.info("  Tennis markets monitored: %d", len(self.tennis_links))
        log.info("=" * 60)

        await self.engine.close()


async def run_discover_only(target_date: str):
    """Just discover and print markets, don't start feeds."""
    setup_logging()
    orch = SportsOrchestrator(target_date)
    await orch.discover()
    await orch.fetch_fixtures()

    # Print matching summary
    for game_id, game in orch.football_feed.games.items():
        match = match_game_to_market(game, orch.markets)
        status = "✓ MATCHED" if match else "✗ no match"
        poly_title = match.title if match else ""
        log.info("  [%s] %s vs %s → %s %s",
                 game.league, game.home_team, game.away_team,
                 status, poly_title)

    async with aiohttp.ClientSession() as session:
        await orch.nba_feed.fetch_live_scores(session)
    for game_id, game in orch.nba_feed.games.items():
        match = match_game_to_market(game, orch.markets)
        status = "✓ MATCHED" if match else "✗ no match"
        poly_title = match.title if match else ""
        log.info("  [NBA] %s vs %s → %s %s",
                 game.home_team, game.away_team,
                 status, poly_title)


def main():
    parser = argparse.ArgumentParser(description="Polymarket Sports Monitor")
    parser.add_argument("--discover-only", action="store_true",
                       help="Just discover markets, don't start feeds")
    parser.add_argument("--date", type=str, default=None,
                       help="Target date (YYYY-MM-DD), default: today")
    args = parser.parse_args()

    # Default to today (UTC)
    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    setup_logging()

    if args.discover_only:
        asyncio.run(run_discover_only(target_date))
    else:
        orch = SportsOrchestrator(target_date)
        asyncio.run(orch.run())


if __name__ == "__main__":
    main()
