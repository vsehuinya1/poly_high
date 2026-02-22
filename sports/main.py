#!/usr/bin/env python3
"""
Sports Market Orchestrator — main entry point.

Connects game feeds (API-Football, NBA) with Polymarket WebSocket,
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
    DISCOVERY_INTERVAL_S, API_FOOTBALL_KEY,
)
from sports.discovery import discover_sports_markets, SportMarket
from sports.feeds import FootballFeed, NBAFeed, PolymarketFeed, GameState
from sports.engine import SignalEngine, GameMarketLink

log = logging.getLogger("sports.main")


def setup_logging():
    """Configure logging to console + file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = time.strftime("%Y%m%d")

    fmt = "%(asctime)s [%(name)-18s] %(levelname)-5s  %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / f"sports_{today}.log"),
        ],
    )
    # Quiet some noisy loggers
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def fuzzy_match_score(a: str, b: str) -> float:
    """Fuzzy match score between two team names (0-1)."""
    a = a.lower().strip()
    b = b.lower().strip()
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.85
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

    # Get pre-game prob from home token's last price in discovery
    pregame_prob = 0.5
    for o in market.outcomes:
        if o.token_id == home_tid:
            pregame_prob = o.last_price
            break

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
        pregame_home_prob=pregame_prob,
    )


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
        """Fetch today's football fixtures and pre-load schedule."""
        if not API_FOOTBALL_KEY:
            log.warning("API_FOOTBALL_KEY not set — football feed disabled")
            return

        log.info("fetching football fixtures for %s...", self.target_date)
        async with aiohttp.ClientSession() as session:
            fixtures = await self.football_feed.fetch_todays_fixtures(
                session, self.target_date
            )
        log.info("found %d football fixtures across all leagues", len(fixtures))

        # Pre-create game states for scheduled fixtures
        for fix in fixtures:
            fixture_info = fix.get("fixture", {})
            fixture_id = str(fixture_info.get("id", ""))
            teams = fix.get("teams", {})
            league_name = fix.get("_league_name", "Unknown")
            status = fixture_info.get("status", {}).get("short", "NS")

            gs = GameState(
                game_id=fixture_id,
                sport="football",
                league=league_name,
                status=status,
                home_team=teams.get("home", {}).get("name", ""),
                away_team=teams.get("away", {}).get("name", ""),
                total_minutes=90.0,
                timestamp=time.time(),
            )
            self.football_feed.games[fixture_id] = gs

            log.info("  [%s] %s vs %s (%s) — fixture #%s",
                     league_name, gs.home_team, gs.away_team,
                     status, fixture_id)

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

        log.info("matched %d games to Polymarket markets", len(self.links))

        # Collect all token IDs for WS subscription
        all_tokens = []
        for link in self.links.values():
            all_tokens.extend(link.all_token_ids)

        # Also subscribe to all discovered market tokens (for data collection)
        for m in self.markets:
            for o in m.outcomes:
                if o.token_id not in all_tokens:
                    all_tokens.append(o.token_id)

        log.info("subscribing to %d token IDs on Polymarket WS", len(all_tokens))
        self.poly_feed.set_tokens(all_tokens)

    async def _score_polling_loop(self):
        """Poll live scores for football and NBA."""
        while not self._shutdown:
            try:
                async with aiohttp.ClientSession() as session:
                    while not self._shutdown:
                        # Football
                        if API_FOOTBALL_KEY:
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

                summary = self.engine.get_summary()

                log.info(
                    "STATUS | Football: %d live | NBA: %d live | "
                    "WS: %s (msgs=%d) | Links: %d | "
                    "Trades: %d (wins=%d, PnL=$%.2f)",
                    live_football, live_nba,
                    "OK" if self.poly_feed.is_connected else "DOWN",
                    self.poly_feed.message_count,
                    len(self.links),
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

                # Telegram status every 5 minutes
                tg_interval += 1
                if tg_interval >= 5 and (live_football > 0 or live_nba > 0):
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
            except Exception as e:
                log.error("re-discovery error: %s", e)

    async def run(self):
        """Main entry point — start all loops."""
        log.info("=" * 60)
        log.info("  SPORTS MARKET SYSTEM STARTING")
        log.info("  Date: %s", self.target_date)
        log.info("  API-Football: %s", "CONFIGURED" if API_FOOTBALL_KEY else "MISSING")
        log.info("  Data dir: %s", DATA_DIR.absolute())
        log.info("=" * 60)

        # Phase 1: Discover markets
        await self.discover()

        # Phase 2: Fetch fixture schedule
        await self.fetch_fixtures()

        # Phase 3: Build game-market links
        await self.build_links()

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
            asyncio.create_task(self._score_polling_loop(), name="score_polling"),
            asyncio.create_task(self._signal_processing_loop(), name="signal_processing"),
            asyncio.create_task(self._status_printer_loop(), name="status_printer"),
            asyncio.create_task(self._rematching_loop(), name="rematching"),
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
