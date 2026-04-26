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

from config import POLYMARKET_API_KEY, POLYMARKET_SECRET, POLYMARKET_PASSPHRASE
from sports.config import (
    DATA_DIR, LOG_DIR, SCORE_POLL_INTERVAL_S, POLYMARKET_SNAPSHOT_S,
    DISCOVERY_INTERVAL_S,
)
from sports.discovery import discover_sports_markets, SportMarket
from sports.feeds import FootballFeed, NBAFeed, NCAAFeed, PolymarketFeed, GameState
from sports.engine import SignalEngine, GameMarketLink
from sports.models import invert_1x2_to_lambdas
from sports.tick_recorder import TickRecorder
from sports.microstructure import MicrostructureScanner

# Tennis engine imports
from tennis.state import TennisState, PointScore, update_from_point, compute_momentum_delta
from tennis.model import get_win_prob as tennis_get_win_prob
from tennis.strategy import InflectionStrategy, TennisSignal
from tennis.execution import TennisExecutionGuard
from tennis.exit_manager import TennisExitManager
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
    TENNIS_PRICE_FLOOR, TENNIS_MAX_SIGNALS_HR, TENNIS_PRICE_FLOOR_BYPASS,
    # v4.6 Tennis Entry Timing
    TENNIS_ENTRY_DELAY_S, TENNIS_ENTRY_CONFIRM_TICKS,
    TENNIS_ENTRY_MIN_EDGE, TENNIS_EDGE_DECAY_THRESH, TENNIS_ENTRY_MAX_DELAY_S,
    TENNIS_STALE_OVERRIDE_EDGE,
    TENNIS_STALE_DISABLE_COUNT, TENNIS_STALE_DISABLE_S,
    TENNIS_LIVE_MODE, TENNIS_BANKROLL, TENNIS_KELLY_PCT,
    TENNIS_MIN_ORDER_USD, POLY_PRIVATE_KEY, POLY_FUNDER_ADDRESS,
    CLOB_PROXY_URL,
)
from sports.guards import validate_trade_execution, circuit_breaker, STRAT_TENNIS_INFLECTION, STRAT_TENNIS_SB, STRAT_CRICKET_MOM
from tennis.live_executor import LiveExecutor
from tennis.spread_breakout import SpreadBreakoutDetector
from tennis.signal_snapshots import SignalSnapshotScheduler
from tennis.pending_store import PendingStore

# Cricket engine imports
from cricket import (
    CricketState, CricketFeed, CricketStrategy,
    CricketExecutionGuard, CricketCSVLogger,
    CricketBookHealthMonitor, check_cricket_readiness,
    ReadinessStatus, SpreadPhase, get_spread_phase,
    FailureReason, get_liquidity_threshold,
)
from cricket.tick_strategy import CricketTickDetector
from cricket.exit_manager import CricketExitManager
from cricket.live_executor import CricketLiveExecutor
# Cricket v1.0.1 — Sportmonks pipeline
from cricket.sm_feeds import SmCricketFeed
from cricket.sm_state import SmCricketState
from cricket.sm_strategy import SmCricketStrategy
from cricket.sm_mapping import get_mapping, get_all_fixture_ids, add_mapping
from sports.config import (
    CRICKET_PAPER_ONLY, CRICKET_TRADE_SIZE, CRICKET_MAX_SPREAD,
    CRICKET_MOMENTUM_RR_THRESH, CRICKET_MOMENTUM_EDGE,
    CRICKET_WICKET_EDGE, CRICKET_LATENCY_THRESH_MS,
    CRICKET_COOLDOWN_S,
    CRICKET_READINESS_CHECK_INTERVAL_S,
    CRICKET_LIVE_MODE, CRICKET_BANKROLL, CRICKET_KELLY_PCT,
    CRICKET_MIN_ORDER_USD, CRICKET_LIMIT_OFFSET,
    SPORTMONKS_API_TOKEN, CRICKET_SM_POLL_S,
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


# ═══════════════════════════════════════════════════════════════════════
#  v9.2: Tournament Tier Classifier
# ═══════════════════════════════════════════════════════════════════════

def _classify_tournament(name: str) -> str:
    """Classify a tennis tournament name into a tier for logging."""
    n = name.lower()
    if any(x in n for x in ["grand slam", "wimbledon", "roland garros",
                             "us open", "australian open"]):
        return "slam"
    if any(x in n for x in ["masters", "1000", "rome", "madrid",
                             "indian wells", "miami", "shanghai",
                             "montreal", "cincinnati", "paris"]):
        return "atp1000"
    if "500" in n or any(x in n for x in ["barcelona", "hamburg", "vienna",
                                           "beijing", "washington"]):
        return "atp500"
    if "250" in n:
        return "atp250"
    if any(x in n for x in ["wta 1000", "wta1000"]):
        return "wta1000"
    if "wta" in n:
        return "wta"
    if any(x in n for x in ["challenger", "itf", "futures"]):
        return "low_tier"
    return "unknown"


class SportsOrchestrator:
    """Main async orchestrator for the sports trading system."""

    def __init__(self, target_date: str):
        self.target_date = target_date
        self.football_feed = FootballFeed()
        self.nba_feed = NBAFeed()
        self.ncaa_feed = NCAAFeed()
        self.poly_feed = PolymarketFeed()
        self.engine = SignalEngine(DATA_DIR)
        self.tick_recorder = TickRecorder()
        self.micro_scanner = MicrostructureScanner()
        self.markets: list[SportMarket] = []
        self.links: dict[str, GameMarketLink] = {}  # game_id → link
        self._shutdown = False
        self._tennis_pending: dict[tuple, dict] = {}  # v4.6: (match_id, token, dir) → pending state
        self._tennis_active: set = set()  # v10: canonical trade_key set for duplicate prevention
        self._session: aiohttp.ClientSession | None = None

        # ── Tennis Engine ─────────────────────────────────────────
        self.tennis_strategy = InflectionStrategy(
            panic_edge_threshold=TENNIS_PANIC_EDGE,
            reversion_edge_threshold=TENNIS_REVERSION_EDGE,
            price_floor=TENNIS_PRICE_FLOOR,
            price_floor_bypass_edge=TENNIS_PRICE_FLOOR_BYPASS,
        )
        self.tennis_guard = TennisExecutionGuard(
            price_cap=TENNIS_PRICE_CAP,
            staleness_s=TENNIS_STALENESS_S,
            cooldown_s=TENNIS_COOLDOWN_S,
            max_signals_per_hour=TENNIS_MAX_SIGNALS_HR,
            stale_disable_count=TENNIS_STALE_DISABLE_COUNT,
            stale_disable_s=TENNIS_STALE_DISABLE_S,
        )
        self.tennis_logger = TennisCSVLogger(DATA_DIR)
        self.tennis_exit_mgr = TennisExitManager(
            DATA_DIR,
            on_close=self._tennis_live_sell_callback,
        )

        # ── Tennis Live Executor (v4.4) ───────────────────────────
        self.tennis_live = None
        if TENNIS_LIVE_MODE:
            self.tennis_live = LiveExecutor(
                private_key=POLY_PRIVATE_KEY,
                funder_address=POLY_FUNDER_ADDRESS,
                api_key=POLYMARKET_API_KEY,
                api_secret=POLYMARKET_SECRET,
                api_passphrase=POLYMARKET_PASSPHRASE,
                proxy_url=CLOB_PROXY_URL,
                initial_bankroll=TENNIS_BANKROLL,
                kelly_pct=TENNIS_KELLY_PCT,
                min_order_usd=TENNIS_MIN_ORDER_USD,
                data_dir=DATA_DIR,
            )
            if self.tennis_live.is_ready:
                log.info("TENNIS LIVE MODE: ON | bankroll=$%.2f | kelly=%.0f%%",
                         TENNIS_BANKROLL, TENNIS_KELLY_PCT * 100)
            else:
                log.warning("TENNIS LIVE MODE: credentials missing — paper only")
                self.tennis_live = None
        self.tennis_score_feed = TennisScoreFeed(poll_interval_s=TENNIS_FEED_POLL_S)
        self.tennis_markets: list[SportMarket] = []  # discovered tennis markets
        self.tennis_links: dict[str, GameMarketLink] = {}  # match_id → link
        self.tennis_states: dict[str, TennisState] = {}  # match_id → latest state
        self._tennis_fs_map: dict[str, str] = {}  # poly_event_id → flashscore_match_id

        # ── Spread Breakout Detector (v5.2) ────────────────────
        self.sb_detector = SpreadBreakoutDetector()
        # v7.1: Real price-change tracking for stale market filter
        self._tennis_last_price: dict[str, tuple] = {}    # token_id → (bid, ask)
        self._tennis_price_change_ts: dict[str, float] = {}  # token_id → last change ts
        # v9.2: Per-tier R tracking for summary logging
        self._tier_r: dict[str, list] = {}  # tier → [R values]
        self._tier_trade_count = 0
        # v9.7: Post-signal price snapshot scheduler
        self._signal_snapshots = SignalSnapshotScheduler(self.poly_feed)
        # v9.8A: Pending queue persistence
        self._pending_store = PendingStore(DATA_DIR)
        # v10: Max pending lifetime (seconds)
        self._MAX_PENDING_LIFETIME = 45

        # ── Cricket Engine (Paper + Live) ──────────────────────────
        # Legacy (kept for backwards compat)
        self.cricket_feed = CricketFeed()
        self.cricket_strategy = CricketStrategy(
            momentum_rr_threshold=CRICKET_MOMENTUM_RR_THRESH,
            momentum_edge_threshold=CRICKET_MOMENTUM_EDGE,
            wicket_edge_threshold=CRICKET_WICKET_EDGE,
            latency_threshold_ms=CRICKET_LATENCY_THRESH_MS,
        )
        self.cricket_guard = CricketExecutionGuard(
            max_spread=CRICKET_MAX_SPREAD,
            cooldown_s=CRICKET_COOLDOWN_S,
            trade_size_usd=CRICKET_TRADE_SIZE,
        )
        self.cricket_logger = CricketCSVLogger(DATA_DIR)
        self.cricket_exit_mgr = CricketExitManager(
            data_dir=DATA_DIR,
            on_close=self._cricket_live_sell_callback,
        )
        self.cricket_links: dict[str, GameMarketLink] = {}  # match_id → link
        self.cricket_states: dict[str, CricketState] = {}  # match_id → latest state
        self._cricket_price_buf: dict[str, list[float]] = {}  # match_id → last N mid prices
        self._cricket_last_trade_ts: dict[str, float] = {}    # match_id → last trade timestamp
        self._cricket_prev_spread: dict[str, float] = {}      # match_id → previous spread
        # v6.0: Tick-based reversion detector (no ESPN dependency)
        self.cricket_tick_detector = CricketTickDetector()
        # v7.0: Book health monitor + readiness
        self.cricket_health = CricketBookHealthMonitor()
        self._cricket_readiness_status: dict[str, str] = {}  # match_id → last status
        # v7.1: Runtime state tracking (DEAD → ACTIVE activation)
        self._cricket_runtime_state: dict[str, str] = {}     # match_id → DEAD/ACTIVE
        self._cricket_state_log_ts: dict[str, float] = {}    # throttled state logging

        # ── Cricket v1.0.1: Sportmonks Pipeline ─────────────────────
        self.sm_cricket_feed = SmCricketFeed(
            api_token=SPORTMONKS_API_TOKEN,
            poll_interval_s=CRICKET_SM_POLL_S,
        )
        self.sm_cricket_state = SmCricketState()
        self.sm_cricket_strategy = SmCricketStrategy()
        log.info("CRICKET v1.0.1 | Sportmonks scoreboard mode | poll=%.0fs", CRICKET_SM_POLL_S)

        # ── Cricket Live Executor ─────────────────────────────────
        self.cricket_live = None
        if CRICKET_LIVE_MODE and not CRICKET_PAPER_ONLY:
            self.cricket_live = CricketLiveExecutor(
                private_key=POLY_PRIVATE_KEY,
                funder_address=POLY_FUNDER_ADDRESS,
                api_key=POLYMARKET_API_KEY,
                api_secret=POLYMARKET_SECRET,
                api_passphrase=POLYMARKET_PASSPHRASE,
                proxy_url=CLOB_PROXY_URL,
                initial_bankroll=CRICKET_BANKROLL,
                kelly_pct=CRICKET_KELLY_PCT,
                min_order_usd=CRICKET_MIN_ORDER_USD,
                limit_offset=CRICKET_LIMIT_OFFSET,
                data_dir=DATA_DIR,
            )
            if self.cricket_live.is_ready:
                log.info(
                    "CRICKET LIVE MODE: ON | bankroll=$%.2f | kelly=%.0f%% | offset=%.3f",
                    CRICKET_BANKROLL, CRICKET_KELLY_PCT * 100, CRICKET_LIMIT_OFFSET,
                )
            else:
                log.warning("CRICKET LIVE MODE: credentials missing — paper only")
                self.cricket_live = None

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
                await self.ncaa_feed.fetch_live_scores(session)
                self.nba_feed.games.update(self.ncaa_feed.games)  # merge NCAA into NBA
                
                log.info("fetching cricket scoreboard...")
                await self.cricket_feed.fetch_live_scores(session)


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
            await self.ncaa_feed.fetch_live_scores(session)
            self.nba_feed.games.update(self.ncaa_feed.games)
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

            # v9.2: Extract tournament name and tier
            _tourn_name = tm.title.split(":")[0].strip() if ":" in tm.title else tm.title
            _tourn_tier = _classify_tournament(_tourn_name)

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
            link.tournament = _tourn_name
            link.tier = _tourn_tier
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

        # ── Cricket market links (v7.0 — deterministic token assignment) ──
        self.cricket_markets = [m for m in self.markets if m.sport == "cricket"]
        for cm in self.cricket_markets:
            if cm.event_id in self.cricket_links:
                continue
            
            # Deterministic token assignment: outcomes[0] → yes (home), outcomes[1] → no (away)
            all_tids = [o.token_id for o in cm.outcomes]
            home_tid = cm.outcomes[0].token_id if len(cm.outcomes) >= 1 else ""
            away_tid = cm.outcomes[1].token_id if len(cm.outcomes) >= 2 else ""

            # Parse team names from title
            home_name, away_name = self._parse_cricket_teams(cm.title)

            link = GameMarketLink(
                game_id=cm.event_id,
                sport="cricket",
                league=cm.league,
                home_team=home_name or cm.title,
                away_team=away_name or "",
                polymarket_event_id=cm.event_id,
                polymarket_title=cm.title,
                polymarket_slug=cm.slug,
                home_token_id=home_tid,
                away_token_id=away_tid,
                all_token_ids=all_tids,
            )
            
            self.cricket_links[cm.event_id] = link
            # v7.1: Initialize runtime state from discovery
            init_state = getattr(cm, 'initial_state', 'ACTIVE')
            self._cricket_runtime_state[cm.event_id] = init_state
            log.info(
                "CRICKET LINK: %s | tokens=%d | home_tid=%s | away_tid=%s | state=%s",
                cm.title, len(all_tids),
                home_tid[:12] + '...' if home_tid else 'NONE',
                away_tid[:12] + '...' if away_tid else 'NONE',
                init_state,
            )

        log.info("matched %d games + %d tennis + %d cricket matches to Polymarket",
                 len(self.links), len(self.tennis_links), len(self.cricket_links))

        # Collect token IDs for WS subscription
        all_tokens = []
        for link in self.links.values():
            all_tokens.extend(link.all_token_ids)
        for link in self.tennis_links.values():
            all_tokens.extend(link.all_token_ids)
        for link in self.cricket_links.values():
            all_tokens.extend(link.all_token_ids)

        log.info("subscribing to %d token IDs for %d matched games on Polymarket WS",
                 len(all_tokens), len(self.links) + len(self.tennis_links))
        self.poly_feed.set_tokens(all_tokens)

        # Register tick recorder + microstructure scanner callbacks
        self.poly_feed._on_update_callbacks = [
            self.tick_recorder.record_tick,
            self.micro_scanner.on_book_update,
        ]
        # Register token labels for queryability (all sports including cricket)
        all_links = (list(self.links.values()) + list(self.tennis_links.values())
                     + list(self.cricket_links.values()))
        for link in all_links:
            for tid in link.all_token_ids:
                self.tick_recorder.set_label(tid, link.polymarket_title, link.sport)
                self.micro_scanner.register_token(tid, link.polymarket_title, link.sport)

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
                            await self.ncaa_feed.fetch_live_scores(session)
                            self.nba_feed.games.update(self.ncaa_feed.games)
                        except Exception as e:
                            log.error("nba feed error: %s", e)

                        # Cricket
                        try:
                            await self.cricket_feed.fetch_live_scores(session)
                        except Exception as e:
                            log.error("cricket feed error: %s", e)

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
                import traceback
                log.error("signal processing error:\n%s", traceback.format_exc())

            await asyncio.sleep(POLYMARKET_SNAPSHOT_S)

    async def _tennis_score_polling_loop(self):
        """Poll Flashscore for live tennis scores and update TennisState."""
        # Wait for initial discovery + link building
        await asyncio.sleep(5)
        await self.tennis_score_feed.start()
        log.info("Tennis score polling started")
        last_score_diag = 0  # v4.6.3 diagnostic

        while not self._shutdown:
            try:
                count = await self.tennis_score_feed.poll_once()

                # v4.6.3: diagnostic counters for score feed
                sd = {"total": 0, "no_fs": 0, "not_live_fs": 0, "no_old_state": 0, "updated": 0}

                # For each Polymarket tennis link, find matching Flashscore match
                for poly_id, link in list(self.tennis_links.items()):
                    sd["total"] += 1
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

                    if not fs_match:
                        sd["no_fs"] += 1
                        continue
                    if not fs_match.is_live:
                        sd["not_live_fs"] += 1
                        continue

                    # Update TennisState from Flashscore data
                    old_state = self.tennis_states.get(poly_id)
                    if not old_state:
                        sd["no_old_state"] += 1
                        continue

                    sd["updated"] += 1

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
                    point_map = {"0": PointScore.LOVE, "15": PointScore.P15,
                                 "30": PointScore.P30, "40": PointScore.P40,
                                 "A": PointScore.AD, "AD": PointScore.AD,
                                 "50": PointScore.AD}
                    try:
                        pt_a = point_map.get(str(point_a_raw), PointScore.LOVE)
                        pt_b = point_map.get(str(point_b_raw), PointScore.LOVE)
                    except Exception:
                        pt_a, pt_b = PointScore.LOVE, PointScore.LOVE

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

                # v4.6.3: score feed diagnostic every 60s
                now_sd = time.time()
                if now_sd - last_score_diag >= 60:
                    last_score_diag = now_sd
                    # Count how many states have non-zero scores
                    live_count = sum(1 for s in self.tennis_states.values()
                                    if s.sets_a > 0 or s.sets_b > 0 or s.games_a > 0 or s.games_b > 0)
                    log.info("TENNIS_SCORE_DIAG | poly=%d mapped=%d no_fs=%d not_live_fs=%d updated=%d | states_with_scores=%d/%d",
                             sd["total"], sd["total"] - sd["no_fs"], sd["no_fs"],
                             sd["not_live_fs"], sd["updated"],
                             live_count, len(self.tennis_states))

            except Exception as e:
                log.error("tennis score poll error: %s", e)

            await asyncio.sleep(TENNIS_FEED_POLL_S)

    async def _tennis_signal_loop(self):
        """Tennis Strategy B signal processing — runs on every tick.

        v2.0: Integrates should_evaluate() pre-check, state dedup via
        selection_id, position loop breaker, and health stats.
        """
        # Log health summary every hour
        last_health_log = time.time()
        last_diag_log = 0  # v4.6.2 diagnostic

        # v9.8A: Restore persisted pending entries on startup
        self._restore_pending_from_disk()

        while not self._shutdown:
            try:
                # v4.6.2: diagnostic counters
                diag = {"total": 0, "no_state": 0, "not_live": 0, "no_fav": 0,
                        "no_book": 0, "guard_skip": 0, "no_signal": 0,
                        "guard_block": 0, "signal_ok": 0}

                for match_id, link in list(self.tennis_links.items()):
                    diag["total"] += 1
                    state = self.tennis_states.get(match_id)
                    if not state:
                        diag["no_state"] += 1
                        continue

                    # Only process live matches (state updated by score feed)
                    if state.sets_a == 0 and state.sets_b == 0 and state.games_a == 0 and state.games_b == 0:
                        # State never updated — match not live yet
                        diag["not_live"] += 1
                        continue

                    # Get current market price for the favorite
                    fav_token = link.home_token_id if state.pregame_favorite_id == link.home_team else link.away_token_id
                    fav_book = self.poly_feed.books.get(fav_token)
                    if not fav_book or fav_book.mid <= 0:
                        diag["no_book"] += 1
                        continue

                    market_price = fav_book.mid

                    # v7.1: Track real price changes (not heartbeats)
                    _prev = self._tennis_last_price.get(fav_token)
                    _cur = (fav_book.best_bid, fav_book.best_ask)
                    if _prev != _cur:
                        self._tennis_price_change_ts[fav_token] = time.time()
                        self._tennis_last_price[fav_token] = _cur

                    # ── v4.6.5: Pending entry fast-path ───────────────
                    # If we already have a pending entry for this match,
                    # bypass strategy.evaluate() (which has state dedup
                    # and would suppress the signal) and directly recheck
                    # the edge from stored fair_price vs current market.
                    pending_key = (match_id, fav_token, "BUY")
                    if pending_key in self._tennis_pending:
                        pend = self._tennis_pending[pending_key]
                        now_t = time.time()
                        elapsed_p = now_t - pend["start_time"]
                        current_edge = pend["signal"].fair_price - market_price
                        pend["last_edge"] = current_edge
                        pend["market_price"] = market_price

                        # Edge persistence check
                        if current_edge >= TENNIS_ENTRY_MIN_EDGE:
                            pend["confirm_count"] += 1
                        else:
                            pend["confirm_count"] = max(0, pend["confirm_count"] - 1)

                        # Decay guard
                        if pend["initial_edge"] > 0 and current_edge < pend["initial_edge"] * (1 - TENNIS_EDGE_DECAY_THRESH):
                            log.info("TENNIS_PENDING_DROPPED | reason=edge_decay | trade_id=%s | edge %.4f → %.4f | %s",
                                     pend.get("trade_id", "?"), pend["initial_edge"], current_edge,
                                     link.polymarket_title[:40])
                            # v10: Remove from active set on drop
                            self._tennis_active.discard(pend.get("trade_key", ""))
                            del self._tennis_pending[pending_key]
                            self._pending_store.save(self._tennis_pending)
                            continue

                        # v10: MAX_PENDING_LIFETIME (45s hard cap)
                        if elapsed_p > self._MAX_PENDING_LIFETIME:
                            log.info("TENNIS_PENDING_DROPPED | reason=timeout | trade_id=%s | %.0fs | %s",
                                     pend.get("trade_id", "?"), elapsed_p, link.polymarket_title[:40])
                            self._tennis_active.discard(pend.get("trade_key", ""))
                            del self._tennis_pending[pending_key]
                            self._pending_store.save(self._tennis_pending)
                            continue

                        # Check entry conditions
                        time_ok = elapsed_p >= TENNIS_ENTRY_DELAY_S
                        confirm_ok = pend["confirm_count"] >= TENNIS_ENTRY_CONFIRM_TICKS
                        edge_ok = current_edge >= TENNIS_ENTRY_MIN_EDGE

                        if not (time_ok and confirm_ok and edge_ok):
                            log.info("TENNIS_PENDING_UPDATE | time=%.0f/%ds conf=%d/%d edge=%.4f | %s",
                                     elapsed_p, TENNIS_ENTRY_DELAY_S,
                                     pend["confirm_count"], TENNIS_ENTRY_CONFIRM_TICKS,
                                     current_edge, link.polymarket_title[:30])
                            continue

                        # v10: One-shot entry attempt — never attempt twice
                        if pend.get("attempted", False):
                            continue
                        pend["attempted"] = True

                        # All conditions met → execute
                        log.info("TENNIS_ENTRY_ATTEMPT | %s | mkt=%.4f | edge=%.4f | elapsed=%.0fs",
                                 link.polymarket_title[:40], market_price, current_edge, elapsed_p)
                        entry_delay_actual = elapsed_p
                        confirm_at_entry = pend["confirm_count"]
                        signal = pend["signal"]
                        state = pend["state"]
                        del self._tennis_pending[pending_key]
                        self._pending_store.save(self._tennis_pending)

                        # ═══════════════════════════════════════════════
                        # HARD PRICE BAND GATE (v5.1 — non-bypassable)
                        # Final check on CURRENT market_price before any
                        # execution. No trade can pass outside [0.20, 0.80].
                        # ═══════════════════════════════════════════════
                        if market_price < 0.20 or market_price > 0.80:
                            log.info("TENNIS_SIGNAL_REJECTED | %s | match=%s | mkt=%.4f | edge=%.4f | reason=PRICE_FLOOR",
                                     state.pregame_favorite_id or link.home_team or "?",
                                     match_id, market_price, current_edge)
                            # v10: Remove from active on rejection
                            self._tennis_active.discard(pend.get("trade_key", ""))
                            continue

                        # ── v7.1 / v9.9: Price-change activity filter ──
                        # Tennis books can be stale between points. Allow
                        # bypass when edge is strong (v9.9).
                        price_age = time.time() - self._tennis_price_change_ts.get(fav_token, time.time())
                        if price_age > 60:
                            if current_edge >= TENNIS_STALE_OVERRIDE_EDGE:
                                log.info(
                                    "TENNIS_STALE_OVERRIDE | %s | price_age=%.0fs | edge=%.4f | ALLOWED",
                                    link.polymarket_title[:40], price_age, current_edge,
                                )
                            else:
                                log.info(
                                    "TENNIS_SIGNAL_REJECTED | %s | match=%s | mkt=%.4f | edge=%.4f | reason=NO_PRICE_MOVEMENT",
                                    state.pregame_favorite_id or link.home_team or "?",
                                    match_id, market_price, current_edge,
                                )
                                # v10: Remove from active on rejection
                                self._tennis_active.discard(pend.get("trade_key", ""))
                                continue

                        log.info("TENNIS_DELAYED_ENTRY | tier=%s | tourn=%s | delay=%.0fs confirm=%d edge=%.4f | %s",
                                 link.tier, link.tournament,
                                 entry_delay_actual, confirm_at_entry,
                                 current_edge, link.polymarket_title[:40])
                        diag["signal_ok"] += 1

                        # Jump to entry execution (reuse signal + state)
                        # ═══ GLOBAL EDGE GUARD (v6.0) ═══
                        can_exec, block_reason = validate_trade_execution(
                            edge=current_edge,
                            price=market_price,
                            sport="tennis",
                            context=f"{signal.trigger_type} | {link.polymarket_title[:50]}",
                            strategy=STRAT_TENNIS_INFLECTION,
                        )
                        if not can_exec:
                            continue

                        self.tennis_guard.record_entry(
                            match_id, state_key=self.tennis_strategy._state_key(state, fav_token),
                            edge=current_edge
                        )
                        self.tennis_logger.log_trade_entry(signal, market_price_at_bp=market_price)
                        score_str = f"{state.sets_a}-{state.sets_b} {state.games_a}-{state.games_b}"
                        fav_name = state.pregame_favorite_id or link.home_team
                        current_spread = fav_book.spread if fav_book else 0.0
                        self.tennis_exit_mgr.register_trade(
                            match_id=match_id, selection_id=fav_token,
                            player=fav_name, trigger_type=signal.trigger_type,
                            entry_price=market_price, fair_value=signal.fair_price,
                            edge=current_edge, entry_score=score_str, spread=current_spread,
                            tournament=link.tournament, tier=link.tier,
                        )
                        log.info("TENNIS_ENTRY_CONFIRMED [v10] | %s | trade_id=%s | entry_price=%.4f | edge=%.4f | delay=%.0fs | %s %d-%d %d-%d",
                                 signal.trigger_type, pend.get("trade_id", "?"),
                                 market_price, current_edge,
                                 entry_delay_actual, link.polymarket_title[:30],
                                 state.sets_a, state.sets_b, state.games_a, state.games_b)
                        # Live order
                        live_tag = "PAPER"
                        if self.tennis_live and self.tennis_live.is_ready:
                            result = self.tennis_live.buy(token_id=fav_token, price=market_price,
                                                          match_info=f"{link.polymarket_title[:40]} {state.sets_a}-{state.sets_b}")
                            if result.success:
                                live_tag = f"LIVE ${result.filled_size:.2f}"
                                self.tennis_live.record_fill(match_id)
                            else:
                                live_tag = f"LIVE FAIL: {result.error}"
                        try:
                            await self.engine.tg.send(
                                f"🎾 <b>Tennis Signal [{live_tag}]</b>\n"
                                f"Trigger: {signal.trigger_type}\n"
                                f"Edge: {current_edge:+.4f} (delay={entry_delay_actual:.0f}s)\n"
                                f"Fair: {signal.fair_price:.4f} | Mkt: {market_price:.4f}\n"
                                f"Match: {link.polymarket_title}\n"
                                f"Score: {state.sets_a}-{state.sets_b} | {state.games_a}-{state.games_b}"
                            )
                        except Exception:
                            pass
                        continue  # done with this match

                    # ── Pre-check: should we even evaluate? ───────
                    # Compute state key for position loop breaker
                    state_key = self.tennis_strategy._state_key(state, fav_token)

                    if not self.tennis_guard.should_evaluate(
                        match_id, state_key=state_key, edge=0.0
                    ):
                        diag["guard_skip"] += 1
                        continue

                    # Run Strategy B evaluation (with price floor + dedup built in)
                    signal = self.tennis_strategy.evaluate(
                        state, market_price, selection_id=fav_token
                    )
                    if signal is None:
                        diag["no_signal"] += 1
                        continue

                    # Log signal
                    self.tennis_logger.log_signal(signal)
                    log.info("TENNIS SIGNAL | %s | tier=%s | tourn=%s | edge=%+.4f | fair=%.4f | mkt=%.4f",
                             signal.trigger_type, link.tier, link.tournament,
                             signal.edge, signal.fair_price, market_price)

                    # v9.7: Schedule post-signal price snapshots (+5s, +10s, +30s)
                    _snap_id = f"{match_id}_{int(time.time())}"
                    self._signal_snapshots.schedule(
                        signal_id=_snap_id,
                        token_id=fav_token,
                        signal_time=time.time(),
                        price_signal=market_price,
                        match_id=match_id,
                        trigger_type=signal.trigger_type,
                        logger=self.tennis_logger,
                    )

                    # Check execution guards
                    decision = self.tennis_guard.can_execute(signal, state)
                    if not decision.can_execute:
                        diag["guard_block"] += 1
                        log.info("TENNIS BLOCKED | %s | %s", decision.reason, match_id)
                        continue

                    diag["signal_ok"] += 1

                    # ── v5.1: Price band pre-check (before pending) ──
                    # Block even creating a pending entry for out-of-band prices.
                    if market_price < 0.20 or market_price > 0.80:
                        log.info("TENNIS_SIGNAL_REJECTED | %s | match=%s | mkt=%.4f | edge=%.4f | reason=PRICE_FLOOR",
                                 state.pregame_favorite_id or link.home_team or "?",
                                 match_id, market_price, signal.edge if signal else 0.0)
                        continue

                    # ── v4.6.5: Create pending entry (first signal tick) ──
                    # The fast-path above (line ~853) handles all subsequent
                    # ticks: persistence, decay, expiry, and execution.
                    pending_key = (match_id, fav_token, "BUY")
                    now_t = time.time()
                    fav_name = state.pregame_favorite_id or link.home_team or "?"
                    trade_key = f"{match_id}:{fav_name}"
                    trade_id = f"{match_id}_{fav_name}_{time.time_ns()}"

                    # v10: Duplicate lock — skip if already pending or active
                    if trade_key in self._tennis_active:
                        log.info("TENNIS_SKIP_DUPLICATE | %s | %s", fav_name, match_id)
                        continue

                    self._tennis_active.add(trade_key)
                    self._tennis_pending[pending_key] = {
                        "start_time": now_t,
                        "confirm_count": 1,
                        "initial_edge": signal.edge,
                        "last_edge": signal.edge,
                        "signal": signal,
                        "link": link,
                        "state": state,
                        "fav_token": fav_token,
                        "fav_book": fav_book,
                        "market_price": market_price,
                        "trade_id": trade_id,
                        "trade_key": trade_key,
                        "attempted": False,
                    }
                    # v9.8A: Persist to disk + enhanced logging
                    self._pending_store.save(self._tennis_pending)
                    log.info("TENNIS_PENDING_CREATED [v10] | %s | %s | trade_id=%s | mkt=%.4f | edge=%.4f | delay=%ds | %s",
                             fav_name, signal.trigger_type, trade_id, market_price,
                             signal.edge, TENNIS_ENTRY_DELAY_S,
                             link.polymarket_title[:40])

                # ── Spread Breakout: tick detection (v5.2) ─────────
                # Runs on ALL live matches with book data, independent
                # of mean reversion state/dedup. Uses both token sides.
                for sb_mid, sb_link in list(self.tennis_links.items()):
                    for sb_tid in [sb_link.home_token_id, sb_link.away_token_id]:
                        if not sb_tid:
                            continue
                        sb_book = self.poly_feed.books.get(sb_tid)
                        if not sb_book or sb_book.mid <= 0:
                            continue
                        sb_spread = sb_book.spread if sb_book.spread else 0
                        sb_sig = self.sb_detector.tick(
                            token_id=sb_tid,
                            match_id=sb_mid,
                            mid=sb_book.mid,
                            spread=sb_spread,
                            match_title=sb_link.polymarket_title,
                        )
                        if sb_sig:
                            # ═══ GLOBAL EDGE GUARD (v6.0) ═══
                            sb_edge = sb_sig.get("edge", 0)
                            can_exec, block_reason = validate_trade_execution(
                                edge=sb_edge if sb_edge > 0 else max(abs(sb_sig["entry_price"] - sb_sig.get("pre_widen_mid", sb_sig["entry_price"])), 0.01),
                                price=sb_sig["entry_price"],
                                sport="tennis_sb",
                                context=f"SPREAD_BREAKOUT {sb_sig['direction']} | {sb_link.polymarket_title[:50]}",
                                strategy=STRAT_TENNIS_SB,
                            )
                            if not can_exec:
                                continue

                            # ── v7.1: Price-change activity filter ────
                            sb_price_age = time.time() - self._tennis_price_change_ts.get(sb_tid, time.time())
                            if sb_price_age > 60:
                                log.info(
                                    "TENNIS_BLOCK_REASON | %s | reason=NO_PRICE_MOVEMENT | price_age=%.1fs",
                                    sb_link.polymarket_title[:40], sb_price_age,
                                )
                                continue

                            # Register trade for exit tracking
                            player = sb_link.home_team if sb_tid == sb_link.home_token_id else sb_link.away_team
                            self.sb_detector.register_trade(
                                token_id=sb_tid,
                                match_id=sb_mid,
                                player=player or "?",
                                entry_price=sb_sig["entry_price"],
                                direction=sb_sig["direction"],
                            )
                            # v8.1: Spread breakout TG notifications removed (noise)
                            # Logging and execution remain active.

                # ── Spread Breakout: exit checks (v5.2) ────────────
                def _sb_get_price(tid):
                    b = self.poly_feed.books.get(tid)
                    return b.mid if b and b.mid > 0 else None

                sb_closed = self.sb_detector.check_exits(_sb_get_price)
                for sbt in sb_closed:
                    # v8.1: Spread breakout exit TG removed (noise) — log only
                    _sb_tier = sb_link.tier if hasattr(sb_link, 'tier') else "unknown"
                    _sb_tourn = sb_link.tournament if hasattr(sb_link, 'tournament') else ""
                    log.info("SB_EXIT [v10] | %s | trade_id=%s | %s | tier=%s | tourn=%s | entry=%.4f exit=%.4f R=%s | dur=%.0fs",
                             sbt.player, sbt.trade_id, sbt.exit_reason, _sb_tier, _sb_tourn,
                             sbt.entry_price, sbt.exit_price,
                             f"{sbt.r_multiple:+.4f}", sbt.duration_s)
                    # v10: Remove from active set on SB exit
                    _sb_trade_key = f"{sbt.match_id}:{sbt.player}"
                    self._tennis_active.discard(_sb_trade_key)
                    # v9.2: Accumulate per-tier R
                    self._tier_r.setdefault(_sb_tier, []).append(sbt.r_multiple)
                    self._tier_trade_count += 1
                    if self._tier_trade_count % 10 == 0:
                        _parts = []
                        for _t, _rs in sorted(self._tier_r.items()):
                            _parts.append(f"{_t}: {len(_rs)} trades R={sum(_rs):+.3f}")
                        log.info("TENNIS_TIER_SUMMARY | trades=%d | %s",
                                 self._tier_trade_count, " | ".join(_parts))
                    # v9.0: Feed per-strategy CB with SB exit outcome
                    circuit_breaker.record_trade_outcome(
                        sbt.r_multiple,
                        sport="tennis_sb",
                        strategy=STRAT_TENNIS_SB,
                    )

                # ── Exit Manager: check all open trades ───────────
                self._tennis_check_exits()

                # v4.6.2: diagnostic log every 60s
                now_diag = time.time()
                if now_diag - last_diag_log >= 60:
                    last_diag_log = now_diag
                    log.info("TENNIS_DIAG | total=%d no_state=%d not_live=%d no_book=%d "
                             "guard_skip=%d no_signal=%d guard_block=%d signal_ok=%d",
                             diag["total"], diag["no_state"], diag["not_live"],
                             diag["no_book"], diag["guard_skip"], diag["no_signal"],
                             diag["guard_block"], diag["signal_ok"])

                # ── Track WS reconnects + hourly health summary ───
                self.tennis_guard.stats.ws_reconnects = self.poly_feed.reconnect_count
                self.tennis_guard.stats.merge_exit_stats(self.tennis_exit_mgr.stats)
                now = time.time()
                if now - last_health_log >= 3600:
                    self.tennis_guard.stats.log_summary()
                    last_health_log = now

            except Exception as e:
                log.error("tennis signal loop error: %s", e)

            await asyncio.sleep(POLYMARKET_SNAPSHOT_S)

    # ── v9.8A: Pending queue restore from disk ──────────────────────

    def _restore_pending_from_disk(self):
        """Restore persisted pending entries after engine restart.

        For each entry, re-lookup the live link, state, and book objects.
        Reconstruct a TennisSignal from the stored scalars.
        Skip entries whose match is no longer tracked.
        """
        from tennis.state import TennisState, PointScore, TennisModelOutput
        from tennis.model import get_win_prob

        entries = self._pending_store.load(TENNIS_ENTRY_MAX_DELAY_S)
        restored = 0

        for entry in entries:
            match_id = entry.get("match_id", "")
            token_id = entry.get("token_id", "")
            direction = entry.get("direction", "BUY")

            # Look up live objects
            link = self.tennis_links.get(match_id)
            state = self.tennis_states.get(match_id)
            fav_book = self.poly_feed.books.get(token_id)

            if not link or not state:
                log.info("TENNIS_PENDING_RESTORE_SKIP | %s | reason=no_link_or_state", match_id)
                continue

            # Reconstruct TennisSignal from stored fields + live model
            try:
                model = get_win_prob(state)
                if state.pregame_favorite_id == state.player_a_id:
                    fair_fav = model.p_a
                else:
                    fair_fav = model.p_b

                signal = TennisSignal(
                    timestamp=entry.get("start_time", time.time()),
                    match_id=match_id,
                    trigger_type=entry.get("trigger_type", "SET_MEAN_REVERSION"),
                    edge=entry.get("signal_edge", 0.0),
                    fair_price=fair_fav,  # Use current fair value
                    market_price=entry.get("signal_market_price", 0.0),
                    state_snapshot=state,
                    model_output=model,
                )
            except Exception as e:
                log.warning("TENNIS_PENDING_RESTORE_FAIL | %s | %s", match_id, e)
                continue

            pending_key = (match_id, token_id, direction)

            # Dedup: skip if already in memory
            if pending_key in self._tennis_pending:
                continue

            self._tennis_pending[pending_key] = {
                "start_time": entry["start_time"],
                "confirm_count": entry.get("confirm_count", 1),
                "initial_edge": entry.get("initial_edge", 0.0),
                "last_edge": entry.get("last_edge", 0.0),
                "signal": signal,
                "link": link,
                "state": state,
                "fav_token": token_id,
                "fav_book": fav_book,
                "market_price": entry.get("market_price", 0.0),
            }
            remaining = TENNIS_ENTRY_MAX_DELAY_S - (time.time() - entry["start_time"])
            log.info(
                "TENNIS_PENDING_RESTORED_ENTRY | %s | remaining=%.0fs | edge=%.4f | %s",
                match_id, remaining, entry.get("initial_edge", 0),
                entry.get("polymarket_title", ""),
            )
            restored += 1

        if restored > 0:
            log.info("TENNIS_PENDING_RESTORE_COMPLETE | restored=%d", restored)

    def _tennis_check_exits(self):
        """Run ExitManager check_all with accessor lambdas."""

        def _get_mkt(match_id: str, selection_id: str):
            book = self.poly_feed.books.get(selection_id)
            if book and book.mid > 0:
                return book.mid
            return None

        def _get_fair(match_id: str):
            state = self.tennis_states.get(match_id)
            if not state:
                return None
            try:
                from tennis.model import get_win_prob
                out = get_win_prob(state)
                if state.pregame_favorite_id == state.player_a_id:
                    return out.p_a
                return out.p_b
            except Exception:
                return None

        def _get_score(match_id: str):
            state = self.tennis_states.get(match_id)
            if not state:
                return None
            return f"{state.sets_a}-{state.sets_b} {state.games_a}-{state.games_b}"

        def _is_finished(match_id: str):
            # Check Flashscore for match completion
            fs_id = self._tennis_fs_map.get(match_id)
            if fs_id:
                fs_match = self.tennis_score_feed._matches.get(fs_id)
                if fs_match and not fs_match.is_live:
                    return True
            return False

        self.tennis_exit_mgr.check_all(
            get_market_price=_get_mkt,
            get_fair_value=_get_fair,
            get_score=_get_score,
            is_match_finished=_is_finished,
        )

    def _tennis_live_sell_callback(self, trade):
        """Called by ExitManager when a trade closes — send Telegram + fire live SELL if filled."""
        # v10: Remove trade_key from active set on exit
        _tk = f"{trade.match_id}:{trade.player}"
        self._tennis_active.discard(_tk)

        exit_price = trade.exit_price or 0.0

        # Calculate $ PnL (paper estimate based on $3.90 trade size)
        if trade.entry_price and trade.entry_price > 0:
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
            pnl_usd = pnl_pct * 3.90  # approximate
        else:
            pnl_pct = 0.0
            pnl_usd = 0.0

        r_mult = trade.R_multiple if hasattr(trade, 'R_multiple') else 0.0
        win_emoji = "✅" if r_mult > 0 else "❌" if r_mult < -0.05 else "➖"

        # Running tally from closed trades
        closed = self.tennis_exit_mgr.closed_trades
        wins = sum(1 for t in closed if getattr(t, 'R_multiple', 0) > 0)
        losses = len(closed) - wins
        total_r = sum(getattr(t, 'R_multiple', 0) for t in closed)

        # Live sell if we had a live fill
        live_tag = "PAPER"
        if self.tennis_live and self.tennis_live.is_ready and self.tennis_live.has_live_fill(trade.match_id):
            sell_size = self.tennis_live.order_size
            match_desc = f"{trade.player} R={r_mult:+.4f}"
            result = self.tennis_live.sell(
                token_id=trade.selection_id,
                size_usd=sell_size,
                price=exit_price,
                match_info=match_desc,
            )
            self.tennis_live.record_exit_pnl(
                entry_size=self.tennis_live.order_size,
                exit_price=exit_price,
                entry_price=trade.entry_price,
            )
            live_tag = "LIVE SELL" if result.success else f"LIVE SELL FAIL: {result.error}"

        # ALWAYS send Telegram exit notification
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.engine.tg.send(
                    f"{win_emoji} <b>Tennis Exit [{live_tag}]</b>\n"
                    f"Player: {trade.player}\n"
                    f"Reason: {trade.exit_reason}\n"
                    f"Entry: {trade.entry_price:.4f} → Exit: {exit_price:.4f}\n"
                    f"<b>R: {r_mult:+.4f}</b> | ~${pnl_usd:+.2f}\n"
                    f"Duration: {trade.duration_seconds:.0f}s\n"
                    f"Record: {wins}W-{losses}L | ΣR: {total_r:+.4f}"
                ))
        except Exception:
            pass

    # ── Cricket Live Sell Callback ─────────────────────────────

    def _cricket_live_sell_callback(self, trade):
        """Called by CricketExitManager when a paper trade closes — fire live SELL."""
        exit_price = trade.exit_price or 0.0
        pnl = trade.paper_pnl

        # Determine direction from signal type
        # DLS signals default to LONG; tick signals encode direction in entry_score
        direction = "LONG"  # default
        if hasattr(trade, 'entry_score') and trade.entry_score:
            if trade.entry_score.startswith("SHORT"):
                direction = "SHORT"
            elif trade.entry_score.startswith("LONG"):
                direction = "LONG"

        # Live exit if we have a live fill for this match
        live_tag = "PAPER"
        if self.cricket_live and self.cricket_live.is_ready and self.cricket_live.has_live_fill(trade.match_id):
            token_id = trade.selection_id
            sell_size = self.cricket_live.order_size
            match_desc = f"{trade.signal_type} R={pnl:+.4f}"

            log.info(
                "CRICKET_EXECUTION | EXIT TRIGGER | match=%s | dir=%s "
                "| exit_price=%.4f | reason=%s",
                trade.match_id, direction, exit_price, trade.exit_reason,
            )

            result = self.cricket_live.place_exit(
                token_id=token_id,
                mid=exit_price,
                direction=direction,
                size_usd=sell_size,
                match_id=trade.match_id,
                match_info=match_desc,
            )

            self.cricket_live.record_exit_pnl(
                entry_size=self.cricket_live.order_size,
                exit_price=exit_price,
                entry_price=trade.entry_price,
            )
            live_tag = "LIVE SELL" if result.success else f"LIVE SELL FAIL: {result.error}"

        # Telegram notification
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            pnl_emoji = "✅" if pnl > 0 else "❌" if pnl < -0.005 else "➖"
            r_mult = pnl / trade.entry_price if trade.entry_price > 0 else 0.0
            if loop.is_running():
                loop.create_task(self.engine.tg.send(
                    f"📤 <b>Paper Exit</b>\n"
                    f"Entry: {trade.entry_price:.4f}\n"
                    f"Exit: {exit_price:.4f}\n"
                    f"R: {r_mult:+.2f}\n"
                    f"Reason: {trade.exit_reason}\n"
                    f"Duration: {trade.duration_seconds:.0f}s\n"
                    f"Match: {trade.match_title[:40]}"
                ))
        except Exception:
            pass

        # Release position lock so new signals can fire for this fixture
        try:
            fid = int(trade.match_id)
            self.sm_cricket_strategy.record_exit(fid)
        except (ValueError, AttributeError):
            pass

    # ── Cricket helpers (v4.9) ─────────────────────────────────────

    @staticmethod
    def _parse_cricket_teams(poly_title: str) -> tuple[str, str]:
        """Parse team names from a Polymarket cricket market title.

        Examples:
            'Indian Premier League: Mumbai Indians vs Kolkata Knight Riders'
            → ('mumbai indians', 'kolkata knight riders')

            'Legends Cricket League: Daredevils Delhi vs Royal Riders Punjab'
            → ('daredevils delhi', 'royal riders punjab')

            'T20 World Cup, Sub Regional Africa, Qualifier B: Ghana vs Saint Helena'
            → ('ghana', 'saint helena')
        """
        title = poly_title
        # 1. Remove league prefix before last ":"
        if ":" in title:
            title = title.rsplit(":", 1)[-1].strip()
        # 2. Remove prop suffix after " - "
        if " - " in title:
            title = title.split(" - ", 1)[0].strip()
        # 3. Split on " vs. " or " vs "
        for sep in [" vs. ", " vs "]:
            if sep in title.lower():
                idx = title.lower().index(sep)
                team_a = title[:idx].strip().lower()
                team_b = title[idx + len(sep):].strip().lower()
                return (team_a, team_b)
        return ("", "")

    # ── Team name alias map for matching ──────────────────────────
    _CRICKET_NAME_ALIASES: dict[str, str] = {
        "bangalore": "bengaluru",
        "rcb": "royal challengers bengaluru",
        "csk": "chennai super kings",
        "mi": "mumbai indians",
        "kkr": "kolkata knight riders",
        "srh": "sunrisers hyderabad",
        "dc": "delhi capitals",
        "rr": "rajasthan royals",
        "gt": "gujarat titans",
        "lsg": "lucknow super giants",
        "pbks": "punjab kings",
    }

    @staticmethod
    def _normalize_cricket_name(name: str) -> str:
        """Normalize a cricket team name for comparison.

        Applies:
          - lowercase + strip
          - alias replacement (bangalore → bengaluru, etc.)
        """
        n = name.lower().strip()
        for alias, canonical in SportsOrchestrator._CRICKET_NAME_ALIASES.items():
            n = n.replace(alias, canonical)
        return n

    @staticmethod
    def _teams_match(poly_name: str, espn_name: str) -> bool:
        """Check if a Poly team name matches an ESPN team name.

        Uses normalized names + substring containment in both directions.
        'mumbai indians' matches 'Mumbai Indians'
        'kolkata' matches 'Kolkata Knight Riders'
        'royal challengers bangalore' matches 'Royal Challengers Bengaluru'
        """
        pn = SportsOrchestrator._normalize_cricket_name(poly_name)
        en = SportsOrchestrator._normalize_cricket_name(espn_name)
        if not pn or not en:
            return False
        return pn in en or en in pn

    def _cricket_fallback_signals(
        self,
        match_id: str,
        book,
        link,
    ) -> list[dict]:
        """Generate fallback signals from pure price microstructure.

        Returns list of signal dicts with:
            signal_type: MEAN_REVERT | MOMENTUM
            direction: LONG | SHORT
            edge: abs(price_move)
            confidence: consecutive_ticks
            market_price: current mid
        """
        mid = book.mid
        if mid <= 0:
            return []

        # Update price buffer
        buf = self._cricket_price_buf.setdefault(match_id, [])
        buf.append(mid)
        if len(buf) > 60:  # keep last 60 ticks
            self._cricket_price_buf[match_id] = buf[-60:]
            buf = self._cricket_price_buf[match_id]

        # ── Hard safety filters ──────────────────────────────────
        if len(buf) < 10:
            return []
        if mid >= 0.90 or mid <= 0.10:
            return []
        if book.spread > 0.05:
            return []

        signals = []

        # ── MEAN_REVERT: single-tick jump ≥ 0.05 ─────────────────
        if len(buf) >= 2:
            tick_move = buf[-1] - buf[-2]
            if abs(tick_move) >= 0.05:
                # Continuation filter: check if 2-3 prior ticks were
                # already moving in the same direction as the jump.
                # If so, this is momentum — NOT an overshoot to fade.
                prior_same_dir = 0
                for k in range(len(buf) - 2, max(0, len(buf) - 5), -1):
                    if k == 0:
                        break
                    prior_delta = buf[k] - buf[k - 1]
                    if prior_delta == 0:
                        break
                    if (prior_delta > 0 and tick_move > 0) or \
                       (prior_delta < 0 and tick_move < 0):
                        prior_same_dir += 1
                    else:
                        break

                if prior_same_dir >= 2:
                    log.info(
                        "CRICKET_SKIP_CONTINUATION | move=%.4f | "
                        "prior_same_dir=%d | %s",
                        tick_move, prior_same_dir, match_id,
                    )
                else:
                    # Fade the move
                    direction = "LONG" if tick_move < 0 else "SHORT"
                    signals.append({
                        "signal_type": "MEAN_REVERT",
                        "direction": direction,
                        "edge": abs(tick_move),
                        "confidence": 1,
                        "market_price": mid,
                        "move": tick_move,
                        "ticks": 1,
                    })

        # ── MOMENTUM: ≥ 5 consecutive same-direction ticks, ≥ 0.03 ──
        if len(buf) >= 6:
            consecutive = 1
            for i in range(len(buf) - 1, max(0, len(buf) - 21), -1):
                if i == 0:
                    break
                delta = buf[i] - buf[i - 1]
                prev_delta = buf[min(i + 1, len(buf) - 1)] - buf[i] if i < len(buf) - 1 else delta
                if delta == 0:
                    break
                if i == len(buf) - 1:
                    # First iteration — just record direction
                    consecutive = 1
                    continue
                # Check same direction as the latest tick
                latest_delta = buf[-1] - buf[-2]
                if latest_delta == 0:
                    break
                if (delta > 0 and latest_delta > 0) or (delta < 0 and latest_delta < 0):
                    consecutive += 1
                else:
                    break

            if consecutive >= 5:
                total_move = buf[-1] - buf[-consecutive]
                if abs(total_move) >= 0.03:
                    # Follow the move
                    direction = "LONG" if total_move > 0 else "SHORT"
                    signals.append({
                        "signal_type": "MOMENTUM",
                        "direction": direction,
                        "edge": abs(total_move),
                        "confidence": consecutive,
                        "market_price": mid,
                        "move": total_move,
                        "ticks": consecutive,
                    })

        return signals

    # ═══════════════════════════════════════════════════════════════════
    #  Cricket v1.0.1: Sportmonks Scoreboard Signal Loop
    # ═══════════════════════════════════════════════════════════════════

    async def _cricket_sm_signal_loop(self):
        """Cricket v2.0 — Paper-only Sportmonks scoreboard signal loop.

        Polls mapped fixtures via Sportmonks v2 API, derives events
        from scoreboard deltas, evaluates strategy, and executes
        PAPER trades only. Full Telegram notifications.

        No Polymarket API calls for execution.
        """
        log.info("Cricket v2.0 PAPER-ONLY Sportmonks signal loop started")
        log.info("CRICKET ACTIVE")
        log.info("PAPER TRADING ENABLED")
        log.info("TELEGRAM ACTIVE")
        await asyncio.sleep(10)  # wait for discovery + WS init

        last_diag_log = time.time()

        async with aiohttp.ClientSession() as sm_session:
            while not self._shutdown:
                try:
                    fixture_ids = get_all_fixture_ids()

                    if not fixture_ids:
                        # STEP 9: Auto-discover live IPL fixtures
                        try:
                            url = f"{SmCricketFeed.BASE_URL}/livescores"
                            params = {
                                "api_token": SPORTMONKS_API_TOKEN,
                                "filter[league_id]": "1",
                                "include": "scoreboards",
                            }
                            async with sm_session.get(
                                url, params=params,
                                timeout=aiohttp.ClientTimeout(total=10),
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    for fix in (data.get("data") or []):
                                        auto_fid = fix.get("id", 0)
                                        if auto_fid and not get_mapping(auto_fid):
                                            home = fix.get("localteam", {}).get("name", "Home")
                                            away = fix.get("visitorteam", {}).get("name", "Away")
                                            add_mapping(
                                                fixture_id=auto_fid,
                                                home_team=home,
                                                away_team=away,
                                                poly_token_yes="PAPER_ONLY",
                                                poly_token_no="PAPER_ONLY",
                                                poly_market_title=f"{home} vs {away}",
                                            )
                                            log.info(
                                                "CRICKET_AUTO_MAP | fixture=%d | %s vs %s",
                                                auto_fid, home, away,
                                            )
                        except Exception as e:
                            log.warning("CRICKET_AUTO_MAP_ERR | %s", e)

                        fixture_ids = get_all_fixture_ids()
                        if not fixture_ids:
                            now = time.time()
                            if now - last_diag_log > 300:
                                log.info(
                                    "CRICKET_SM_IDLE | no fixtures mapped | "
                                    "no live IPL fixtures found"
                                )
                                last_diag_log = now
                            await asyncio.sleep(CRICKET_SM_POLL_S)
                            continue

                    for fid in fixture_ids:
                        if self._shutdown:
                            break

                        # ── 1. Poll Sportmonks scoreboard ─────────────
                        snapshot = await self.sm_cricket_feed.poll_fixture(
                            sm_session, fid
                        )
                        if snapshot is None:
                            continue

                        # ── 2. Derive events from scoreboard delta ────
                        event_result = self.sm_cricket_state.update(snapshot)
                        if event_result is None:
                            continue

                        # ── TG: Event firehose ────────────────────────
                        mapping = get_mapping(fid)
                        if not mapping:
                            log.warning(
                                "CRICKET_SM_NO_MAPPING | fixture=%d", fid
                            )
                            continue

                        try:
                            await self.engine.tg.send(
                                f"🏏 <b>EVENT</b>\n"
                                f"Type: {event_result.event.value}\n"
                                f"Score: {snapshot.runs}/{snapshot.wickets}\n"
                                f"Overs: {snapshot.overs:.1f}\n"
                                f"Match: {mapping.home_team} vs "
                                f"{mapping.away_team}"
                            )
                        except Exception:
                            pass

                        # ── 3. Get market data (synthetic fallback) ───
                        token_id = mapping.poly_token_yes
                        book = self.poly_feed.books.get(token_id)

                        # Paper mode: use book if available, else synthetic
                        if book and book.mid > 0:
                            market_price = book.mid
                            spread = book.spread
                        else:
                            market_price = 0.50  # synthetic mid
                            spread = 0.01        # synthetic spread

                        # ── 4. Evaluate strategy ──────────────────────
                        decision = self.sm_cricket_strategy.evaluate(
                            event_result=event_result,
                            market_price=market_price,
                            spread=spread,
                            book_age_s=0.0,  # paper mode: no stale check
                        )

                        if not decision.should_trade:
                            # ── TG: Skip notification ─────────────────
                            try:
                                await self.engine.tg.send(
                                    f"⏭️ <b>Cricket Skip</b>\n"
                                    f"Reason: {decision.skip_reason}\n"
                                    f"Event: {decision.event}\n"
                                    f"Match: {mapping.home_team} vs "
                                    f"{mapping.away_team}"
                                )
                            except Exception:
                                pass
                            continue

                        # ── 5. Paper trade entry ──────────────────────
                        direction = decision.direction
                        match_label = (
                            f"{mapping.home_team} vs {mapping.away_team}"
                        )

                        # ── TG: Trade Decision (WHY) ─────────────────
                        try:
                            await self.engine.tg.send(
                                f"🚨 <b>TRADE DECISION</b>\n"
                                f"Event: {decision.event}\n"
                                f"Regime: {decision.regime}\n"
                                f"Dir: {direction}\n"
                                f"Price: {market_price:.3f}\n"
                                f"Reason: {decision.reason}"
                            )
                        except Exception:
                            pass

                        log.info(
                            "CRICKET_PAPER_ENTRY | fixture=%d | %s | "
                            "dir=%s | mid=%.4f | spread=%.4f | "
                            "event=%s | regime=%s | pressure=%.2f",
                            fid, decision.reason, direction,
                            market_price, spread,
                            decision.event, decision.regime,
                            decision.pressure,
                        )

                        # Record position lock
                        self.sm_cricket_strategy.record_entry(fid)

                        # Register with exit manager (paper)
                        self.cricket_exit_mgr.register_trade(
                            match_id=str(fid),
                            selection_id=token_id,
                            signal_type=f"SM_{decision.event}",
                            entry_price=market_price,
                            fair_value=market_price,
                            edge=0.0,
                            entry_score=(
                                f"{decision.regime} {decision.event} → "
                                f"{direction}"
                            ),
                            spread=spread,
                            match_title=mapping.poly_market_title or match_label,
                        )

                        # ── TG: Signal detected ───────────────────────
                        try:
                            await self.engine.tg.send(
                                f"🏏 <b>Cricket Signal</b>\n"
                                f"Event: {decision.event}\n"
                                f"Regime: {decision.regime}\n"
                                f"Dir: {direction}\n"
                                f"Price: {market_price:.3f}\n"
                                f"Reason: {decision.reason}\n"
                                f"Match: {match_label}"
                            )
                        except Exception:
                            pass

                        # ── TG: Paper entry ───────────────────────────
                        try:
                            await self.engine.tg.send(
                                f"📥 <b>Paper Entry</b>\n"
                                f"Dir: {direction}\n"
                                f"Entry: {market_price:.4f}\n"
                                f"Match: {match_label}"
                            )
                        except Exception:
                            pass

                        # NO LIVE EXECUTION — paper only

                    # ── Exit manager checks ───────────────────────────
                    self.cricket_exit_mgr.check_all(
                        books=self.poly_feed.books,
                        match_states={},
                    )

                    # ── v2.1: Debug counters every 60s ────────────────
                    now = time.time()
                    if now - last_diag_log >= 60:
                        last_diag_log = now
                        sig_counts = {}
                        for fid_d in fixture_ids:
                            sig_counts[fid_d] = self.sm_cricket_state.get_event_counts(fid_d)
                        open_trades = len(self.cricket_exit_mgr.open_trades)
                        closed_trades = len(self.cricket_exit_mgr.closed_trades)
                        log.info(
                            "CRICKET_DEBUG | SIGNALS FLOWING | "
                            "events=%s | open=%d | closed=%d | "
                            "feeds=%s",
                            sig_counts, open_trades, closed_trades,
                            self.sm_cricket_feed.stats_line,
                        )

                except Exception as e:
                    log.error("cricket SM signal loop error: %s", e)

                await asyncio.sleep(CRICKET_SM_POLL_S)

    async def _cricket_signal_loop(self):
        """Cricket signal processing loop (v4.9).

        Two signal paths:
          1. DLS-based signals — when ESPN state is available
          2. Fallback price-based signals — when ESPN mapping fails
        """
        last_health_log = time.time()
        last_map_attempt: dict[str, float] = {}  # match_id → last attempt ts
        MAP_RETRY_S = 120.0  # retry mapping every 2 min
        # v8.7: persistent mapping cache (ESPN ID → match state source)
        cricket_espn_map: dict[str, str] = {}
        # v8.7: persistent mapping source tracker
        cricket_map_source: dict[str, str] = {}  # match_id → "LIVE" | "ESPN"

        while not self._shutdown:
            try:
                for match_id, link in list(self.cricket_links.items()):
                    # ═══════════════════════════════════════════════════
                    #  v8.7: Two-pass mapping (LIVE primary, ESPN fallback)
                    #  Persistent: once mapped, reuse until invalid
                    # ═══════════════════════════════════════════════════
                    espn_id = cricket_espn_map.get(match_id)
                    state = None

                    # ── Reuse existing mapping if valid ────────────
                    if espn_id:
                        state = self.cricket_feed.games.get(espn_id)
                        if state:
                            pass  # mapping still valid, use it
                        else:
                            # mapping went stale — clear it for re-mapping
                            del cricket_espn_map[match_id]
                            cricket_map_source.pop(match_id, None)
                            espn_id = None

                    # ── Mapping needed — run two-pass ────────────
                    if not state:
                        now_map = time.time()
                        last_try = last_map_attempt.get(match_id, 0)
                        if now_map - last_try < MAP_RETRY_S and last_try > 0:
                            pass  # rate-limited, skip
                        else:
                            last_map_attempt[match_id] = now_map
                            poly_a, poly_b = self._parse_cricket_teams(
                                link.polymarket_title
                            )
                            if poly_a and poly_b:
                                live_candidates = []
                                # ── PASS 1: CRICKET_LIVE feed (PRIMARY) ──
                                for eid, cs in self.cricket_feed.games.items():
                                    if not cs.team_a or not cs.team_b:
                                        continue
                                    live_candidates.append(
                                        f"{cs.team_a} vs {cs.team_b}"
                                    )
                                    a_match = (
                                        self._teams_match(poly_a, cs.team_a) or
                                        self._teams_match(poly_a, cs.team_b)
                                    )
                                    b_match = (
                                        self._teams_match(poly_b, cs.team_a) or
                                        self._teams_match(poly_b, cs.team_b)
                                    )
                                    if a_match and b_match:
                                        espn_id = eid
                                        state = cs
                                        cricket_espn_map[match_id] = espn_id
                                        cricket_map_source[match_id] = "LIVE"
                                        log.info(
                                            'CRICKET_MAP_SUCCESS | source=LIVE '
                                            '| poly="%s" | espn="%s" '
                                            '| teams=%s vs %s',
                                            link.polymarket_title[:60],
                                            espn_id, cs.team_a, cs.team_b,
                                        )
                                        break

                                if not state:
                                    log.info(
                                        'CRICKET_MAP_FAIL | source=BOTH '
                                        '| poly="%s" | parsed=[%s vs %s] '
                                        '| live_candidates=%s',
                                        link.polymarket_title[:60],
                                        poly_a, poly_b,
                                        live_candidates[:5],
                                    )
                            else:
                                log.info(
                                    'CRICKET_MAP_FAIL | poly="%s" '
                                    '| parse_failed=True',
                                    link.polymarket_title[:60],
                                )

                    # ── Get book data (try all token IDs) ─────────
                    book = None
                    used_token_id = link.home_token_id
                    # Try home token first, then all others
                    candidate_ids = [link.home_token_id] + [
                        t for t in (link.all_token_ids or [])
                        if t != link.home_token_id
                    ]
                    for tid in candidate_ids:
                        if not tid:
                            continue
                        b = self.poly_feed.books.get(tid)
                        if b and b.mid > 0:
                            book = b
                            used_token_id = tid
                            break

                    if not book or book.mid <= 0:
                        # Periodic log: which cricket markets have no book
                        if not hasattr(self, '_cricket_nobook_log'):
                            self._cricket_nobook_log = {}
                        now_t = time.time()
                        if now_t - self._cricket_nobook_log.get(match_id, 0) > 120:
                            self._cricket_nobook_log[match_id] = now_t
                            log.info(
                                "CRICKET_NO_BOOK | %s | tokens=%s | %s",
                                match_id,
                                [t[:8] + '...' if t and len(t) > 8 else t for t in candidate_ids[:3]],
                                link.polymarket_title[:50],
                            )
                        continue

                    # ── v7.0: Feed health monitor ─────────────────
                    self.cricket_health.tick(
                        match_id=match_id,
                        mid=book.mid,
                        spread=book.spread,
                        bid=book.best_bid,
                        ask=book.best_ask,
                        ts=book.timestamp,
                        market_title=link.polymarket_title,
                    )

                    # ── v7.1: Simple activation (DEAD → ACTIVE) ──
                    rt_state = self._cricket_runtime_state.get(match_id, "ACTIVE")
                    spread = book.spread
                    if rt_state == "DEAD":
                        if spread <= 0.12 and book.best_bid > 0.05 and book.best_ask < 0.95:
                            self._cricket_runtime_state[match_id] = "ACTIVE"
                            rt_state = "ACTIVE"
                            log.info(
                                "CRICKET_ACTIVATED | %s | spread=%.4f | bid=%.2f ask=%.2f",
                                link.polymarket_title[:60], spread,
                                book.best_bid, book.best_ask,
                            )

                    # ── v7.1: State visibility (throttled 60s) ────
                    now_st = time.time()
                    if now_st - self._cricket_state_log_ts.get(match_id, 0) > 60:
                        self._cricket_state_log_ts[match_id] = now_st
                        log.info(
                            "CRICKET_STATE | %s | state=%s | spread=%.4f | bid=%.2f ask=%.2f",
                            link.polymarket_title[:60], rt_state, spread,
                            book.best_bid, book.best_ask,
                        )

                    # ── v7.1: Gate — only evaluate signals for ACTIVE markets ──
                    if rt_state != "ACTIVE":
                        continue

                    market_price = book.mid

                    # ════════════════════════════════════════════════
                    #  PATH 1: DLS-based signals (ESPN state available)
                    # ════════════════════════════════════════════════
                    if state:
                        signals = self.cricket_strategy.evaluate(
                            state, market_price
                        )
                        for sig in signals:
                            self.cricket_logger.log_signal(sig)
                            decision = self.cricket_guard.can_execute(
                                sig,
                                spread=book.spread,
                                data_age_s=time.time() - book.timestamp,
                            )
                            if decision.can_execute:
                                # ═══ GLOBAL EDGE GUARD (v6.0) ═══
                                can_exec, block_reason = validate_trade_execution(
                                    edge=sig.edge,
                                    price=market_price,
                                    sport="cricket",
                                    context=f"{sig.signal_type} | {link.polymarket_title[:50]}",
                                    strategy=STRAT_CRICKET_MOM,
                                )
                                if not can_exec:
                                    continue

                                self.cricket_guard.record_entry(match_id)
                                score_str = str(state)
                                log.info(
                                    "CRICKET PAPER ENTRY | %s | "
                                    "edge=%.4f | mkt=%.4f | %s",
                                    sig.signal_type, sig.edge,
                                    market_price,
                                    link.polymarket_title[:40],
                                )
                                token_id = link.home_token_id or (
                                    link.all_token_ids[0]
                                    if link.all_token_ids else ""
                                )
                                self.cricket_exit_mgr.register_trade(
                                    match_id=match_id,
                                    selection_id=token_id,
                                    signal_type=sig.signal_type,
                                    entry_price=market_price,
                                    fair_value=sig.fair_price,
                                    edge=sig.edge,
                                    entry_score=score_str,
                                    spread=book.spread,
                                    match_title=link.polymarket_title,
                                )
                                self.cricket_logger.log_state(
                                    state, market_price
                                )
                                try:
                                    await self.engine.tg.send(
                                        f"🏏 <b>Cricket Signal (PAPER)</b>\n"
                                        f"Type: {sig.signal_type}\n"
                                        f"Edge: {sig.edge:+.4f}\n"
                                        f"Mkt: {market_price:.3f} | "
                                        f"Fair: {sig.fair_price:.3f}\n"
                                        f"Match: {link.polymarket_title}\n"
                                        f"Score: {state}"
                                    )
                                except Exception:
                                    pass

                                # ═══ CRICKET LIVE BUY (DLS path) ═══
                                if self.cricket_live and self.cricket_live.is_ready:
                                    log.info(
                                        "CRICKET_EVENT_TRIGGER | path=DLS | %s | "
                                        "edge=%.4f | mkt=%.4f",
                                        sig.signal_type, sig.edge, market_price,
                                    )
                                    log.info(
                                        "CRICKET_REGIME | spread=%.4f | age=%.1fs "
                                        "| state=%s",
                                        book.spread,
                                        time.time() - book.timestamp,
                                        rt_state,
                                    )
                                    log.info(
                                        "CRICKET_DECISION | action=LIVE_BUY "
                                        "| direction=LONG | mid=%.4f",
                                        market_price,
                                    )
                                    live_result = self.cricket_live.place_order(
                                        token_id=token_id,
                                        mid=market_price,
                                        direction="LONG",
                                        match_id=match_id,
                                        match_info=f"DLS {sig.signal_type} | {link.polymarket_title[:40]}",
                                        regime=CricketLiveExecutor.classify_regime(state),
                                    )
                                    if live_result.success:
                                        try:
                                            await self.engine.tg.send(
                                                f"🟢 <b>Cricket LIVE BUY</b>\n"
                                                f"Type: {sig.signal_type}\n"
                                                f"Limit: {live_result.avg_price:.4f} "
                                                f"(mid={market_price:.4f})\n"
                                                f"Size: ${live_result.filled_size:.2f}\n"
                                                f"Order: {live_result.order_id}\n"
                                                f"Match: {link.polymarket_title[:40]}"
                                            )
                                        except Exception:
                                            pass

                        # DLS path done — but ALWAYS fall through to tick detector

                    # ════════════════════════════════════════════════
                    #  PATH 2: Tick-based continuation
                    #  v8.7: Safety guard — block blind trading
                    # ════════════════════════════════════════════════
                    if state is None:
                        # Observe-only: run detection for diagnostics
                        # but do NOT create entries or pullbacks
                        self.cricket_tick_detector.on_tick(
                            match_id=match_id,
                            mid=book.mid,
                            spread=book.spread,
                            timestamp=book.timestamp,
                            market_title=link.polymarket_title,
                            match_state=None,
                            observe_only=True,
                        )
                        continue

                    tick_signal = self.cricket_tick_detector.on_tick(
                        match_id=match_id,
                        mid=book.mid,
                        spread=book.spread,
                        timestamp=book.timestamp,
                        market_title=link.polymarket_title,
                        match_state=state,
                    )

                    if tick_signal:
                        # ═══ GLOBAL EDGE GUARD (v6.0) ═══
                        can_exec, block_reason = validate_trade_execution(
                            edge=tick_signal.edge,
                            price=tick_signal.entry_price,
                            sport="cricket_tick",
                            context=f"{tick_signal.signal_type} {tick_signal.direction} | {link.polymarket_title[:50]}",
                            strategy=STRAT_CRICKET_MOM,
                        )
                        if not can_exec:
                            continue

                        # Register with the existing cricket exit manager
                        token_id = link.home_token_id or (
                            link.all_token_ids[0]
                            if link.all_token_ids else ""
                        )
                        self.cricket_exit_mgr.register_trade(
                            match_id=match_id,
                            selection_id=token_id,
                            signal_type=tick_signal.signal_type,
                            entry_price=tick_signal.entry_price,
                            fair_value=tick_signal.fair_price,
                            edge=tick_signal.edge,
                            entry_score=f"{tick_signal.direction} {tick_signal.signal_type}",
                            spread=tick_signal.spread,
                            match_title=link.polymarket_title,
                        )
                        # Also register in the tick detector for its own exit tracking
                        self.cricket_tick_detector.register_trade(tick_signal)
                        self.cricket_guard.record_entry(match_id)

                        log.info(
                            "CRICKET PAPER ENTRY | %s | "
                            "direction=%s | edge=%.4f | "
                            "mkt=%.4f | move=%.4f | %s",
                            tick_signal.signal_type,
                            tick_signal.direction,
                            tick_signal.edge,
                            tick_signal.entry_price,
                            tick_signal.move,
                            link.polymarket_title[:40],
                        )
                        try:
                            await self.engine.tg.send(
                                f"🏏 <b>Cricket Tick Signal (PAPER)</b>\n"
                                f"Type: {tick_signal.signal_type}\n"
                                f"Dir: {tick_signal.direction}\n"
                                f"Edge: {tick_signal.edge:+.4f} | "
                                f"Move: {tick_signal.move:.4f}\n"
                                f"Mkt: {tick_signal.entry_price:.3f}\n"
                                f"Match: {link.polymarket_title}"
                            )
                        except Exception:
                            pass

                        # ═══ CRICKET LIVE BUY (tick path) ═══
                        if self.cricket_live and self.cricket_live.is_ready:
                            log.info(
                                "CRICKET_EVENT_TRIGGER | path=TICK | %s %s | "
                                "edge=%.4f | mkt=%.4f | move=%.4f",
                                tick_signal.signal_type, tick_signal.direction,
                                tick_signal.edge, tick_signal.entry_price,
                                tick_signal.move,
                            )
                            log.info(
                                "CRICKET_REGIME | spread=%.4f | age=%.1fs "
                                "| state=%s",
                                tick_signal.spread,
                                time.time() - book.timestamp,
                                rt_state,
                            )
                            log.info(
                                "CRICKET_DECISION | action=LIVE_BUY "
                                "| direction=%s | mid=%.4f",
                                tick_signal.direction,
                                tick_signal.entry_price,
                            )
                            live_result = self.cricket_live.place_order(
                                token_id=token_id,
                                mid=tick_signal.entry_price,
                                direction=tick_signal.direction,
                                match_id=match_id,
                                match_info=f"TICK {tick_signal.signal_type} {tick_signal.direction} | {link.polymarket_title[:40]}",
                                regime=CricketLiveExecutor.classify_regime(state),
                            )
                            if live_result.success:
                                try:
                                    await self.engine.tg.send(
                                        f"🟢 <b>Cricket LIVE {tick_signal.direction}</b>\n"
                                        f"Type: {tick_signal.signal_type}\n"
                                        f"Limit: {live_result.avg_price:.4f} "
                                        f"(mid={tick_signal.entry_price:.4f})\n"
                                        f"Size: ${live_result.filled_size:.2f}\n"
                                        f"Order: {live_result.order_id}\n"
                                        f"Match: {link.polymarket_title[:40]}"
                                    )
                                except Exception:
                                    pass

                # ── Tick detector exit checks ─────────────────────────
                def _get_cricket_price(mid):
                    lnk = self.cricket_links.get(mid)
                    if not lnk:
                        return None
                    tid = lnk.home_token_id or (
                        lnk.all_token_ids[0] if lnk.all_token_ids else ""
                    )
                    bk = self.poly_feed.books.get(tid)
                    if not bk or bk.mid <= 0:
                        return None
                    return (bk.mid, bk.spread)

                tick_exits = self.cricket_tick_detector.check_exits(_get_cricket_price)
                for (exit_mid, exit_sig_type, exit_entry, exit_price, exit_reason) in tick_exits:
                    pnl = exit_price - exit_entry
                    r_mult = pnl / 0.06 if 0.06 > 0 else 0  # SL = 0.06
                    log.info(
                        "CRICKET_TICK_CLOSED | %s | %s | "
                        "entry=%.4f exit=%.4f pnl=%+.4f R=%+.3f",
                        exit_sig_type, exit_reason,
                        exit_entry, exit_price, pnl, r_mult,
                    )
                    try:
                        pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                        await self.engine.tg.send(
                            f"{pnl_emoji} <b>Cricket Trade Closed</b>\n"
                            f"Type: {exit_sig_type}\n"
                            f"Exit: {exit_reason}\n"
                            f"Entry: {exit_entry:.3f} → Exit: {exit_price:.3f}\n"
                            f"PnL: {pnl:+.4f} | R: {r_mult:+.2f}"
                        )
                    except Exception:
                        pass

                # Check open paper trades for MAE/MFE updates + exits
                self.cricket_exit_mgr.check_all(
                    books=self.poly_feed.books,
                    match_states={
                        mid: self.cricket_feed.games.get(
                            cricket_espn_map.get(mid, "")
                        )
                        for mid in self.cricket_exit_mgr.open_trades
                    },
                )

                # ── Pre-fill validation (v1.1 safety patch) ──────────
                if self.cricket_live and self.cricket_live.is_ready:
                    # Derive current regime from any active match state
                    current_regime = ""
                    for _mid, _lnk in self.cricket_links.items():
                        _espn = cricket_espn_map.get(_mid, "")
                        _st = self.cricket_feed.games.get(_espn) if _espn else None
                        if _st:
                            current_regime = CricketLiveExecutor.classify_regime(_st)
                            break

                    invalidated = self.cricket_live.validate_pending_fills(
                        books=self.poly_feed.books,
                        regime=current_regime,
                    )
                    for oid in invalidated:
                        log.info("CRICKET_EXECUTION | PRE_FILL_REJECTED | order=%s", oid)

                # ── GTC order timeout management ────────────────────
                if self.cricket_live and self.cricket_live.is_ready:
                    cancelled = self.cricket_live.check_pending_orders()
                    for oid in cancelled:
                        log.info("CRICKET_EXECUTION | GTC_EXPIRED | order=%s", oid)

                # Hourly health log
                now = time.time()
                if now - last_health_log >= 3600:
                    self.cricket_guard.stats.log_summary()
                    log.info(
                        "CRICKET EXIT SUMMARY: %s",
                        self.cricket_exit_mgr.summary(),
                    )
                    last_health_log = now

                # v7.0: Per-market health diagnostics (every 60s)
                titles = {mid: lnk.polymarket_title
                          for mid, lnk in self.cricket_links.items()}
                self.cricket_health.log_health(market_titles=titles)

            except Exception as e:
                log.error("cricket signal loop error: %s", e)

            await asyncio.sleep(POLYMARKET_SNAPSHOT_S)

    async def _cricket_readiness_loop(self):
        """Cricket readiness monitoring (v7.0).

        Periodically checks all cricket markets for tradability.
        Sends Telegram alerts at regular intervals and on status changes.
        """
        await asyncio.sleep(30)  # wait for initial discovery + first ticks
        log.info("Cricket readiness loop started")

        while not self._shutdown:
            try:
                for match_id, link in list(self.cricket_links.items()):
                    # Determine spread phase
                    # Try to parse match start time from endDate
                    match_start_ts = 0.0
                    if link.polymarket_slug:
                        # For now, use current time as reference (pre-match = PRE)
                        # Markets are discovered when active, so default LIVE
                        pass

                    # Default to LIVE phase (most conservative threshold)
                    phase = SpreadPhase.LIVE

                    result = check_cricket_readiness(
                        match_id=match_id,
                        token_ids=link.all_token_ids,
                        books=self.poly_feed.books,
                        health_monitor=self.cricket_health,
                        phase=phase,
                    )

                    new_status = result.status.value  # "READY" or "NOT_READY"
                    old_status = self._cricket_readiness_status.get(match_id, "")

                    # Log readiness result
                    if result.status == ReadinessStatus.READY:
                        log.info(
                            "CRICKET_READY | %s | spread=%.4f | "
                            "tick_rate=%.0f | price_range=%.4f | "
                            "last_tick_age=%.0fs",
                            link.polymarket_title[:50],
                            result.spread, result.tick_rate,
                            result.price_range, result.last_tick_age,
                        )
                    else:
                        log.info(
                            "CRICKET_NOT_READY | %s | reason=%s | "
                            "issues=%s | spread=%.4f | tick_rate=%.0f | "
                            "price_range=%.4f | last_tick_age=%.0fs",
                            link.polymarket_title[:50],
                            result.reason.value if result.reason else "UNKNOWN",
                            result.issues,
                            result.spread, result.tick_rate,
                            result.price_range, result.last_tick_age,
                        )

                    # Status change detection (Part 5: immediate alert)
                    if old_status and old_status != new_status:
                        log.warning(
                            "CRICKET_STATUS_CHANGE | %s | %s → %s",
                            link.polymarket_title[:50], old_status, new_status,
                        )
                        try:
                            await self.engine.tg.notify_cricket_status_change(
                                match_name=link.polymarket_title,
                                old_status=old_status,
                                new_status=new_status,
                                spread=result.spread,
                                tick_rate=result.tick_rate,
                                price_range=result.price_range,
                            )
                        except Exception:
                            pass

                        # ── v2.0: TG alert on late activation ──────
                        if (new_status == "READY" and
                                self.cricket_health.was_late_activated(match_id)):
                            try:
                                await self.engine.tg.send(
                                    f"🏏 <b>CRICKET LIVE READY</b>\n"
                                    f"Match: {link.polymarket_title}\n"
                                    f"Spread: {result.spread:.4f}\n"
                                    f"Ticks/min: {result.tick_rate:.0f}\n"
                                    f"Status: ACTIVE (late-activated)"
                                )
                            except Exception:
                                pass

                    elif not old_status:
                        # First check — send initial status
                        try:
                            if result.status == ReadinessStatus.READY:
                                await self.engine.tg.notify_cricket_ready(
                                    match_name=link.polymarket_title,
                                    spread=result.spread,
                                    tick_rate=result.tick_rate,
                                    price_range=result.price_range,
                                )
                            else:
                                await self.engine.tg.notify_cricket_not_ready(
                                    match_name=link.polymarket_title,
                                    reason=result.reason.value if result.reason else "UNKNOWN",
                                    issues=result.issues,
                                    spread=result.spread,
                                    tick_rate=result.tick_rate,
                                    price_range=result.price_range,
                                )
                        except Exception:
                            pass

                    self._cricket_readiness_status[match_id] = new_status

            except Exception as e:
                log.error("cricket readiness loop error: %s", e)

            await asyncio.sleep(CRICKET_READINESS_CHECK_INTERVAL_S)

    async def _status_printer_loop(self):
        """Periodically print system status + Telegram updates."""
        tg_interval = 0  # send Telegram every 15th iteration (15 min)
        while not self._shutdown:
            try:
                live_football = sum(
                    1 for g in self.football_feed.games.values() if g.is_live
                )
                live_nba = sum(
                    1 for g in self.nba_feed.games.values() if g.is_live
                )
                live_tennis = len(self.tennis_links)
                live_cricket = sum(
                    1 for g in self.cricket_feed.games.values() if g.is_live
                )

                summary = self.engine.get_summary()

                log.info(
                    "STATUS | Football: %d live | NBA: %d live | Tennis: %d mkts | Cricket: %d live | "
                    "WS: %s (msgs=%d) | Links: %d | "
                    "Trades: %d (wins=%d, PnL=$%.2f)",
                    live_football, live_nba, live_tennis, live_cricket,
                    "OK" if self.poly_feed.is_connected else "DOWN",
                    self.poly_feed.message_count,
                    len(self.links) + len(self.tennis_links) + len(self.cricket_links),
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
                if tg_interval >= 15:
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
        log.info("  SPORTS MARKET SYSTEM STARTING [v10.2]")
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
        log.info("STARTUP_TG_SENDING | markets=%d links=%d tokens=%d",
                 len(self.markets), len(self.links), len(all_tokens))
        try:
            await self.engine.tg.notify_startup(
                len(self.markets), len(self.links), len(all_tokens)
            )
            log.info("STARTUP_TG_SUCCESS")
        except Exception as e:
            log.error("STARTUP_TG_FAIL | %s", e)
            # Retry once after short delay (session warmup)
            try:
                await asyncio.sleep(2)
                await self.engine.tg.notify_startup(
                    len(self.markets), len(self.links), len(all_tokens)
                )
                log.info("STARTUP_TG_RETRY_SUCCESS")
            except Exception as e2:
                log.error("STARTUP_TG_RETRY_FAIL | %s", e2)

        # v9.0 / v10: Wire circuit breaker → Telegram notification callback
        import asyncio as _asyncio
        _tg_ref = self.engine.tg
        def _cb_telegram_callback(sport, strategy, streak, reset=False):
            if reset:
                _asyncio.ensure_future(_tg_ref.notify_circuit_breaker_reset(sport, strategy))
            else:
                _asyncio.ensure_future(_tg_ref.notify_circuit_breaker(sport, strategy, streak))
        circuit_breaker.set_telegram_callback(_cb_telegram_callback)
        log.info("CB_TELEGRAM_CALLBACK_WIRED")

        # Phase 4: Start all async loops
        # v2.0: Football/NBA DISABLED — cricket paper-only mode
        tasks = [
            asyncio.create_task(self.poly_feed.run(), name="polymarket_ws"),
            asyncio.create_task(self.poly_feed.run_book_polling(), name="book_rest_polling"),
            # asyncio.create_task(self._score_polling_loop(), name="score_polling"),       # v2.0: OFF (football/NBA)
            # asyncio.create_task(self._signal_processing_loop(), name="signal_processing"), # v2.0: OFF (football/NBA)
            asyncio.create_task(self._status_printer_loop(), name="status_printer"),
            asyncio.create_task(self._rematching_loop(), name="rematching"),
            # Tennis — score polling + signal processing (UNTOUCHED)
            asyncio.create_task(self._tennis_score_polling_loop(), name="tennis_scores"),
            asyncio.create_task(self._tennis_signal_loop(), name="tennis_signals"),
            # Cricket v2.0 — Sportmonks paper-only signal loop
            asyncio.create_task(self._cricket_sm_signal_loop(), name="cricket_sm_signals"),
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
        self.cricket_logger.close()

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
        await orch.ncaa_feed.fetch_live_scores(session)
        orch.nba_feed.games.update(orch.ncaa_feed.games)
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
