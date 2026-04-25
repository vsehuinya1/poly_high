"""
Sports system configuration — loaded from .env.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Telegram ─────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── ESPN Football ────────────────────────────────────────────────────
ESPN_FOOTBALL_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

ESPN_LEAGUES = [
    "eng.1",            # EPL
    "eng.2",            # Championship
    "esp.1",            # La Liga
    "ger.1",            # Bundesliga
    "ita.1",            # Serie A
    "fra.1",            # Ligue 1
    "ned.1",            # Eredivisie
    "uefa.champions",   # UCL
    "uefa.europa",      # Europa League
    "uefa.europa.conf", # Conference League
]

# ── Football Competition Whitelist (v4.8) ────────────────────────────
# Only these competitions are allowed for trading.
# Maps to the 'league' field returned by classify_market() in discovery.py.
FOOTBALL_ALLOWED_COMPETITIONS = [
    "Premier League", "EPL",
    "La Liga",
    "Bundesliga",
    "Ligue 1",
    "Serie A",
    "Eredivisie",
    "Championship",
    "Champions League",
    "Europa League",
    "Conference League",
    "FIFA World Cup",
    "World Cup Qualifiers",
    "Euro Qualifiers",
    "AFCON Qualifiers",
    "International",
]

# ── NBA ──────────────────────────────────────────────────────────────
NBA_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
NCAA_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?limit=100"
NBA_BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"

# ── Polymarket ───────────────────────────────────────────────────────
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_API_URL = "https://gamma-api.polymarket.com"

# ── Polling Intervals ────────────────────────────────────────────────
SCORE_POLL_INTERVAL_S = 30       # poll live scores every 30s
POLYMARKET_SNAPSHOT_S = 5        # record Polymarket book state every 5s
DISCOVERY_INTERVAL_S  = 300      # re-scan for new markets every 5 min

# ── REST Polling Fallback (BBO freshness) ────────────────────────
BOOK_REST_ENABLED       = True   # enable REST polling for stale books
BOOK_REST_POLL_INTERVAL_S = 12   # seconds between polls per token (staggered)
BOOK_REST_STALE_LOG_S   = 60     # log warning when token is stale for this long
POLYMARKET_CLOB_BOOK_URL = "https://clob.polymarket.com/book"

# ── Paper Trading Parameters ─────────────────────────────────────────
ENTRY_EDGE_THRESHOLD  = 0.05     # min 5 cent edge to enter (base — overridden per sport)
# EXIT_CONVERGENCE    = 0.01     # v5.0: REMOVED — 33% WR, −2.65 ΣR (never import this)
MAX_POSITION_PER_MARKET = 500.0  # max $500 per market
MAX_CONCURRENT_POSITIONS = 10
MAX_DAILY_LOSS        = 1000.0   # kill-switch

# ── Execution Hygiene (v3.4 — controlled participation) ────────────
# Hard entry filters — sweet-spot zone
PRICE_BAND_LO         = 0.45     # legacy — used as fallback
PRICE_BAND_HI         = 0.65     # legacy — used as fallback
# Direction-specific price bands (v3.8)
SELL_PRICE_BAND_LO    = 0.55     # SELL: market price must be >= this
SELL_PRICE_BAND_HI    = 0.90     # SELL: market price must be <= this
BUY_PRICE_BAND_LO     = 0.10     # BUY: market price must be >= this
BUY_PRICE_BAND_HI     = 0.45     # BUY: market price must be <= this
MAX_SPREAD            = 0.06     # spread must be <= this
MAX_BOOK_AGE_S        = 20.0     # book_age must be <= this
MAX_SCORE_DIFF        = 15       # |home - away| must be <= this
EDGE_TRADE_THRESHOLD  = 0.10     # min edge to open a trade (base — overridden per sport)
FOOTBALL_EDGE_TRADE   = 0.15     # v4.5: higher entry edge for football (was 0.10)
NBA_EDGE_TRADE        = 0.10     # v4.5: NBA keeps 0.10
NBA_DISABLED          = True     # v4.9: disabled — no paper trades, no notifications

# ── Entry Timing Engine (v4.6) ──────────────────────────────────
ENTRY_DELAY_NBA_S     = 120      # NBA/NCAA: 120s before entry
ENTRY_DELAY_FB_S      = 90       # Football: 90s before entry
ENTRY_DELAY_DEFAULT_S = 30       # Tennis/other: 30s before entry
ENTRY_PERSIST_EDGE    = 0.12     # edge must stay above this during delay
ENTRY_PERSIST_TICKS   = 3        # must re-confirm edge N times during delay
ENTRY_MAX_DELAY_S     = 300      # discard pending signal if not filled by 5min
MAX_ELAPSED_PCT       = 0.75     # block entries past 75% of game
LATE_GAME_HARD_STOP_NBA = 36.0   # absolute minute cutoff for NBA entries
LATE_GAME_HARD_STOP_FB  = 67.0   # absolute minute cutoff for football entries
NBA_TRADE_WINDOW_START = 720     # adj_sec >= this (36min elapsed → 12min remaining)
NBA_TRADE_WINDOW_END   = 1800    # adj_sec < this  (18min elapsed → 30min remaining)
FB_TRADE_WINDOW_START  = 1200    # adj_sec >= this (70min elapsed → 20min remaining)
FB_TRADE_WINDOW_END    = 4500    # adj_sec < this  (15min elapsed → 75min remaining)
# Position control
MAX_POS_PER_DIRECTION  = 1        # max 1 open position per direction per game
SELL_ONLY_MODE         = True     # v3.5: disable BUY entries (25% WR vs SELL 50%)

# Game activation gate
GATE_FRESH_THRESHOLD  = 30.0     # book_age <= this to count as "fresh" for gate
GATE_STREAK_S         = 60.0     # required continuous fresh streak (seconds)
GATE_ROLLING_WINDOW_S = 300.0    # 5-minute rolling window
GATE_ROLLING_FRESH_PCT = 0.05    # 5% of ticks in window must be fresh (age<=20s)

# Intra-game freeze
FREEZE_STALE_THRESHOLD = 60.0    # book_age > this triggers freeze counter
FREEZE_STALE_DURATION_S = 90.0   # must be stale for this long to freeze
UNFREEZE_STREAK_S     = 60.0     # fresh streak needed to unfreeze

# Cooldown and per-game stop
COOLDOWN_S            = 300.0    # 5-minute cooldown per game after exit
PER_GAME_STOP         = 200.0    # max loss per game before stopping

# ── Execution Stability (v4.0 — stability patch) ────────────────────
MIN_HOLD_S            = 5        # Patch 1: base min hold (overridden per sport)
NBA_MIN_HOLD_S        = 90       # v4.5: NBA min 90s hold — stops instant edge_flip churn
FOOTBALL_MIN_HOLD_S   = 30       # v4.5: football 30s min hold
EDGE_CONFIRM_TICKS    = 3        # Patch 2: edge must persist 3 consecutive ticks
MAX_TRADES_PER_GAME   = 20       # Patch 3: cap total entries per game
POST_EXIT_COOLDOWN_S  = 30       # Patch 4: game-level cooldown after any exit
STOP_LOSS_TICKS       = 6        # Patch 5: hard stop at 6 ticks (0.06) adverse
EDGE_FLIP_THRESHOLD   = 0.03     # Patch 6: edge reversal must exceed this to exit (base)
NBA_EDGE_FLIP_THRESHOLD = 0.05   # v4.5: NBA needs bigger reversal to exit (was 0.03)
ENTRY_MAX_SPREAD      = 0.03     # Patch 7: max 3-tick spread at execution moment
ENTRY_MAX_BOOK_AGE_S  = 3.0      # Patch 7: book must be <3s old at execution

# ── Football Risk Controls (v4.3 — MAE-based) ───────────────────────
FOOTBALL_STOP_LOSS_TICKS = 12     # MAE-based statistical stop (median MAE ≈ 11.25)
FOOTBALL_FAST_MOVE_TICKS = 3      # early momentum exit threshold (ticks)
FOOTBALL_FAST_MOVE_S     = 300    # momentum window (5 minutes)
FOOTBALL_TIMEOUT_S       = 600    # v4.5: reduced timeout (10 min, was 15 min)
NBA_TIMEOUT_S            = 1200   # v4.5: NBA timeout (20 min, was 30 min)
DEFAULT_TIMEOUT_S        = 1800   # fallback timeout (30 min)

# v4.5: Time-based entry blocks
NBA_Q1_BLOCK             = True   # block NBA entries in Q1 entirely
NBA_QUARTER_END_BLOCK_S  = 120    # block last 2 min of each NBA quarter
FB_HALFTIME_BLOCK_START  = 40     # block football entries minute 40-50 (half-time zone)
FB_HALFTIME_BLOCK_END    = 50

# ── Tennis (Strategy B — Inflection Sniping) ─────────────────────────
TENNIS_SERVE_WIN_P    = 0.64      # ATP average service point win rate
TENNIS_PANIC_EDGE     = 0.06      # min edge for panic discount trigger
TENNIS_REVERSION_EDGE = 0.08      # min edge for set mean reversion trigger (raised from 0.05 — sub-0.08 edges produce near-zero R)
TENNIS_PRICE_CAP      = 0.85      # no entries above this market price
TENNIS_STALENESS_S    = 30.0      # v4.6.2: was 3.0 — too tight, killed ALL signals via stale cascade
TENNIS_COOLDOWN_S     = 120.0     # cooldown after position exit (seconds)
TENNIS_FEED_POLL_S    = 3.0       # feed poll interval (seconds)
TENNIS_FEED_STALL_S   = 60.0      # feed stall detection threshold
TENNIS_FEED_HEALTH_S  = 60.0      # feed health log interval

# Tennis execution hardening (v2.0)
TENNIS_PRICE_FLOOR         = 0.20   # v4.9.2: raised from 0.15 — align with exit floor strategy
TENNIS_PRICE_FLOOR_BYPASS  = 0.15   # v4.9.2: high-edge trades bypass floor if max_edge >= this

# Tennis Entry Timing — Lightweight (v4.6)
TENNIS_ENTRY_DELAY_S       = 15     # v9.8A: reduced from 30 — improves fill rate, reduces restart exposure
TENNIS_ENTRY_CONFIRM_TICKS = 2      # 2 confirmations required
TENNIS_ENTRY_MIN_EDGE      = 0.06   # edge must persist above this
TENNIS_EDGE_DECAY_THRESH   = 0.30   # cancel if edge drops >30% from initial
TENNIS_STALE_OVERRIDE_EDGE = 0.18   # v9.9: bypass NO_PRICE_MOVEMENT if edge >= this
TENNIS_ENTRY_MAX_DELAY_S   = 90     # discard if not filled by 90s
TENNIS_MAX_SIGNALS_HR      = 10     # max signals per match per rolling hour
TENNIS_STALE_DISABLE_COUNT = 20     # v4.6.2: was 5 — too aggressive, disabled matches permanently
TENNIS_STALE_DISABLE_S     = 60     # v4.6.2: was 300 — 1min disable instead of 5min

# ── Cricket (Live + Paper Mode) ──────────────────────────────────────
CRICKET_PAPER_ONLY         = False  # live execution enabled alongside paper
CRICKET_TRADE_SIZE         = 200.0  # paper trade size ($)
CRICKET_MAX_SPREAD         = 0.02   # abort if spread > this
CRICKET_MOMENTUM_RR_THRESH = 2.0    # rolling RR must exceed RRR by this
CRICKET_MOMENTUM_EDGE      = 0.08   # min edge for momentum signal
CRICKET_WICKET_EDGE        = 0.10   # min edge for wicket overreaction
CRICKET_LATENCY_THRESH_MS  = 2000.0 # min latency for snipe logging
CRICKET_COOLDOWN_S         = 120.0  # cooldown between trades

# ── Cricket Live Execution ───────────────────────────────────────────
CRICKET_LIVE_MODE      = os.getenv("CRICKET_LIVE_MODE", "true").lower() == "true"
CRICKET_BANKROLL       = float(os.getenv("CRICKET_BANKROLL", "50.0"))
CRICKET_KELLY_PCT      = float(os.getenv("CRICKET_KELLY_PCT", "0.20"))
CRICKET_MIN_ORDER_USD  = 1.0       # minimum order size
CRICKET_LIMIT_OFFSET   = 0.01      # limit offset from mid (avoids ghost liquidity)

# ── Tennis Live Execution (v4.4) ─────────────────────────────────────
TENNIS_LIVE_MODE       = os.getenv("TENNIS_LIVE_MODE", "false").lower() == "true"
TENNIS_BANKROLL        = float(os.getenv("TENNIS_BANKROLL", "24.0"))
TENNIS_KELLY_PCT       = float(os.getenv("TENNIS_KELLY_PCT", "0.30"))
TENNIS_MIN_ORDER_USD   = 1.0      # minimum order size
POLY_PRIVATE_KEY       = os.getenv("POLY_PRIVATE_KEY", "")
POLY_FUNDER_ADDRESS    = os.getenv("POLY_FUNDER_ADDRESS", "")
CLOB_PROXY_URL         = os.getenv("CLOB_PROXY_URL", "")  # US proxy for geoblock bypass

# ── Data Storage ─────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("SPORTS_DATA_DIR", "sports_data"))
LOG_DIR  = Path(os.getenv("LOG_DIR", "logs"))

# ── Polymarket Sports Market Detection ───────────────────────────────
# Slug patterns that identify sports markets
SPORTS_SLUG_PATTERNS = [
    "nba-", "nfl-", "nhl-", "mlb-",
    "epl-", "la-liga", "bundesliga", "serie-a", "ligue-1",
    "eredivisie", "championship-",
    "champions-league", "europa-league", "conference-league",
    "ucl-", "uel-", "uecl-",
    "fifa-", "world-cup",
    "premier-league",
    # Polymarket shorthand football codes (v3.8)
    "lal-",   # La Liga
    "efa-",   # English Football (EPL / FA Cup)
    "bun-",   # Bundesliga
    "fl1-",   # Ligue 1
    "sa-",    # Serie A
    "ere-",   # Eredivisie
    "chm-",   # Championship
    # Football qualifiers / internationals (v4.8)
    "wcq-",   # World Cup Qualifiers
    "ecq-",   # Euro Qualifiers
    "afcon-", # AFCON / AFCON Qualifiers
    "qualifier", "qualifiers",
    # International catch-all (v4.8.1)
    "international-", "friendly-",
    "fifa-world-cup", "uefa-nations", "caf-",
    # Tennis
    "tennis", "atp-", "wta-",
    "australian-open", "french-open", "wimbledon", "us-open",
    # Cricket — Polymarket slug prefixes
    "crint-",    # Cricket International (NZ vs SA, AFG vs SL, etc.)
    "criclcl-",  # Cricket Legends League
    "ipl-",      # Indian Premier League
    "icc-", "t20-", "odi-", "cricket",
    # Cricket nation slugs (catches "india-vs-new-zealand-final" etc.)
    "india-vs", "australia-vs", "england-vs", "pakistan-vs",
    "new-zealand-vs", "south-africa-vs", "west-indies-vs",
    "sri-lanka-vs", "bangladesh-vs", "afghanistan-vs",
    "cayman-islands-vs", "ireland-vs", "zimbabwe-vs",
    "nepal-vs", "namibia-vs", "scotland-vs", "usa-vs",
    "netherlands-vs", "uae-vs", "oman-vs", "canada-vs",
]

# Polymarket slug team abbreviation → full name mapping
# (discovered dynamically, but these help with fuzzy matching)
NBA_TEAM_MAP = {
    "atl": "Atlanta Hawks", "bos": "Boston Celtics", "bkn": "Brooklyn Nets",
    "cha": "Charlotte Hornets", "chi": "Chicago Bulls", "cle": "Cleveland Cavaliers",
    "dal": "Dallas Mavericks", "den": "Denver Nuggets", "det": "Detroit Pistons",
    "gsw": "Golden State Warriors", "hou": "Houston Rockets", "ind": "Indiana Pacers",
    "lac": "LA Clippers", "lal": "Los Angeles Lakers", "mem": "Memphis Grizzlies",
    "mia": "Miami Heat", "mil": "Milwaukee Bucks", "min": "Minnesota Timberwolves",
    "nor": "New Orleans Pelicans", "nyk": "New York Knicks", "okc": "Oklahoma City Thunder",
    "orl": "Orlando Magic", "phi": "Philadelphia 76ers", "pho": "Phoenix Suns",
    "por": "Portland Trail Blazers", "sac": "Sacramento Kings", "sas": "San Antonio Spurs",
    "tor": "Toronto Raptors", "uta": "Utah Jazz", "was": "Washington Wizards",
}

# ── Sportmonks Cricket v1.0.1 ────────────────────────────────────────
SPORTMONKS_API_TOKEN   = os.getenv("SPORTMONKS_API_TOKEN", "c601bm9fdnWxe94eDVlFvEDK3K9LrsXeEOhlIbBS8mespVG52AU8WPJnrhMA")
CRICKET_SM_POLL_S      = 5.0       # scoreboard poll interval (seconds)
CRICKET_SM_LEAGUE_IDS  = [1]       # IPL = league_id 1

# ── Cricket Discovery + Health (v7.0) ────────────────────────────────
CRICKET_MIN_LIQUIDITY = 50_000          # minimum Gamma liquidity for discovery
CRICKET_MAX_DISCOVERY_SPREAD = 0.10     # reject during discovery if spread > this
CRICKET_HEALTH_INTERVAL_S = 60          # health log interval
CRICKET_READINESS_SPREAD = 0.08         # live readiness spread threshold
CRICKET_READINESS_TICK_RATE = 5         # minimum ticks/min for READY
CRICKET_READINESS_PRICE_RANGE = 0.01    # minimum price_range_60s for READY
CRICKET_READINESS_CHECK_INTERVAL_S = 300  # readiness check every 5 min
