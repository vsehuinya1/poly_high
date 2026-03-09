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

# ── NBA ──────────────────────────────────────────────────────────────
NBA_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
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
ENTRY_EDGE_THRESHOLD  = 0.05     # min 5 cent edge to enter
EXIT_CONVERGENCE      = 0.01     # exit when edge < 1 cent
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
EDGE_TRADE_THRESHOLD  = 0.10     # min edge to open a trade (separate from signal threshold)
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

# ── Tennis (Strategy B — Inflection Sniping) ─────────────────────────
TENNIS_SERVE_WIN_P    = 0.64      # ATP average service point win rate
TENNIS_PANIC_EDGE     = 0.06      # min edge for panic discount trigger
TENNIS_REVERSION_EDGE = 0.08      # min edge for set mean reversion trigger (raised from 0.05 — sub-0.08 edges produce near-zero R)
TENNIS_PRICE_CAP      = 0.85      # no entries above this market price
TENNIS_STALENESS_S    = 3.0       # max state age for execution
TENNIS_COOLDOWN_S     = 120.0     # cooldown after position exit (seconds)
TENNIS_FEED_POLL_S    = 3.0       # feed poll interval (seconds)
TENNIS_FEED_STALL_S   = 60.0      # feed stall detection threshold
TENNIS_FEED_HEALTH_S  = 60.0      # feed health log interval

# Tennis execution hardening (v2.0)
TENNIS_PRICE_FLOOR         = 0.05   # min market price — below this = dead book
TENNIS_MAX_SIGNALS_HR      = 10     # max signals per match per rolling hour
TENNIS_STALE_DISABLE_COUNT = 5      # consecutive stale events before auto-disable
TENNIS_STALE_DISABLE_S     = 300    # auto-disable duration (5 minutes)

# ── Cricket (Paper-Only Research Mode) ───────────────────────────────
CRICKET_PAPER_ONLY         = True   # NEVER allow live execution
CRICKET_TRADE_SIZE         = 200.0  # paper trade size ($)
CRICKET_MAX_SPREAD         = 0.02   # abort if spread > this
CRICKET_MOMENTUM_RR_THRESH = 2.0    # rolling RR must exceed RRR by this
CRICKET_MOMENTUM_EDGE      = 0.08   # min edge for momentum signal
CRICKET_WICKET_EDGE        = 0.10   # min edge for wicket overreaction
CRICKET_LATENCY_THRESH_MS  = 2000.0 # min latency for snipe logging
CRICKET_COOLDOWN_S         = 120.0  # cooldown between trades

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
    # Tennis
    "tennis", "atp-", "wta-",
    "australian-open", "french-open", "wimbledon", "us-open",
    # Cricket
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
