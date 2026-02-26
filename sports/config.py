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

# ── Paper Trading Parameters ─────────────────────────────────────────
ENTRY_EDGE_THRESHOLD  = 0.05     # min 5 cent edge to enter
EXIT_CONVERGENCE      = 0.01     # exit when edge < 1 cent
MAX_POSITION_PER_MARKET = 500.0  # max $500 per market
MAX_CONCURRENT_POSITIONS = 10
MAX_DAILY_LOSS        = 1000.0   # kill-switch

# ── Execution Hygiene (v1.4) ─────────────────────────────────────────
# Hard entry filters
PRICE_BAND_LO         = 0.15     # mid must be >= this
PRICE_BAND_HI         = 0.70     # mid must be <= this
MAX_SPREAD            = 0.08     # spread must be <= this
MAX_BOOK_AGE_S        = 20.0     # book_age must be <= this
MAX_SCORE_DIFF        = 15       # |home - away| must be <= this

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
