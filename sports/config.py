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

# ── API-Football ─────────────────────────────────────────────────────
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_HOST = "v3.football.api-sports.io"
API_FOOTBALL_BASE = f"https://{API_FOOTBALL_HOST}"

# European football league IDs (API-Football)
FOOTBALL_LEAGUES = {
    39:  "EPL",
    40:  "Championship",
    140: "La Liga",
    78:  "Bundesliga",
    135: "Serie A",
    61:  "Ligue 1",
    88:  "Eredivisie",
    2:   "UEFA Champions League",
    3:   "UEFA Europa League",
    848: "Conference League",
}

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
