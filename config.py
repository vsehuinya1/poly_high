"""
Configuration — loaded from .env, never hardcoded.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Polymarket CLOB credentials ──────────────────────────────────────
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_SECRET = os.getenv("POLYMARKET_SECRET", "")
POLYMARKET_PASSPHRASE = os.getenv("POLYMARKET_PASSPHRASE", "")

# ── Endpoints ────────────────────────────────────────────────────────
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
CLOB_REST_URL = "https://clob.polymarket.com"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

# ── Market discovery ─────────────────────────────────────────────────
# Comma-separated slug prefixes (e.g. "btc-updown-5m,btc-updown-4h")
_slug_raw = os.getenv("SERIES_SLUGS", os.getenv("SERIES_SLUG", "btc-updown-5m"))
SERIES_SLUGS = [s.strip() for s in _slug_raw.split(",") if s.strip()]
DISCOVERY_INTERVAL_S = int(os.getenv("DISCOVERY_INTERVAL_S", "60"))

# ── Binance ──────────────────────────────────────────────────────────
ENABLE_BINANCE = os.getenv("ENABLE_BINANCE", "true").lower() == "true"
BINANCE_SYMBOLS = ["btcusdt"]  # extend with solusdt if needed

# ── Storage ──────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
FLUSH_INTERVAL_S = int(os.getenv("FLUSH_INTERVAL_S", "30"))
FLUSH_ROW_THRESHOLD = int(os.getenv("FLUSH_ROW_THRESHOLD", "10000"))

# ── Health ───────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL_S = int(os.getenv("HEARTBEAT_INTERVAL_S", "10"))
HEALTH_FILE = DATA_DIR / "health.json"

# ── Logging ──────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))

# ── Reconnect ────────────────────────────────────────────────────────
RECONNECT_BASE_S = 1.0
RECONNECT_MAX_S = 60.0
RECONNECT_FACTOR = 2.0
