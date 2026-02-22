"""
Structured rotating log setup.
"""
import logging
import logging.handlers
import sys
from pathlib import Path

from config import LOG_LEVEL, LOG_DIR


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Rotating file handler (50 MB x 5 backups) ────────────────────
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "collector.log",
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)

    # ── Console handler ──────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.addHandler(fh)
    root.addHandler(ch)

    # Silence noisy libs
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    return root
