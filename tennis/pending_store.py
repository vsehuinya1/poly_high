"""
v9.8A — Tennis Pending Queue Persistence.

Lightweight JSON persistence for the _tennis_pending queue.
Survives engine restarts by writing state to disk on every mutation
and restoring on startup.

Only serializes scalar fields; TennisSignal/GameMarketLink/TennisState
are reconstructed from live orchestrator state on restore.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("tennis.pending_store")

# Default path (relative to DATA_DIR)
PENDING_FILE = "tennis_pending.json"


class PendingStore:
    """JSON-backed persistence for tennis pending entries.

    Write-through: every mutation flushes to disk immediately.
    Read on startup: loads and filters expired entries.
    """

    def __init__(self, data_dir: Path):
        self._path = data_dir / PENDING_FILE
        data_dir.mkdir(parents=True, exist_ok=True)

    # ── Serialization ────────────────────────────────────────────

    @staticmethod
    def _serialize_pending(key: tuple, pend: dict) -> dict:
        """Extract serializable fields from a pending entry.

        Args:
            key:  (match_id, token_id, direction) tuple
            pend: The pending dict from _tennis_pending
        """
        signal = pend.get("signal")
        state = pend.get("state")
        link = pend.get("link")

        return {
            "match_id": key[0],
            "token_id": key[1],
            "direction": key[2],
            "start_time": pend["start_time"],
            "confirm_count": pend["confirm_count"],
            "initial_edge": pend["initial_edge"],
            "last_edge": pend["last_edge"],
            "market_price": pend["market_price"],
            # v10: Trade identity
            "trade_id": pend.get("trade_id", ""),
            "trade_key": pend.get("trade_key", ""),
            "attempted": pend.get("attempted", False),
            # Signal fields (needed for rehydration)
            "trigger_type": signal.trigger_type if signal else "",
            "fair_price": signal.fair_price if signal else 0.0,
            "signal_edge": signal.edge if signal else 0.0,
            "signal_market_price": signal.market_price if signal else 0.0,
            # State fields (for score context on restore)
            "sets_a": state.sets_a if state else 0,
            "sets_b": state.sets_b if state else 0,
            "games_a": state.games_a if state else 0,
            "games_b": state.games_b if state else 0,
            # Link context
            "polymarket_title": link.polymarket_title[:60] if link else "",
            "tournament": getattr(link, "tournament", "") if link else "",
            "tier": getattr(link, "tier", "unknown") if link else "unknown",
        }

    @staticmethod
    def _make_key(entry: dict) -> tuple:
        """Reconstruct the pending key tuple from a serialized entry."""
        return (entry["match_id"], entry["token_id"], entry["direction"])

    # ── Write ────────────────────────────────────────────────────

    def save(self, pending: dict) -> None:
        """Flush entire pending queue to disk.

        Called on every mutation (add, update, delete).
        """
        entries = []
        for key, pend in pending.items():
            try:
                entries.append(self._serialize_pending(key, pend))
            except Exception as e:
                log.warning("TENNIS_PENDING_SERIALIZE_FAIL | %s | %s", key, e)

        try:
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(entries, f, indent=2)
            tmp.rename(self._path)
            log.info("TENNIS_PENDING_SAVED | count=%d", len(entries))
        except Exception as e:
            log.error("TENNIS_PENDING_SAVE_FAIL | %s", e)

    # ── Read ─────────────────────────────────────────────────────

    def load(self, max_delay_s: float) -> list[dict]:
        """Load pending entries from disk, filtering expired ones.

        Args:
            max_delay_s: Maximum allowed age (TENNIS_ENTRY_MAX_DELAY_S).

        Returns:
            List of non-expired serialized entries.
        """
        if not self._path.exists():
            return []

        try:
            with open(self._path) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log.error("TENNIS_PENDING_LOAD_FAIL | %s", e)
            return []

        now = time.time()
        valid = []
        expired_count = 0

        for entry in entries:
            elapsed = now - entry.get("start_time", 0)
            if elapsed > max_delay_s:
                log.info(
                    "TENNIS_PENDING_EXPIRED_ON_RESTORE | trade_id=%s_%s | elapsed=%.0fs | %s",
                    entry.get("match_id", "?")[:12],
                    entry.get("token_id", "?")[:8],
                    elapsed,
                    entry.get("polymarket_title", ""),
                )
                expired_count += 1
            else:
                valid.append(entry)

        total = len(entries)
        log.info(
            "TENNIS_PENDING_RESTORED | count=%d | expired=%d | total_on_disk=%d",
            len(valid), expired_count, total,
        )

        # Clean up the file (remove expired entries)
        if expired_count > 0:
            try:
                with open(self._path, "w") as f:
                    json.dump(valid, f, indent=2)
            except Exception:
                pass

        return valid

    # ── Cleanup ──────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove the persistence file."""
        try:
            self._path.unlink(missing_ok=True)
        except Exception:
            pass
