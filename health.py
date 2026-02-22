"""
Health monitor — periodic heartbeat + health.json.
"""
import asyncio
import json
import logging
import time
from typing import Any

from config import HEARTBEAT_INTERVAL_S, HEALTH_FILE

log = logging.getLogger("health")


class HealthMonitor:
    def __init__(self) -> None:
        self._start = time.monotonic()
        self._msg_count = 0
        self._last_report_count = 0
        self._last_report_time = time.monotonic()
        self._ws_state: str = "disconnected"
        self._binance_state: str = "disabled"
        self._buffer_sizes: dict[str, int] = {}
        self._active_tokens: list[str] = []

    def record_message(self) -> None:
        self._msg_count += 1

    def set_ws_state(self, state: str) -> None:
        self._ws_state = state

    def set_binance_state(self, state: str) -> None:
        self._binance_state = state

    def set_buffer_sizes(self, sizes: dict[str, int]) -> None:
        self._buffer_sizes = sizes

    def set_active_tokens(self, tokens: list[str]) -> None:
        self._active_tokens = tokens

    def _snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = now - self._last_report_time
        msgs = self._msg_count - self._last_report_count
        rate = msgs / elapsed if elapsed > 0 else 0.0
        self._last_report_time = now
        self._last_report_count = self._msg_count

        return {
            "uptime_s": round(now - self._start, 1),
            "total_messages": self._msg_count,
            "messages_per_sec": round(rate, 2),
            "polymarket_ws": self._ws_state,
            "binance_ws": self._binance_state,
            "buffer_sizes": self._buffer_sizes,
            "active_tokens": len(self._active_tokens),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    async def run(self) -> None:
        HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        while True:
            snap = self._snapshot()
            log.info(
                "heartbeat | up=%ss msgs=%d rate=%.1f/s poly_ws=%s binance=%s bufs=%s tokens=%d",
                snap["uptime_s"],
                snap["total_messages"],
                snap["messages_per_sec"],
                snap["polymarket_ws"],
                snap["binance_ws"],
                snap["buffer_sizes"],
                snap["active_tokens"],
            )
            try:
                HEALTH_FILE.write_text(json.dumps(snap, indent=2))
            except OSError:
                pass
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
