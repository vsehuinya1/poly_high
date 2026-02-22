"""
Binance WebSocket client (optional) — captures BTCUSDT trades + BBO.
Isolated failure: if Binance dies, the Polymarket collector continues.
"""
import asyncio
import json
import logging
import time
from typing import Optional

import websockets
import websockets.exceptions

from config import (
    BINANCE_WS_URL,
    BINANCE_SYMBOLS,
    ENABLE_BINANCE,
    RECONNECT_BASE_S,
    RECONNECT_MAX_S,
    RECONNECT_FACTOR,
)
from collector.storage import StorageEngine
from utils.health import HealthMonitor

log = logging.getLogger("binance")


class BinanceWebSocket:
    """
    Async WebSocket client for Binance public streams.
    Subscribes to trade + bookTicker for configured symbols.
    """

    def __init__(self, storage: StorageEngine, health: HealthMonitor) -> None:
        self._storage = storage
        self._health = health
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._backoff = RECONNECT_BASE_S
        self._shutdown = False

    def _build_url(self) -> str:
        """Build combined stream URL."""
        streams = []
        for sym in BINANCE_SYMBOLS:
            streams.append(f"{sym}@trade")
            streams.append(f"{sym}@bookTicker")
        combined = "/".join(streams)
        return f"{BINANCE_WS_URL}/{combined}"

    async def _handle_message(self, raw: str) -> None:
        self._health.record_message()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        event = data.get("e", "")
        ts_local = time.time_ns()

        if event == "trade":
            await self._storage.binance.append({
                "timestamp_local": ts_local,
                "timestamp_exchange": int(data.get("T", 0)),
                "symbol": data.get("s", ""),
                "event_type": "trade",
                "price": float(data.get("p", 0)),
                "size": float(data.get("q", 0)),
                "side": "sell" if data.get("m", False) else "buy",
                "best_bid": 0.0,
                "best_ask": 0.0,
            })

        elif event == "bookTicker" or "b" in data:
            # bookTicker doesn't have "e" field in combined streams
            await self._storage.binance.append({
                "timestamp_local": ts_local,
                "timestamp_exchange": int(data.get("u", 0)),
                "symbol": data.get("s", ""),
                "event_type": "bbo",
                "price": 0.0,
                "size": 0.0,
                "side": "",
                "best_bid": float(data.get("b", 0)),
                "best_ask": float(data.get("a", 0)),
            })

    async def run(self) -> None:
        if not ENABLE_BINANCE:
            log.info("binance feed disabled")
            self._health.set_binance_state("disabled")
            return

        url = self._build_url()
        log.info("binance WS starting: %s", url)

        while not self._shutdown:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=5,
                    max_size=2**20,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._backoff = RECONNECT_BASE_S
                    self._health.set_binance_state("connected")
                    log.info("binance connected")

                    while not self._shutdown:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                            await self._handle_message(raw)
                        except asyncio.TimeoutError:
                            try:
                                pong = await ws.ping()
                                await asyncio.wait_for(pong, timeout=10)
                            except Exception:
                                log.warning("binance ping failed")
                                break

            except websockets.exceptions.ConnectionClosed as e:
                log.warning("binance connection closed: %s", e)
            except Exception as e:
                log.error("binance error: %s (non-fatal)", e)
            finally:
                self._connected = False
                self._ws = None
                self._health.set_binance_state("disconnected")

            if not self._shutdown:
                log.info("binance reconnecting in %.1fs", self._backoff)
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * RECONNECT_FACTOR, RECONNECT_MAX_S)

    async def shutdown(self) -> None:
        self._shutdown = True
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        log.info("binance WS stopped")
