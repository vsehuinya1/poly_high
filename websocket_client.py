"""
Polymarket WebSocket client — connects to CLOB market channel,
subscribes to active token IDs, processes book snapshots + trade ticks.
Reconnects with exponential backoff. Feeds data to storage engine.
"""
import asyncio
import json
import logging
import time
from typing import Optional

import websockets
import websockets.exceptions

from config import (
    POLYMARKET_WS_URL,
    RECONNECT_BASE_S,
    RECONNECT_MAX_S,
    RECONNECT_FACTOR,
)
from collector.orderbook import OrderBookManager
from collector.storage import StorageEngine
from utils.health import HealthMonitor

log = logging.getLogger("poly_ws")


class PolymarketWebSocket:
    """
    Async WebSocket client for Polymarket CLOB market channel.
    Handles subscription, book/trade parsing, reconnection, and re-subscription
    on market rotation.
    """

    def __init__(
        self,
        storage: StorageEngine,
        health: HealthMonitor,
    ) -> None:
        self._storage = storage
        self._health = health
        self._book_mgr = OrderBookManager()
        self._token_ids: list[str] = []
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._backoff = RECONNECT_BASE_S
        self._sequence: int = 0
        self._subscribe_event = asyncio.Event()
        self._shutdown = False

    async def update_tokens(self, token_ids: list[str], markets: list[dict]) -> None:
        """Called by MarketDiscovery when active token IDs change."""
        old = set(self._token_ids)
        new = set(token_ids)

        if old == new:
            return

        self._token_ids = token_ids
        log.info("token update: %d → %d tokens", len(old), len(new))

        # Signal the run loop to re-subscribe
        self._subscribe_event.set()

    async def _subscribe(self) -> None:
        """Send subscription message for current token IDs."""
        if not self._ws or not self._token_ids:
            return

        msg = {
            "type": "market",
            "assets_ids": self._token_ids,
        }
        await self._ws.send(json.dumps(msg))
        log.info("subscribed to %d token IDs", len(self._token_ids))
        self._health.set_active_tokens(self._token_ids)

    async def _handle_message(self, raw: str) -> None:
        """Route incoming WS messages by event_type."""
        self._health.record_message()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("non-JSON message: %s", raw[:200])
            return

        # Handle arrays (Polymarket may batch messages)
        messages = data if isinstance(data, list) else [data]

        for msg in messages:
            event_type = msg.get("event_type", "")

            if event_type == "book":
                await self._handle_book(msg)
            elif event_type == "tick":
                await self._handle_tick(msg)
            elif event_type == "last_trade_price":
                await self._handle_trade_price(msg)
            elif event_type == "price_change":
                pass  # Informational, no action needed
            else:
                log.debug("unhandled event_type=%s", event_type)

    async def _handle_book(self, msg: dict) -> None:
        """Process full book snapshot."""
        asset_id = msg.get("asset_id", "")
        market_id = msg.get("market", "")
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        book_hash = msg.get("hash", "")
        ts_exchange = msg.get("timestamp", "")
        ts_local = time.time_ns()

        book = self._book_mgr.get_or_create(asset_id, market_id)
        quote = book.apply_snapshot(bids, asks, book_hash)

        # Store orderbook snapshot rows
        rows = []
        for level in bids:
            rows.append({
                "timestamp_local": ts_local,
                "timestamp_exchange": ts_exchange,
                "market_id": market_id,
                "asset_id": asset_id,
                "side": "bid",
                "price": float(level.get("price", 0)),
                "size": float(level.get("size", 0)),
                "event_type": "snapshot",
            })
        for level in asks:
            rows.append({
                "timestamp_local": ts_local,
                "timestamp_exchange": ts_exchange,
                "market_id": market_id,
                "asset_id": asset_id,
                "side": "ask",
                "price": float(level.get("price", 0)),
                "size": float(level.get("size", 0)),
                "event_type": "snapshot",
            })

        if rows:
            await self._storage.orderbook.append_batch(rows)
        await self._storage.quotes.append(quote)

    async def _handle_tick(self, msg: dict) -> None:
        """Process price tick / trade update."""
        asset_id = msg.get("asset_id", "")
        market_id = msg.get("market", "")
        price = msg.get("price")
        size = msg.get("size")
        ts_local = time.time_ns()

        if price is not None:
            await self._storage.trades.append({
                "timestamp_local": ts_local,
                "trade_id": msg.get("id", str(ts_local)),
                "market_id": market_id,
                "asset_id": asset_id,
                "price": float(price),
                "size": float(size) if size else 0.0,
                "side": msg.get("side", "unknown"),
            })

            # Update book and emit quote
            book = self._book_mgr.get_or_create(asset_id, market_id)
            quote = book._quote_dict()
            await self._storage.quotes.append(quote)

    async def _handle_trade_price(self, msg: dict) -> None:
        """Process last_trade_price events."""
        asset_id = msg.get("asset_id", "")
        market_id = msg.get("market", "")
        price = msg.get("price")
        ts_local = time.time_ns()

        if price is not None:
            await self._storage.trades.append({
                "timestamp_local": ts_local,
                "trade_id": str(ts_local),
                "market_id": market_id,
                "asset_id": asset_id,
                "price": float(price),
                "size": 0.0,
                "side": "unknown",
            })

    async def run(self) -> None:
        """Main loop — connect, subscribe, process messages, reconnect."""
        log.info("polymarket WS client starting")

        while not self._shutdown:
            try:
                async with websockets.connect(
                    POLYMARKET_WS_URL,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=5,
                    max_size=2**22,  # 4 MB
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._backoff = RECONNECT_BASE_S
                    self._health.set_ws_state("connected")
                    log.info("connected to %s", POLYMARKET_WS_URL)

                    # Subscribe if we already have tokens
                    if self._token_ids:
                        await self._subscribe()

                    # Message loop
                    while not self._shutdown:
                        # Check for re-subscribe signal
                        if self._subscribe_event.is_set():
                            self._subscribe_event.clear()
                            await self._subscribe()

                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                            await self._handle_message(raw)
                        except asyncio.TimeoutError:
                            # No message in 30s — send ping to keep alive
                            try:
                                pong = await ws.ping()
                                await asyncio.wait_for(pong, timeout=10)
                            except Exception:
                                log.warning("ping failed, reconnecting")
                                break

            except websockets.exceptions.ConnectionClosed as e:
                log.warning("WS connection closed: %s", e)
            except Exception as e:
                log.error("WS error: %s", e)
            finally:
                self._connected = False
                self._ws = None
                self._health.set_ws_state("disconnected")

            if not self._shutdown:
                log.info("reconnecting in %.1fs", self._backoff)
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * RECONNECT_FACTOR, RECONNECT_MAX_S)

    async def shutdown(self) -> None:
        self._shutdown = True
        if self._ws:
            await self._ws.close()
        log.info("polymarket WS client stopped")
