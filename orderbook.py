"""
Local order book engine — maintains bid/ask levels from snapshots + deltas,
computes BBO, detects desync via hash mismatch.
"""
import hashlib
import logging
import time
from typing import Any

log = logging.getLogger("orderbook")


class OrderBook:
    """In-memory order book for a single asset_id."""

    def __init__(self, asset_id: str, market_id: str = "") -> None:
        self.asset_id = asset_id
        self.market_id = market_id
        self.bids: dict[float, float] = {}  # price → size
        self.asks: dict[float, float] = {}  # price → size
        self._last_hash: str = ""
        self._last_update = time.time()
        self._synced = False

    @property
    def best_bid(self) -> float:
        return max(self.bids.keys()) if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return min(self.asks.keys()) if self.asks else 0.0

    @property
    def spread(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb > 0 and ba > 0:
            return ba - bb
        return 0.0

    @property
    def midprice(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb > 0 and ba > 0:
            return (bb + ba) / 2.0
        return 0.0

    def apply_snapshot(self, bids: list[dict], asks: list[dict], book_hash: str = "") -> dict[str, Any]:
        """Replace entire book with snapshot data."""
        self.bids.clear()
        self.asks.clear()

        for level in bids:
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))
            if price > 0 and size > 0:
                self.bids[price] = size

        for level in asks:
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))
            if price > 0 and size > 0:
                self.asks[price] = size

        self._last_hash = book_hash
        self._last_update = time.time()
        self._synced = True

        log.debug("snapshot applied: %d bids, %d asks (asset=%s)",
                  len(self.bids), len(self.asks), self.asset_id[:16])

        return self._quote_dict()

    def apply_update(self, side: str, price: float, size: float) -> dict[str, Any]:
        """Apply incremental update. size=0 means remove level."""
        book = self.bids if side == "bid" else self.asks

        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size

        self._last_update = time.time()
        return self._quote_dict()

    def check_hash(self, expected_hash: str) -> bool:
        """Check if book state matches expected hash. Returns True if ok."""
        if not expected_hash or not self._last_hash:
            return True  # Can't verify
        if expected_hash != self._last_hash:
            log.warning("hash mismatch on %s — desync detected", self.asset_id[:16])
            self._synced = False
            return False
        return True

    @property
    def is_synced(self) -> bool:
        return self._synced

    @property
    def stale_seconds(self) -> float:
        return time.time() - self._last_update

    def _quote_dict(self) -> dict[str, Any]:
        return {
            "timestamp_local": time.time_ns(),
            "market_id": self.market_id,
            "asset_id": self.asset_id,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "midprice": self.midprice,
        }


class OrderBookManager:
    """Manages order books for multiple asset IDs."""

    def __init__(self) -> None:
        self._books: dict[str, OrderBook] = {}

    def get_or_create(self, asset_id: str, market_id: str = "") -> OrderBook:
        if asset_id not in self._books:
            self._books[asset_id] = OrderBook(asset_id, market_id)
            log.info("created orderbook for asset %s", asset_id[:16])
        return self._books[asset_id]

    def remove(self, asset_id: str) -> None:
        self._books.pop(asset_id, None)

    def clear(self) -> None:
        self._books.clear()

    @property
    def desynced_books(self) -> list[str]:
        return [aid for aid, book in self._books.items() if not book.is_synced]
