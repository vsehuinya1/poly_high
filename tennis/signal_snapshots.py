"""
v9.7 — Tennis Signal Timing Observability.

Async scheduler that captures post-signal price snapshots at +5s, +10s, +30s.
Read-only instrumentation — zero impact on trading decisions.

Usage:
    scheduler = SignalSnapshotScheduler(poly_feed)
    scheduler.schedule(signal_id, token_id, signal_time, price_signal,
                       match_id, trigger_type, logger)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sports.feeds import PolymarketFeed
    from tennis.logger import TennisCSVLogger

log = logging.getLogger("tennis.snapshots")

# Capture delays (seconds after signal)
SNAPSHOT_DELAYS = [5, 10, 30]


@dataclass
class SignalSnapshot:
    """Stores post-signal price observations."""
    signal_id: str
    match_id: str
    token_id: str
    trigger_type: str
    signal_time: float
    price_signal: float
    price_t5: Optional[float] = None
    price_t10: Optional[float] = None
    price_t30: Optional[float] = None
    time_to_first_tick: Optional[float] = None
    _completed: int = 0  # how many snapshots have been captured


class SignalSnapshotScheduler:
    """Non-blocking async scheduler for post-signal price captures.

    Reads from poly_feed.books — never writes, never modifies state.
    """

    def __init__(self, poly_feed: "PolymarketFeed"):
        self._poly_feed = poly_feed
        self._snapshots: dict[str, SignalSnapshot] = {}
        # Track last known price per token for first-tick detection
        self._price_at_signal: dict[str, tuple[float, float]] = {}  # signal_id → (bid, ask)

    def _get_mid(self, token_id: str) -> Optional[float]:
        """Read current mid price from the feed. Pure read, no side effects."""
        book = self._poly_feed.books.get(token_id)
        if book and book.mid > 0:
            return book.mid
        return None

    def _get_bbo(self, token_id: str) -> Optional[tuple[float, float]]:
        """Read current best bid/ask."""
        book = self._poly_feed.books.get(token_id)
        if book and book.mid > 0:
            return (book.best_bid, book.best_ask)
        return None

    def schedule(
        self,
        signal_id: str,
        token_id: str,
        signal_time: float,
        price_signal: float,
        match_id: str,
        trigger_type: str,
        logger: "TennisCSVLogger",
    ) -> None:
        """Schedule async price captures at +5s, +10s, +30s.

        Non-blocking — fires background tasks and returns immediately.
        """
        snap = SignalSnapshot(
            signal_id=signal_id,
            match_id=match_id,
            token_id=token_id,
            trigger_type=trigger_type,
            signal_time=signal_time,
            price_signal=price_signal,
        )
        self._snapshots[signal_id] = snap

        # Record BBO at signal time for first-tick detection
        bbo = self._get_bbo(token_id)
        if bbo:
            self._price_at_signal[signal_id] = bbo

        # Schedule captures
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._capture_sequence(snap, token_id, logger))
        except RuntimeError:
            log.warning("SNAPSHOT_SCHEDULE_FAIL | no event loop | %s", signal_id)

    async def _capture_sequence(
        self,
        snap: SignalSnapshot,
        token_id: str,
        logger: "TennisCSVLogger",
    ) -> None:
        """Run the 3-step capture sequence as a single coroutine."""
        first_tick_detected = False

        for delay in SNAPSHOT_DELAYS:
            # Wait until target time
            target = snap.signal_time + delay
            wait = target - time.time()
            if wait > 0:
                # Poll every 0.5s while waiting to detect first tick
                while time.time() < target:
                    if not first_tick_detected:
                        bbo = self._get_bbo(token_id)
                        orig = self._price_at_signal.get(snap.signal_id)
                        if bbo and orig and bbo != orig:
                            snap.time_to_first_tick = time.time() - snap.signal_time
                            first_tick_detected = True
                    await asyncio.sleep(0.5)

            # Capture price
            price = self._get_mid(token_id)
            if delay == 5:
                snap.price_t5 = price
            elif delay == 10:
                snap.price_t10 = price
            elif delay == 30:
                snap.price_t30 = price

            snap._completed += 1
            log.info(
                "SIGNAL_SNAPSHOT | %s | t+%ds | price=%.4f | %s",
                snap.signal_id[:12],
                delay,
                price if price else 0.0,
                snap.match_id,
            )

        # All captures done — write to CSV
        try:
            logger.log_signal_snapshot(snap)
        except Exception as e:
            log.error("SNAPSHOT_LOG_FAIL | %s | %s", snap.signal_id, e)

        # Cleanup
        self._price_at_signal.pop(snap.signal_id, None)

    def get_snapshot(self, signal_id: str) -> Optional[SignalSnapshot]:
        """Retrieve a completed or partial snapshot."""
        return self._snapshots.get(signal_id)
