"""
Market discovery — polls Gamma API for active token IDs.
Detects market rotation and notifies WebSocket client.

BTC short-horizon markets use slug patterns like:
  btc-updown-5m-{unix_ts}   (5-minute windows)
  btc-updown-4h-{unix_ts}   (4-hour windows)
Each market has 2 clobTokenIds (Up / Down outcomes).
"""
import asyncio
import json
import logging
from typing import Callable, Awaitable

import aiohttp

from config import GAMMA_API_URL, SERIES_SLUGS, DISCOVERY_INTERVAL_S

log = logging.getLogger("discovery")


class MarketDiscovery:
    """Polls Gamma API, extracts active clobTokenIds matching configured slug prefixes."""

    def __init__(
        self,
        on_tokens_changed: Callable[[list[str], list[dict]], Awaitable[None]],
    ) -> None:
        self._on_tokens_changed = on_tokens_changed
        self._current_token_ids: set[str] = set()
        self._current_markets: list[dict] = []
        self._slug_prefixes = [s.rstrip("-") for s in SERIES_SLUGS]

    @property
    def active_token_ids(self) -> list[str]:
        return sorted(self._current_token_ids)

    async def _fetch_markets(self, session: aiohttp.ClientSession) -> list[dict]:
        """Fetch active markets matching any of the configured slug prefixes."""
        params = {
            "closed": "false",
            "active": "true",
            "limit": "100",
            "order": "createdAt",
            "ascending": "false",
        }
        try:
            async with session.get(
                GAMMA_API_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    log.warning("gamma API returned %d", resp.status)
                    return []
                data = await resp.json()
        except Exception as e:
            log.error("gamma API request failed: %s", e)
            return []

        filtered = []
        for market in data:
            slug = market.get("slug", "")
            if any(slug.startswith(prefix) for prefix in self._slug_prefixes):
                filtered.append(market)

        return filtered

    def _extract_token_ids(self, markets: list[dict]) -> set[str]:
        """Extract all clobTokenIds from market list."""
        token_ids: set[str] = set()
        for market in markets:
            raw = market.get("clobTokenIds", "[]")
            if isinstance(raw, str):
                try:
                    ids = json.loads(raw)
                except json.JSONDecodeError:
                    continue
            else:
                ids = raw
            for tid in ids:
                if tid and isinstance(tid, str):
                    token_ids.add(tid)
        return token_ids

    async def run(self) -> None:
        log.info("market discovery started (slugs=%s, poll=%ds)",
                 self._slug_prefixes, DISCOVERY_INTERVAL_S)

        async with aiohttp.ClientSession() as session:
            while True:
                markets = await self._fetch_markets(session)
                new_ids = self._extract_token_ids(markets)

                if new_ids and new_ids != self._current_token_ids:
                    added = new_ids - self._current_token_ids
                    removed = self._current_token_ids - new_ids

                    if self._current_token_ids:
                        log.info(
                            "market rotation detected: +%d -%d tokens (total=%d, markets=%d)",
                            len(added), len(removed), len(new_ids), len(markets),
                        )
                        for m in markets[:3]:
                            log.info("  active: %s | %s",
                                     m.get("slug", "")[:50],
                                     m.get("question", "")[:60])
                    else:
                        log.info("initial discovery: %d tokens across %d markets",
                                 len(new_ids), len(markets))
                        for m in markets[:5]:
                            log.info("  market: %s | %s",
                                     m.get("slug", "")[:50],
                                     m.get("question", "")[:60])

                    self._current_token_ids = new_ids
                    self._current_markets = markets

                    await self._on_tokens_changed(
                        sorted(new_ids),
                        markets,
                    )

                elif not new_ids and self._current_token_ids:
                    log.warning("no active markets found — keeping existing subscriptions")

                await asyncio.sleep(DISCOVERY_INTERVAL_S)
