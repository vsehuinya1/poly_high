"""
Market Microstructure Scanner — detects liquidity gaps and slow-reaction windows.

Monitors Polymarket book state in real-time and flags:
1. Spread widening events (liquidity withdraw → entry opportunity)
2. Stale books (market not reacting to score changes)
3. Price momentum (sustained directional movement)
4. Volume spikes (sudden interest = information event)

Sends Telegram alerts when opportunities are detected.

Wired into main.py as a callback on every book update.
"""
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger("sports.microstructure")


@dataclass
class TokenMicroState:
    """Per-token microstructure tracking state."""
    token_id: str
    market_title: str = ""
    sport: str = ""

    # Price history (circular buffer of last 60 mid-prices)
    price_history: deque = field(default_factory=lambda: deque(maxlen=120))

    # Spread tracking
    spread_history: deque = field(default_factory=lambda: deque(maxlen=60))
    normal_spread: float = 0.0  # rolling median spread
    spread_alert_sent: float = 0.0  # last spread alert time

    # Momentum tracking
    last_mid: float = 0.0
    momentum_ticks: int = 0  # consecutive same-direction ticks
    momentum_dir: int = 0  # +1 = up, -1 = down
    momentum_start_price: float = 0.0
    momentum_alert_sent: float = 0.0

    # Staleness
    last_update_ts: float = 0.0
    stale_alert_sent: float = 0.0


class MicrostructureScanner:
    """Real-time market microstructure analysis."""

    def __init__(self,
                 spread_threshold: float = 3.0,  # alert if spread > 3x normal
                 momentum_ticks: int = 8,  # consecutive same-dir ticks
                 momentum_min_move: float = 0.03,  # min price move for momentum alert
                 stale_threshold_s: float = 120.0,  # alert if no update for 2min
                 alert_cooldown_s: float = 300.0,  # 5min between alerts per token
                 tg_callback: Optional[Callable] = None):

        self.states: dict[str, TokenMicroState] = {}
        self.spread_threshold = spread_threshold
        self.momentum_ticks = momentum_ticks
        self.momentum_min_move = momentum_min_move
        self.stale_threshold = stale_threshold_s
        self.alert_cooldown = alert_cooldown_s
        self._tg = tg_callback
        self._scan_count = 0
        self._alerts_sent = 0
        self._last_summary = 0

        # Summary counters
        self.spread_events = 0
        self.momentum_events = 0
        self.stale_events = 0

    def register_token(self, token_id: str, market_title: str = "", sport: str = ""):
        """Register a token for monitoring."""
        if token_id not in self.states:
            self.states[token_id] = TokenMicroState(
                token_id=token_id,
                market_title=market_title,
                sport=sport,
            )

    def on_book_update(self, token_id: str, bid: float, ask: float,
                       mid: float, spread: float):
        """Called on every BBO update — core scan logic."""
        now = time.time()
        state = self.states.get(token_id)
        if not state:
            return

        state.last_update_ts = now
        self._scan_count += 1

        # Record price history
        state.price_history.append((now, mid))
        state.spread_history.append(spread)

        # Update normal spread (rolling median of last 30 spreads)
        if len(state.spread_history) >= 10:
            sorted_spreads = sorted(state.spread_history)
            state.normal_spread = sorted_spreads[len(sorted_spreads) // 2]

        # ── 1. Spread widening detection ──────────────────────────
        if (state.normal_spread > 0 and spread > 0 and
                spread > state.normal_spread * self.spread_threshold and
                now - state.spread_alert_sent > self.alert_cooldown):
            self.spread_events += 1
            state.spread_alert_sent = now
            self._alert(
                f"📊 <b>Spread Widening</b>\n"
                f"Market: {state.market_title[:50]}\n"
                f"Spread: {spread:.4f} ({spread/state.normal_spread:.1f}x normal)\n"
                f"Normal: {state.normal_spread:.4f}\n"
                f"Mid: {mid:.4f} | Bid: {bid:.4f} | Ask: {ask:.4f}\n"
                f"⚡ Liquidity withdrawn — potential entry window"
            )

        # ── 2. Momentum detection ─────────────────────────────────
        if state.last_mid > 0:
            if mid > state.last_mid:
                if state.momentum_dir == 1:
                    state.momentum_ticks += 1
                else:
                    state.momentum_dir = 1
                    state.momentum_ticks = 1
                    state.momentum_start_price = state.last_mid
            elif mid < state.last_mid:
                if state.momentum_dir == -1:
                    state.momentum_ticks += 1
                else:
                    state.momentum_dir = -1
                    state.momentum_ticks = 1
                    state.momentum_start_price = state.last_mid
            else:
                pass  # flat tick, keep current momentum state

            # Fire momentum alert
            move = abs(mid - state.momentum_start_price)
            if (state.momentum_ticks >= self.momentum_ticks and
                    move >= self.momentum_min_move and
                    now - state.momentum_alert_sent > self.alert_cooldown):
                self.momentum_events += 1
                state.momentum_alert_sent = now
                direction = "📈 UP" if state.momentum_dir == 1 else "📉 DOWN"
                self._alert(
                    f"🔥 <b>Momentum Detected</b>\n"
                    f"Market: {state.market_title[:50]}\n"
                    f"Direction: {direction} ({state.momentum_ticks} consecutive ticks)\n"
                    f"Move: {state.momentum_start_price:.4f} → {mid:.4f} ({move:+.4f})\n"
                    f"⚡ Market trending — check if score changed"
                )

        state.last_mid = mid

        # ── 3. Periodic summary (every 5 min) ────────────────────
        if now - self._last_summary > 300:
            self._last_summary = now
            active = sum(1 for s in self.states.values()
                        if now - s.last_update_ts < 60)
            log.info("MICROSTRUCTURE: scanning %d tokens (%d active) | "
                     "events: spread=%d momentum=%d stale=%d | ticks=%d",
                     len(self.states), active,
                     self.spread_events, self.momentum_events,
                     self.stale_events, self._scan_count)

    def check_staleness(self):
        """Periodic check for stale books (call every 30s)."""
        now = time.time()
        for state in self.states.values():
            if (state.last_update_ts > 0 and
                    now - state.last_update_ts > self.stale_threshold and
                    now - state.stale_alert_sent > self.alert_cooldown):
                self.stale_events += 1
                state.stale_alert_sent = now
                age = now - state.last_update_ts
                self._alert(
                    f"⏰ <b>Stale Book</b>\n"
                    f"Market: {state.market_title[:50]}\n"
                    f"No update for {age:.0f}s\n"
                    f"Last mid: {state.last_mid:.4f}\n"
                    f"⚡ Market may not be reacting to game events"
                )

    def get_momentum_tokens(self) -> list[tuple[str, int, float]]:
        """Get tokens currently showing momentum. Returns (token_id, ticks, move)."""
        results = []
        for state in self.states.values():
            if state.momentum_ticks >= 3:
                move = abs(state.last_mid - state.momentum_start_price)
                results.append((state.token_id, state.momentum_ticks, move))
        return sorted(results, key=lambda x: -x[2])

    def get_wide_spread_tokens(self) -> list[tuple[str, float, float]]:
        """Get tokens with above-normal spreads. Returns (token_id, spread, ratio)."""
        results = []
        for state in self.states.values():
            if (state.normal_spread > 0 and state.spread_history and
                    state.spread_history[-1] > state.normal_spread * 1.5):
                ratio = state.spread_history[-1] / state.normal_spread
                results.append((state.token_id, state.spread_history[-1], ratio))
        return sorted(results, key=lambda x: -x[2])

    def _alert(self, msg: str):
        """Send alert via Telegram or log."""
        self._alerts_sent += 1
        log.info("MICRO_ALERT: %s", msg.replace("\n", " | ").replace("<b>", "").replace("</b>", ""))
        if self._tg:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._tg(msg))
            except Exception:
                pass

    def summary(self) -> str:
        """Get summary string."""
        active = sum(1 for s in self.states.values()
                    if time.time() - s.last_update_ts < 60)
        return (f"tokens={len(self.states)} active={active} "
                f"spreads={self.spread_events} momentum={self.momentum_events} "
                f"stale={self.stale_events} alerts={self._alerts_sent}")
