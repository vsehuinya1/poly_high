"""
Cricket Market Health Monitor + Readiness Engine + Dead Market Detection.

v2.0 — 2026-04-09  Late-liquidity activation, WARMUP phase, dead→alive recovery
v1.0 — 2026-04-07  Initial health monitor + readiness engine
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("cricket.health")

# ═══════════════════════════════════════════════════════════════════════
#  Dynamic Liquidity Thresholds (v2.0)
# ═══════════════════════════════════════════════════════════════════════

# Minimum liquidity (USD) by time window relative to match start
# Before T-60:  20k  (loose — early discovery)
# T-60 → T+30:  10k  (loosest — catch late-populating markets)
# After T+30:   50k  (strict — must be genuinely active)
LIQUIDITY_THRESHOLD_PRE = 20_000
LIQUIDITY_THRESHOLD_MATCH = 10_000
LIQUIDITY_THRESHOLD_LIVE = 50_000

# Warmup spread: subscribe but don't trade until spread ≤ this
WARMUP_MAX_SPREAD = 0.20  # subscribe when ≤ 0.20
TRADE_READY_SPREAD = 0.08  # allow trades when ≤ 0.08


# ═══════════════════════════════════════════════════════════════════════
#  Enums + Data Types
# ═══════════════════════════════════════════════════════════════════════

class MarketHealth(Enum):
    HEALTHY  = "HEALTHY"
    DEGRADED = "DEGRADED"
    DEAD     = "DEAD"


class ReadinessStatus(Enum):
    READY     = "READY"
    NOT_READY = "NOT_READY"


class FailureReason(Enum):
    NO_LIQUIDITY = "NO_LIQUIDITY"
    DEAD_BOOK    = "DEAD_BOOK"
    NO_MOVEMENT  = "NO_MOVEMENT"
    LOW_TICKS    = "LOW_TICKS"
    BAD_MAPPING  = "BAD_MAPPING"
    WARMUP       = "WARMUP"       # v2.0: subscribed but spread too wide for trading


@dataclass
class ReadinessResult:
    """Output of check_cricket_readiness()."""
    status: ReadinessStatus
    reason: Optional[FailureReason] = None
    issues: list[str] = field(default_factory=list)
    spread: float = 0.0
    tick_rate: float = 0.0
    price_range: float = 0.0
    last_tick_age: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Per-Market Tick Tracker
# ═══════════════════════════════════════════════════════════════════════

class _MarketTracker:
    """Tracks ticks and mid-price history for a single market."""

    def __init__(self, match_id: str):
        self.match_id = match_id
        self._tick_times: deque[float] = deque(maxlen=600)
        self._mid_history: deque[tuple[float, float]] = deque(maxlen=600)  # (ts, mid)
        self.last_spread: float = 1.0
        self.last_bid: float = 0.0
        self.last_ask: float = 0.0
        self.last_mid: float = 0.0
        self.last_update_ts: float = 0.0
        self.health: MarketHealth = MarketHealth.DEAD
        self._dead_since: float = 0.0  # timestamp when dead conditions first met

    def tick(self, mid: float, spread: float, bid: float, ask: float,
             ts: float) -> None:
        self._tick_times.append(ts)
        self._mid_history.append((ts, mid))
        self.last_spread = spread
        self.last_bid = bid
        self.last_ask = ask
        self.last_mid = mid
        self.last_update_ts = ts

    @property
    def tick_rate(self) -> float:
        """Ticks per minute over the last 60 seconds."""
        if len(self._tick_times) < 2:
            return 0.0
        now = self._tick_times[-1]
        cutoff = now - 60.0
        recent = sum(1 for t in self._tick_times if t >= cutoff)
        return recent  # ticks in last 60s = ticks/min

    @property
    def price_range_60s(self) -> float:
        """Max - min mid price over the last 60 seconds."""
        if len(self._mid_history) < 2:
            return 0.0
        now = self._mid_history[-1][0]
        cutoff = now - 60.0
        prices = [m for t, m in self._mid_history if t >= cutoff and m > 0]
        if len(prices) < 2:
            return 0.0
        return max(prices) - min(prices)

    @property
    def last_update_age(self) -> float:
        if self.last_update_ts <= 0:
            return 9999.0
        return time.time() - self.last_update_ts


# ═══════════════════════════════════════════════════════════════════════
#  Health Monitor (Part 3)
# ═══════════════════════════════════════════════════════════════════════

class CricketBookHealthMonitor:
    """Tracks per-market health and logs diagnostics every 60s.

    Also implements dead market detection (Part 6) and dead→alive
    recovery with CRICKET_LATE_ACTIVATION logging (v2.0).
    """

    HEALTH_LOG_INTERVAL_S = 60.0
    # Dead market thresholds (Part 6)
    DEAD_PERSIST_S = 60.0  # condition must persist > 60s
    DEAD_PRICE_RANGE = 0.005  # price_range_60s < this = no movement
    # Recovery thresholds (v2.0) — to prevent flip-flopping
    RECOVERY_SPREAD_MAX = 0.15   # must tighten below this
    RECOVERY_PRICE_RANGE_MIN = 0.005  # must show some movement
    RECOVERY_PERSIST_S = 30.0  # alive conditions must persist 30s
    # CRICKET_TICK_ACTIVE logging interval
    TICK_ACTIVE_LOG_S = 120.0

    def __init__(self):
        self._trackers: dict[str, _MarketTracker] = {}
        self._last_health_log: float = 0.0
        self._dead_markets: set[str] = set()
        self._last_status: dict[str, MarketHealth] = {}  # for change detection
        # v2.0: recovery tracking
        self._recovery_since: dict[str, float] = {}  # match_id → timestamp
        self._warmup_markets: set[str] = set()  # markets in WARMUP state
        self._activated_markets: set[str] = set()  # markets that transitioned DEAD→ALIVE
        self._last_tick_active_log: dict[str, float] = {}  # per-market active tick logging

    def _get_tracker(self, match_id: str) -> _MarketTracker:
        if match_id not in self._trackers:
            self._trackers[match_id] = _MarketTracker(match_id)
        return self._trackers[match_id]

    def tick(self, match_id: str, mid: float, spread: float,
             bid: float, ask: float, ts: float,
             market_title: str = "") -> None:
        """Record a new tick for a market."""
        tracker = self._get_tracker(match_id)
        tracker.tick(mid, spread, bid, ask, ts)

        # Classify health
        tracker.health = self._classify(tracker)

        # Dead market detection (Part 6)
        self._check_dead(match_id, tracker, ts)

    def _classify(self, t: _MarketTracker) -> MarketHealth:
        """Classify market health: HEALTHY, DEGRADED, or DEAD."""
        spread = t.last_spread
        tick_rate = t.tick_rate
        pr = t.price_range_60s
        age = t.last_update_age

        # DEAD conditions
        if spread >= 0.90:
            return MarketHealth.DEAD
        if age > 60.0:
            return MarketHealth.DEAD
        if t.last_bid <= 0.02 and t.last_ask >= 0.98:
            return MarketHealth.DEAD

        # HEALTHY: tight spread, active ticks, real movement
        if spread <= 0.05 and tick_rate > 10 and pr >= 0.01:
            return MarketHealth.HEALTHY

        # Everything else is DEGRADED
        return MarketHealth.DEGRADED

    def _check_dead(self, match_id: str, tracker: _MarketTracker,
                    now: float) -> None:
        """Part 6: Hard dead market rules with persistence check."""
        is_dead = False

        # Rule 1: bid = 0.01 AND ask = 0.99
        if tracker.last_bid <= 0.01 and tracker.last_ask >= 0.99:
            is_dead = True

        # Rule 2: mid stuck at 0.50 with no movement
        if (abs(tracker.last_mid - 0.50) < 0.005 and
                tracker.price_range_60s < self.DEAD_PRICE_RANGE):
            is_dead = True

        # Rule 3: price_range_60s < 0.005
        if (tracker.price_range_60s < self.DEAD_PRICE_RANGE and
                tracker.tick_rate > 0):  # only if we have ticks
            is_dead = True

        if is_dead:
            if tracker._dead_since <= 0:
                tracker._dead_since = now
            elif now - tracker._dead_since > self.DEAD_PERSIST_S:
                if match_id not in self._dead_markets:
                    self._dead_markets.add(match_id)
                    log.warning(
                        "CRICKET_MARKET_DEAD | %s | bid=%.2f ask=%.2f "
                        "mid=%.4f | spread=%.4f | price_range_60s=%.4f | "
                        "dead_for=%.0fs",
                        match_id, tracker.last_bid, tracker.last_ask,
                        tracker.last_mid, tracker.last_spread,
                        tracker.price_range_60s,
                        now - tracker._dead_since,
                    )
                tracker.health = MarketHealth.DEAD
            # Clear recovery tracking while dead
            self._recovery_since.pop(match_id, None)
        else:
            # ── v2.0: Dead → Alive recovery ──────────────────────
            if match_id in self._dead_markets:
                # Check recovery conditions: spread tightened + some movement
                spread_ok = tracker.last_spread <= self.RECOVERY_SPREAD_MAX
                move_ok = tracker.price_range_60s >= self.RECOVERY_PRICE_RANGE_MIN
                if spread_ok and move_ok:
                    if match_id not in self._recovery_since:
                        self._recovery_since[match_id] = now
                    elif now - self._recovery_since[match_id] > self.RECOVERY_PERSIST_S:
                        # Confirmed recovery — remove from dead set
                        self._dead_markets.discard(match_id)
                        self._recovery_since.pop(match_id, None)
                        self._activated_markets.add(match_id)
                        log.warning(
                            "CRICKET_LATE_ACTIVATION | %s | bid=%.3f ask=%.3f "
                            "spread=%.4f | price_range=%.4f | "
                            "was_dead_for=%.0fs",
                            match_id, tracker.last_bid, tracker.last_ask,
                            tracker.last_spread, tracker.price_range_60s,
                            now - tracker._dead_since if tracker._dead_since > 0 else 0,
                        )
                        tracker._dead_since = 0.0
                else:
                    self._recovery_since.pop(match_id, None)
            else:
                tracker._dead_since = 0.0

    def is_dead(self, match_id: str) -> bool:
        """Check if a market is marked DEAD (Part 6)."""
        return match_id in self._dead_markets

    def is_warmup(self, match_id: str) -> bool:
        """v2.0: Check if market is in WARMUP state (subscribed but not tradable)."""
        return match_id in self._warmup_markets

    def set_warmup(self, match_id: str) -> None:
        """Mark market as WARMUP."""
        self._warmup_markets.add(match_id)

    def clear_warmup(self, match_id: str) -> None:
        """Remove WARMUP flag — market is now fully tradable."""
        self._warmup_markets.discard(match_id)

    def was_late_activated(self, match_id: str) -> bool:
        """Check if market was dead and then recovered (v2.0)."""
        return match_id in self._activated_markets

    def log_tick_active(self, match_id: str, market_title: str,
                        tracker: '_MarketTracker') -> None:
        """v2.0: Log CRICKET_TICK_ACTIVE for markets with real price movement.

        Only logs if:
         - mid deviates from 0.50 (real price discovery)
         - price_range > 0 (not flat)
         - rate limited to every 120s per market
        """
        now = time.time()
        if now - self._last_tick_active_log.get(match_id, 0) < self.TICK_ACTIVE_LOG_S:
            return

        # Only log if mid is NOT stuck at 0.50 (dead default)
        mid_real = abs(tracker.last_mid - 0.50) > 0.01
        has_movement = tracker.price_range_60s > 0.001

        if mid_real or has_movement:
            self._last_tick_active_log[match_id] = now
            log.info(
                "CRICKET_TICK_ACTIVE | %s | mid=%.4f | spread=%.4f | "
                "ticks/min=%.0f | price_range=%.4f | late_activated=%s",
                market_title[:50],
                tracker.last_mid, tracker.last_spread,
                tracker.tick_rate, tracker.price_range_60s,
                match_id in self._activated_markets,
            )

    def log_health(self, market_titles: dict[str, str] | None = None) -> None:
        """Log per-market health every 60s."""
        now = time.time()
        if now - self._last_health_log < self.HEALTH_LOG_INTERVAL_S:
            return
        self._last_health_log = now

        titles = market_titles or {}
        for mid, tracker in self._trackers.items():
            title = titles.get(mid, mid[:20])
            log.info(
                "CRICKET_BOOK_HEALTH | %s | status=%s | spread=%.4f | "
                "bid=%.4f | ask=%.4f | mid=%.4f | tick_rate=%.0f | "
                "last_update_age=%.0fs | price_range_60s=%.4f",
                title[:50], tracker.health.value,
                tracker.last_spread, tracker.last_bid, tracker.last_ask,
                tracker.last_mid, tracker.tick_rate,
                tracker.last_update_age, tracker.price_range_60s,
            )

    def get_tracker(self, match_id: str) -> Optional[_MarketTracker]:
        """Public accessor for readiness checks."""
        return self._trackers.get(match_id)


# ═══════════════════════════════════════════════════════════════════════
#  Time-Aware Spread Filter (Part 2)
# ═══════════════════════════════════════════════════════════════════════

class SpreadPhase(Enum):
    PRE    = "PRE"     # T-60 → T-5
    EARLY  = "EARLY"   # first ~10 min of match
    LIVE   = "LIVE"    # steady-state live
    WARMUP = "WARMUP"  # v2.0: subscribed but not tradable yet

# Phase → max allowed spread
SPREAD_THRESHOLDS = {
    SpreadPhase.PRE:    0.15,
    SpreadPhase.EARLY:  0.12,
    SpreadPhase.LIVE:   0.08,
    SpreadPhase.WARMUP: 0.20,  # v2.0: allow subscription at wide spread
}


def get_spread_phase(match_start_ts: float, now: float | None = None) -> SpreadPhase:
    """Determine spread phase based on match start time."""
    if now is None:
        now = time.time()
    dt = now - match_start_ts
    if dt < -300:  # more than 5 min before start
        return SpreadPhase.PRE
    if dt < 600:  # first 10 min after start
        return SpreadPhase.EARLY
    return SpreadPhase.LIVE


def spread_ok(spread: float, phase: SpreadPhase) -> bool:
    """Check if spread is acceptable for the given phase."""
    return spread <= SPREAD_THRESHOLDS[phase]


# ═══════════════════════════════════════════════════════════════════════
#  Readiness Engine (Part 4)
# ═══════════════════════════════════════════════════════════════════════

def check_cricket_readiness(
    match_id: str,
    token_ids: list[str],
    books: dict,             # token_id → BookState
    health_monitor: CricketBookHealthMonitor,
    phase: SpreadPhase = SpreadPhase.LIVE,
) -> ReadinessResult:
    """Validate that a cricket market is actually tradable.

    Returns a ReadinessResult with READY or NOT_READY + reason + issues.
    """
    issues = []
    spread = 0.0
    tick_rate = 0.0
    price_range = 0.0
    last_tick_age = 9999.0

    # 1. Valid token mapping
    if not token_ids or not any(token_ids):
        return ReadinessResult(
            status=ReadinessStatus.NOT_READY,
            reason=FailureReason.BAD_MAPPING,
            issues=["no token IDs mapped"],
        )

    # 2. Check books exist and are not empty
    has_book = False
    for tid in token_ids:
        if not tid:
            continue
        bk = books.get(tid)
        if bk and getattr(bk, 'mid', 0) > 0:
            has_book = True
            spread = getattr(bk, 'spread', 1.0)
            break

    if not has_book:
        issues.append("no book data from WS")
        return ReadinessResult(
            status=ReadinessStatus.NOT_READY,
            reason=FailureReason.NO_LIQUIDITY,
            issues=issues,
        )

    # 3. Get tracker data from health monitor
    tracker = health_monitor.get_tracker(match_id)
    if tracker:
        tick_rate = tracker.tick_rate
        price_range = tracker.price_range_60s
        last_tick_age = tracker.last_update_age
        spread = tracker.last_spread
    else:
        issues.append("no tick history yet")

    # 4. Dead book check
    if health_monitor.is_dead(match_id):
        return ReadinessResult(
            status=ReadinessStatus.NOT_READY,
            reason=FailureReason.DEAD_BOOK,
            issues=["market marked DEAD"],
            spread=spread, tick_rate=tick_rate,
            price_range=price_range, last_tick_age=last_tick_age,
        )

    # 5. Spread within threshold for phase
    max_spread = SPREAD_THRESHOLDS[phase]
    if spread > max_spread:
        issues.append(f"spread {spread:.4f} > {max_spread:.2f} ({phase.value})")

    # 5b. v2.0: WARMUP check — spread ≤ 0.20 but > 0.08 → subscribe, don't trade
    if spread <= WARMUP_MAX_SPREAD and spread > TRADE_READY_SPREAD:
        if health_monitor:
            health_monitor.set_warmup(match_id)
        return ReadinessResult(
            status=ReadinessStatus.NOT_READY,
            reason=FailureReason.WARMUP,
            issues=[f"WARMUP: spread {spread:.4f} ≤ 0.20 but > 0.08"],
            spread=spread, tick_rate=tick_rate,
            price_range=price_range, last_tick_age=last_tick_age,
        )
    elif spread <= TRADE_READY_SPREAD and health_monitor:
        health_monitor.clear_warmup(match_id)

    # 6. Tick rate ≥ 5/min
    if tick_rate < 5:
        issues.append(f"tick_rate {tick_rate:.0f} < 5/min")

    # 7. Price movement ≥ 0.01 (CRITICAL — no false positives)
    if price_range < 0.01:
        issues.append(f"price_range_60s {price_range:.4f} < 0.01")

    # Determine primary failure reason
    if issues:
        if "DEAD" in str(issues):
            reason = FailureReason.DEAD_BOOK
        elif any("spread" in i for i in issues):
            reason = FailureReason.NO_LIQUIDITY
        elif any("price_range" in i for i in issues):
            reason = FailureReason.NO_MOVEMENT
        elif any("tick_rate" in i for i in issues):
            reason = FailureReason.LOW_TICKS
        elif any("no tick" in i for i in issues):
            reason = FailureReason.LOW_TICKS
        else:
            reason = FailureReason.NO_LIQUIDITY

        return ReadinessResult(
            status=ReadinessStatus.NOT_READY,
            reason=reason,
            issues=issues,
            spread=spread, tick_rate=tick_rate,
            price_range=price_range, last_tick_age=last_tick_age,
        )

    return ReadinessResult(
        status=ReadinessStatus.READY,
        spread=spread, tick_rate=tick_rate,
        price_range=price_range, last_tick_age=last_tick_age,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic Liquidity Helper (v2.0)
# ═══════════════════════════════════════════════════════════════════════

def get_liquidity_threshold(match_start_ts: float,
                            now: float | None = None) -> float:
    """Return minimum liquidity threshold based on proximity to match start.

    Before T-60:  $20k (loose — allows early discovery)
    T-60 → T+30:  $10k (loosest — catch late-populating markets)
    After T+30:   $50k (strict — must be genuinely active)
    """
    if now is None:
        now = time.time()
    dt = now - match_start_ts  # seconds since match start
    if dt < -3600:       # more than 60 min before start
        return LIQUIDITY_THRESHOLD_PRE
    elif dt <= 1800:     # T-60 to T+30
        return LIQUIDITY_THRESHOLD_MATCH
    else:                # after T+30
        return LIQUIDITY_THRESHOLD_LIVE
