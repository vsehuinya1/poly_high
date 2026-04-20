"""
Trade Execution Guards — System-wide safety layer.

Non-bypassable validation that runs DIRECTLY before any trade execution.
All sports, all paths, no exceptions.

v1.0 — 2026-04-04  Initial: edge > 0, price band, sanity checks.
v1.1 — 2026-04-05  Fix: use abs(edge) — edge sign encodes direction,
                    not quality. Negative edge = SELL direction is valid.
v2.0 — 2026-04-06  Final hardening: staleness, empty book, edge drift.
v3.0 — 2026-04-19  Per-strategy circuit breaker. Football disabled.
                    Only counts losses (R<0), ignores breakevens.
"""
import logging
import time
from collections import deque

log = logging.getLogger("sports.guards")


# ═══════════════════════════════════════════════════════════════════════
#  Strategy ID Constants — single source of truth for CB keys
# ═══════════════════════════════════════════════════════════════════════

STRAT_ENGINE = "engine"                  # football / NBA paper trades
STRAT_TENNIS_SB = "spread_breakout"      # tennis spread breakout
STRAT_TENNIS_INFLECTION = "inflection"   # tennis inflection strategy
STRAT_CRICKET_MOM = "tick_momentum"      # cricket tick momentum

# v9.2: Training mode — CB records and notifies but NEVER blocks entries
TRAINING_MODE = True


# ═══════════════════════════════════════════════════════════════════════
#  Edge Quality Tracker — rolling metrics for signal monitoring
# ═══════════════════════════════════════════════════════════════════════

class EdgeQualityTracker:
    """Track rolling edge quality metrics.

    Logs EDGE_DISTRIBUTION every N trades for signal quality monitoring.
    """

    def __init__(self, window: int = 50, log_interval: int = 10):
        self._edges: deque[float] = deque(maxlen=window)
        self._count = 0
        self._log_interval = log_interval

    def record(self, edge: float) -> None:
        self._edges.append(abs(edge))
        self._count += 1
        if self._count % self._log_interval == 0 and len(self._edges) >= 5:
            self._log_distribution()

    def _log_distribution(self) -> None:
        edges = sorted(self._edges)
        n = len(edges)
        avg = sum(edges) / n
        p50 = edges[n // 2]
        p90 = edges[int(n * 0.9)]
        log.info(
            "EDGE_DISTRIBUTION | n=%d | avg=%.4f | p50=%.4f | p90=%.4f",
            n, avg, p50, p90,
        )


# Global singleton — shared across all callers
_edge_tracker = EdgeQualityTracker()


# ═══════════════════════════════════════════════════════════════════════
#  Per-Strategy Circuit Breaker (v3.0)
# ═══════════════════════════════════════════════════════════════════════

class StrategyCircuitBreaker:
    """Per-(sport, strategy) circuit breaker.

    Rules:
      - Tracks consecutive LOSSES only (R < 0)
      - Ignores breakevens (R == 0) — no increment, no reset
      - Resets streak to 0 on any WIN (R > 0)
      - Blocks ONLY the specific (sport, strategy) that triggered

    When triggered: new entries for that (sport, strategy) disabled.
    Exits / stop-losses / runners are NEVER affected.
    """

    def __init__(self, loss_streak_limit: int = 5):
        self.loss_streak_limit = loss_streak_limit

        # Per-key tracking
        self._loss_streaks: dict[tuple[str, str], int] = {}
        self._disabled_keys: set[tuple[str, str]] = set()
        self._disable_times: dict[tuple[str, str], float] = {}

        # Telegram spam protection: only notify once per trip
        self._notified_keys: set[tuple[str, str]] = set()

        # Optional async callback for Telegram notification
        # Set via set_telegram_callback() during orchestrator init
        self._tg_callback = None

    def set_telegram_callback(self, callback) -> None:
        """Set async callback: callback(sport, strategy, streak)."""
        self._tg_callback = callback

    def record_trade_outcome(
        self,
        r_multiple: float,
        sport: str,
        strategy: str,
    ) -> None:
        """Record a closed trade's R-multiple for a specific (sport, strategy).

        - R < 0: increment loss streak
        - R > 0: reset loss streak + re-enable if blocked
        - R == 0: ignore completely (breakeven)
        """
        key = (sport, strategy)
        prev_streak = self._loss_streaks.get(key, 0)

        if r_multiple < 0:
            # LOSS — increment streak
            new_streak = prev_streak + 1
            self._loss_streaks[key] = new_streak

            log.info(
                "CB_LOSS | sport=%s | strat=%s | streak=%d/%d | R=%+.3f",
                sport, strategy, new_streak, self.loss_streak_limit, r_multiple,
            )

            # TRIGGER ONLY ON 4→5 TRANSITION (prevents duplicates)
            if prev_streak == self.loss_streak_limit - 1 and new_streak == self.loss_streak_limit:
                self._trip(sport, strategy, new_streak)

        elif r_multiple > 0:
            # WIN — reset streak + re-enable
            if prev_streak >= self.loss_streak_limit:
                log.info(
                    "CB_RESET | sport=%s | strat=%s | reset_by_win R=%+.3f",
                    sport, strategy, r_multiple,
                )

            elif prev_streak > 0:
                log.info(
                    "CB_WIN | sport=%s | strat=%s | streak_reset %d→0 | R=%+.3f",
                    sport, strategy, prev_streak, r_multiple,
                )

            self._loss_streaks[key] = 0
            self._disabled_keys.discard(key)
            self._notified_keys.discard(key)

        # R == 0: intentionally ignored — no increment, no reset

    def record_signal_result(
        self,
        was_blocked: bool,
        sport: str = "",
        strategy: str = "",
    ) -> None:
        """Record whether a signal was blocked. Currently a no-op for v3.0.

        Kept for API compatibility — block ratio trigger removed
        (was causing global false positives).
        """
        pass

    def _trip(self, sport: str, strategy: str, streak: int) -> None:
        """Trip the circuit breaker for a specific (sport, strategy)."""
        key = (sport, strategy)
        self._disabled_keys.add(key)
        self._disable_times[key] = time.time()

        mode_str = "observe_only" if TRAINING_MODE else "live"
        log.warning(
            "CIRCUIT_BREAKER_TRIGGERED | sport=%s | strat=%s | "
            "streak=%d | mode=%s",
            sport, strategy, streak, mode_str,
        )

        # Telegram notification — only ONCE per trip (4→5 guarantees single call)
        if key not in self._notified_keys:
            self._notified_keys.add(key)
            if self._tg_callback:
                try:
                    self._tg_callback(sport, strategy, streak)
                except Exception as e:
                    log.error("CB telegram callback error: %s", e)

    def reset(self, sport: str = "", strategy: str = "") -> None:
        """Manual reset — only for operator intervention.

        If sport+strategy provided: reset that specific key.
        If empty: reset ALL keys.
        """
        if sport and strategy:
            key = (sport, strategy)
            if key in self._disabled_keys:
                log.warning(
                    "CB_MANUAL_RESET | sport=%s | strat=%s | "
                    "was_disabled_for=%.0fs",
                    sport, strategy,
                    time.time() - self._disable_times.get(key, 0),
                )
                self._disabled_keys.discard(key)
                self._notified_keys.discard(key)
                self._loss_streaks[key] = 0
        else:
            # Reset everything
            if self._disabled_keys:
                log.warning(
                    "CB_MANUAL_RESET_ALL | keys=%s",
                    list(self._disabled_keys),
                )
            self._disabled_keys.clear()
            self._notified_keys.clear()
            self._loss_streaks.clear()
            self._disable_times.clear()

    def check(self, sport: str = "", strategy: str = "") -> tuple[bool, str]:
        """Check if trading is disabled for a specific (sport, strategy).

        Returns:
            (can_trade: bool, reason: str)
        """
        if TRAINING_MODE:
            # NEVER block in training mode
            return True, ""

        key = (sport, strategy)
        if key in self._disabled_keys:
            streak = self._loss_streaks.get(key, 0)
            reason = f"BLOCK_CB|sport={sport}|strat={strategy}|streak={streak}"
            return False, reason
        return True, ""

    @property
    def is_disabled(self) -> bool:
        """True if ANY key is currently blocked."""
        return bool(self._disabled_keys)

    @property
    def disabled_keys(self) -> set:
        return set(self._disabled_keys)


# Global singleton (v3.0: per-strategy)
circuit_breaker = StrategyCircuitBreaker()


# ═══════════════════════════════════════════════════════════════════════
#  Core Execution Guard
# ═══════════════════════════════════════════════════════════════════════

# Maximum edge drift allowed between signal and execution
MAX_EDGE_DRIFT = 0.05

# Maximum book age at execution
MAX_BOOK_AGE_EXECUTION_S = 30.0


def validate_trade_execution(
    *,
    edge: float,
    price: float,
    sport: str,
    context: str,
    strategy: str = STRAT_ENGINE,
    book_age: float = 0.0,
    book_bid: float = -1.0,
    book_ask: float = -1.0,
    signal_edge: float | None = None,
) -> tuple[bool, str]:
    """Validate that a trade is safe to execute.

    Must be called DIRECTLY before register_trade / buy / sell.
    ENTRY ONLY — never call this for exits / stop-losses / runners.

    Edge sign convention: positive = BUY, negative = SELL.
    The MAGNITUDE must be > 0 for any trade.

    Args:
        edge:         Computed edge (sign = direction, abs = magnitude).
        price:        Entry price (market mid or limit).
        sport:        Sport identifier (for logging + CB key).
        context:      Human-readable context.
        strategy:     Strategy identifier (for CB key). Default: STRAT_ENGINE.
        book_age:     Seconds since last book update (0 = unknown/skip).
        book_bid:     Best bid price (-1 = unknown/skip).
        book_ask:     Best ask price (-1 = unknown/skip).
        signal_edge:  Edge at signal time (for drift check, None = skip).

    Returns:
        (can_execute: bool, reason: str)
        reason is empty if can_execute is True.
    """
    # 1. Edge magnitude must be non-zero (abs handles signed edges)
    abs_edge = abs(edge) if edge is not None else 0.0
    if edge is None or abs_edge < 0.001:
        log.error(
            "BLOCK_ZERO_EDGE | edge=%.4f | sport=%s | %s",
            edge if edge is not None else -999.0, sport, context,
        )
        return False, "BLOCK_ZERO_EDGE"

    # 2. Edge sanity — catch calculation bugs (|edge| > 100% is nonsense)
    if abs_edge >= 1.0:
        log.error(
            "BLOCK_INSANE_EDGE | edge=%.4f | sport=%s | %s",
            edge, sport, context,
        )
        return False, "BLOCK_INSANE_EDGE"

    # 3. Price must be in valid range
    if price is None or price < 0.01 or price > 0.99:
        log.error(
            "BLOCK_INVALID_PRICE | price=%.4f | sport=%s | %s",
            price if price is not None else -1.0, sport, context,
        )
        return False, "BLOCK_INVALID_PRICE"

    # 4. Book staleness — data must be fresh (skip if age not provided)
    if book_age > 0 and book_age > MAX_BOOK_AGE_EXECUTION_S:
        log.error(
            "BLOCK_STALE_DATA | book_age=%.1fs > %.1fs | sport=%s | %s",
            book_age, MAX_BOOK_AGE_EXECUTION_S, sport, context,
        )
        return False, "BLOCK_STALE_DATA"

    # 5. Empty book — must have a valid BBO (skip if not provided)
    if book_bid >= 0 and book_ask >= 0:
        if book_bid <= 0 and book_ask <= 0:
            log.error(
                "BLOCK_EMPTY_BOOK | bid=%.4f ask=%.4f | sport=%s | %s",
                book_bid, book_ask, sport, context,
            )
            return False, "BLOCK_EMPTY_BOOK"

    # 6. Edge drift — signal edge vs execution edge must not diverge
    if signal_edge is not None:
        drift = abs(abs(signal_edge) - abs(edge))
        if drift > MAX_EDGE_DRIFT:
            log.error(
                "BLOCK_EDGE_DRIFT | signal_edge=%.4f execution_edge=%.4f "
                "delta=%.4f > %.4f | sport=%s | %s",
                signal_edge, edge, drift, MAX_EDGE_DRIFT, sport, context,
            )
            return False, "BLOCK_EDGE_DRIFT"
        # Always log drift for monitoring (even if within limits)
        if drift > 0.02:
            log.info(
                "EDGE_DRIFT | signal_edge=%.4f execution_edge=%.4f "
                "delta=%.4f | sport=%s | %s",
                signal_edge, edge, drift, sport, context,
            )

    # 7. Per-strategy circuit breaker check
    cb_ok, cb_reason = circuit_breaker.check(sport=sport, strategy=strategy)
    if not cb_ok:
        log.warning(
            "BLOCK_CB | sport=%s | strat=%s | %s",
            sport, strategy, context,
        )
        return False, cb_reason

    # ── PASS — record edge for quality tracking ──
    _edge_tracker.record(edge)

    return True, ""
