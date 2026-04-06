"""
Trade Execution Guards — System-wide safety layer.

Non-bypassable validation that runs DIRECTLY before any trade execution.
All sports, all paths, no exceptions.

v1.0 — 2026-04-04  Initial: edge > 0, price band, sanity checks.
v1.1 — 2026-04-05  Fix: use abs(edge) — edge sign encodes direction,
                    not quality. Negative edge = SELL direction is valid.
v2.0 — 2026-04-06  Final hardening: staleness, empty book, edge drift.
"""
import logging
import time
from collections import deque

log = logging.getLogger("sports.guards")


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
#  Global Kill Switch — circuit breaker for systemic failures
# ═══════════════════════════════════════════════════════════════════════

class TradingCircuitBreaker:
    """System-wide trading halt.

    Triggers on:
      1. Last 5 trades all losses
      2. ΣR over last 10 trades < -2.0
      3. Guard blocks > 30% of recent signals

    When triggered: all new entries disabled, exits continue.
    """

    def __init__(
        self,
        loss_streak_limit: int = 5,
        rolling_r_window: int = 10,
        rolling_r_floor: float = -2.0,
        block_ratio_window: int = 100,
        block_ratio_threshold: float = 0.30,
    ):
        self.loss_streak_limit = loss_streak_limit
        self.rolling_r_window = rolling_r_window
        self.rolling_r_floor = rolling_r_floor
        self.block_ratio_window = block_ratio_window
        self.block_ratio_threshold = block_ratio_threshold

        self._r_values: deque[float] = deque(maxlen=rolling_r_window)
        self._recent_outcomes: deque[bool] = deque(maxlen=loss_streak_limit)
        self._signal_results: deque[bool] = deque(maxlen=block_ratio_window)

        self._disabled = False
        self._disable_reason = ""
        self._disable_time = 0.0

    @property
    def is_disabled(self) -> bool:
        return self._disabled

    @property
    def disable_reason(self) -> str:
        return self._disable_reason

    def record_trade_outcome(self, r_multiple: float) -> None:
        """Record a closed trade's R-multiple."""
        is_win = r_multiple > 0
        self._r_values.append(r_multiple)
        self._recent_outcomes.append(is_win)

        # Check loss streak
        if (len(self._recent_outcomes) >= self.loss_streak_limit
                and not any(self._recent_outcomes)):
            self._trip("LOSS_STREAK_5")
            return

        # Check rolling R
        if len(self._r_values) >= self.rolling_r_window:
            total_r = sum(self._r_values)
            if total_r < self.rolling_r_floor:
                self._trip(f"ROLLING_R_{total_r:+.2f}")
                return

    def record_signal_result(self, was_blocked: bool) -> None:
        """Record whether a signal was blocked (True) or passed (False)."""
        self._signal_results.append(was_blocked)

        if len(self._signal_results) >= self.block_ratio_window:
            blocked = sum(1 for b in self._signal_results if b)
            ratio = blocked / len(self._signal_results)
            if ratio > self.block_ratio_threshold:
                self._trip(f"BLOCK_RATIO_{ratio:.0%}")

    def _trip(self, reason: str) -> None:
        if not self._disabled:
            self._disabled = True
            self._disable_reason = reason
            self._disable_time = time.time()
            log.error(
                "TRADING_DISABLED | reason=%s | "
                "last_R=[%s] | time=%s",
                reason,
                ", ".join(f"{r:+.3f}" for r in self._r_values),
                time.strftime("%H:%M:%S"),
            )

    def reset(self) -> None:
        """Manual reset — only for operator intervention."""
        if self._disabled:
            log.warning(
                "TRADING_RE_ENABLED | was_disabled_for=%.0fs | reason_was=%s",
                time.time() - self._disable_time,
                self._disable_reason,
            )
        self._disabled = False
        self._disable_reason = ""

    def check(self) -> tuple[bool, str]:
        """Check if trading is currently disabled.

        Returns:
            (can_trade: bool, reason: str)
        """
        if self._disabled:
            return False, f"BLOCK_CIRCUIT_BREAKER:{self._disable_reason}"
        return True, ""


# Global singleton
circuit_breaker = TradingCircuitBreaker()


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
    book_age: float = 0.0,
    book_bid: float = -1.0,
    book_ask: float = -1.0,
    signal_edge: float | None = None,
) -> tuple[bool, str]:
    """Validate that a trade is safe to execute.

    Must be called DIRECTLY before register_trade / buy / sell.

    Edge sign convention: positive = BUY, negative = SELL.
    The MAGNITUDE must be > 0 for any trade.

    Args:
        edge:         Computed edge (sign = direction, abs = magnitude).
        price:        Entry price (market mid or limit).
        sport:        Sport identifier (for logging).
        context:      Human-readable context.
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

    # 7. Circuit breaker — system-wide trading halt
    cb_ok, cb_reason = circuit_breaker.check()
    if not cb_ok:
        log.warning(
            "BLOCK_CIRCUIT_BREAKER | reason=%s | sport=%s | %s",
            cb_reason, sport, context,
        )
        return False, cb_reason

    # ── PASS — record edge for quality tracking ──
    _edge_tracker.record(edge)

    return True, ""
