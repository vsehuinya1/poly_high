"""
Trade Execution Guards — System-wide safety layer.

Non-bypassable validation that runs DIRECTLY before any trade execution.
All sports, all paths, no exceptions.

v1.0 — 2026-04-04  Initial: edge > 0, price band, sanity checks.
v1.1 — 2026-04-05  Fix: use abs(edge) — edge sign encodes direction,
                    not quality. Negative edge = SELL direction is valid.
"""
import logging

log = logging.getLogger("sports.guards")


def validate_trade_execution(
    *,
    edge: float,
    price: float,
    sport: str,
    context: str,
) -> tuple[bool, str]:
    """Validate that a trade is safe to execute.

    Must be called DIRECTLY before register_trade / buy / sell.

    Edge sign convention: positive = BUY, negative = SELL.
    The MAGNITUDE must be > 0 for any trade.

    Args:
        edge:    Computed edge (sign = direction, abs = magnitude).
        price:   Entry price (market mid).
        sport:   Sport identifier (for logging).
        context: Human-readable context (match title, signal type, etc.).

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

    return True, ""
