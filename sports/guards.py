"""
Trade Execution Guards — System-wide safety layer.

Non-bypassable validation that runs DIRECTLY before any trade execution.
All sports, all paths, no exceptions.

v1.0 — 2026-04-04  Initial: edge > 0, price band, sanity checks.
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

    Args:
        edge:    Computed edge (must be > 0 for execution).
        price:   Entry price (market mid).
        sport:   Sport identifier (for logging).
        context: Human-readable context (match title, signal type, etc.).

    Returns:
        (can_execute: bool, reason: str)
        reason is empty if can_execute is True.
    """
    # 1. Edge must be strictly positive
    if edge is None or edge <= 0:
        log.error(
            "BLOCK_NEGATIVE_EDGE | edge=%.4f | sport=%s | %s",
            edge if edge is not None else -999.0, sport, context,
        )
        return False, "BLOCK_NEGATIVE_EDGE"

    # 2. Edge sanity — catch calculation bugs (edge > 100% is nonsense)
    if edge >= 1.0:
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
