"""
Cricket Live Executor — Limit-Offset Order Placement on Polymarket.

Uses LIMIT orders (NOT market orders) with configurable offset from mid:
    LONG:  limit = mid + offset  (BUY YES token)
    SHORT: limit = mid - offset  (SELL YES token)

Orders are placed as GTC with automatic cancellation after timeout.
Paper trading is NEVER disrupted — live execution runs alongside it.

v1.0 — 2026-04-24
v1.1 — 2026-04-24 — Pre-fill validation safety patch
"""
import csv
import json
import logging
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("cricket.live_executor")

# ─── Safe import of py-clob-client ────────────────────────────────────
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
    import py_clob_client.http_helpers.helpers as _clob_helpers
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False
    log.warning("py-clob-client not installed — cricket live trading disabled")


# ═══════════════════════════════════════════════════════════════════════
#  Order Result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CricketOrderResult:
    """Result of a live order attempt."""
    success: bool
    order_id: str = ""
    filled_size: float = 0.0
    avg_price: float = 0.0
    error: str = ""
    raw_response: dict = None

    def __post_init__(self):
        if self.raw_response is None:
            self.raw_response = {}


# ═══════════════════════════════════════════════════════════════════════
#  Pending Order Tracker — GTC order lifecycle management
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PendingOrder:
    """A GTC order awaiting fill or cancellation."""
    order_id: str
    match_id: str
    token_id: str
    side: str           # "BUY" or "SELL"
    direction: str      # "LONG" or "SHORT"
    limit_price: float
    size_usd: float
    placed_ts: float
    timeout_s: float    # cancel after this many seconds
    context: str = ""   # human-readable description
    entry_mid: float = 0.0   # v1.1: mid price at placement time
    regime: str = ""         # v1.1: regime at placement time (CHAOS/STRUCTURED/UNKNOWN)


# ═══════════════════════════════════════════════════════════════════════
#  Cricket Live Executor
# ═══════════════════════════════════════════════════════════════════════

class CricketLiveExecutor:
    """
    Manages live Polymarket execution for cricket trades.

    - Uses LIMIT orders (GTC) with offset from mid
    - Tracks pending orders and cancels after timeout
    - Separate bankroll from tennis
    - Logs every order to CSV
    """

    # Order timeout — cancel unfilled GTC orders after this
    ORDER_TIMEOUT_S = 60.0

    # ── Pre-fill validation thresholds (v1.1 safety patch) ────────
    FILL_DRIFT_TOLERANCE = 0.02   # max mid drift from entry before cancel
    FILL_MAX_BOOK_AGE_S  = 60.0   # cancel if book older than this
    FILL_MAX_SPREAD      = 0.03   # cancel if spread exceeds this

    def __init__(
        self,
        private_key: str = "",
        funder_address: str = "",
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        proxy_url: str = "",
        initial_bankroll: float = 50.0,
        kelly_pct: float = 0.20,
        min_order_usd: float = 1.0,
        limit_offset: float = 0.01,
        data_dir: Path = Path("sports_data"),
    ):
        self.kelly_pct = kelly_pct
        self.min_order_usd = min_order_usd
        self.limit_offset = limit_offset
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._bankroll_file = self.data_dir / "cricket_bankroll.json"

        # Load persisted bankroll or use initial
        saved = self._load_bankroll()
        if saved is not None:
            self.bankroll = saved
            log.info("CRICKET_EXECUTION | BANKROLL loaded $%.2f", self.bankroll)
        else:
            self.bankroll = initial_bankroll
            log.info("CRICKET_EXECUTION | BANKROLL starting $%.2f", self.bankroll)

        self._client: Optional[ClobClient] = None
        self._csv_writer = None
        self._csv_fh = None
        self._orders_placed = 0
        self._total_spent = 0.0
        self._total_received = 0.0
        self._live_fills: set = set()  # match_ids with confirmed live fills

        # Pending GTC orders awaiting fill or cancellation
        self._pending_orders: dict[str, PendingOrder] = {}  # order_id → PendingOrder
        # match_id → order_id for quick lookup
        self._match_orders: dict[str, str] = {}

        if not HAS_CLOB:
            log.error("CRICKET_EXECUTION | py-clob-client not available")
            return

        if not private_key:
            log.error("CRICKET_EXECUTION | POLY_PRIVATE_KEY not set — disabled")
            return

        try:
            self._client = ClobClient(
                "https://clob.polymarket.com",
                key=private_key,
                chain_id=137,
                signature_type=0,  # EOA wallet
                funder=funder_address if funder_address else None,
            )

            # Use pre-existing API creds if provided, otherwise derive
            if api_key and api_secret and api_passphrase:
                from py_clob_client.clob_types import ApiCreds
                self._client.set_api_creds(ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                ))
                log.info("CRICKET_EXECUTION | using pre-existing API credentials")
            else:
                self._client.set_api_creds(self._client.create_or_derive_api_creds())
                log.info("CRICKET_EXECUTION | derived API credentials")

            log.info(
                "CRICKET_EXECUTION | initialized (bankroll=$%.2f, kelly=%.0f%%, offset=%.3f)",
                self.bankroll, self.kelly_pct * 100, self.limit_offset,
            )

            # Monkey-patch proxy if needed
            if proxy_url and HAS_CLOB:
                try:
                    import httpx
                    proxied = httpx.Client(
                        http2=True,
                        proxy=proxy_url,
                        timeout=10.0,
                    )
                    _clob_helpers._http_client = proxied
                    log.info("CRICKET_EXECUTION | proxy set")
                except Exception as pe:
                    log.error("CRICKET_EXECUTION | proxy failed: %s", pe)

        except Exception as e:
            log.error("CRICKET_EXECUTION | init failed: %s", e)
            self._client = None

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    def record_fill(self, match_id: str):
        """Record that a live fill happened for this match."""
        self._live_fills.add(match_id)

    def has_live_fill(self, match_id: str) -> bool:
        """Check if a match had a confirmed live fill."""
        return match_id in self._live_fills

    @property
    def order_size(self) -> float:
        """Current trade size based on Kelly and bankroll."""
        size = self.bankroll * self.kelly_pct
        return max(self.min_order_usd, round(size, 2))

    # ═══════════════════════════════════════════════════════════════
    #  ENTRY — Limit Order Placement
    # ═══════════════════════════════════════════════════════════════

    def place_order(
        self,
        token_id: str,
        mid: float,
        direction: str,
        match_id: str = "",
        match_info: str = "",
        regime: str = "",
    ) -> CricketOrderResult:
        """
        Place a LIMIT order for the given token.

        LONG:  BUY at mid + offset
        SHORT: SELL at mid - offset

        Orders are GTC — will be cancelled after ORDER_TIMEOUT_S
        if not filled.

        Args:
            token_id: Polymarket condition token ID
            mid: current market mid price
            direction: "LONG" or "SHORT"
            match_id: match identifier for tracking
            match_info: human-readable match description
            regime: market regime at signal time ('CHAOS'/'STRUCTURED'/'UNKNOWN')

        Returns:
            CricketOrderResult with order details
        """
        if not self.is_ready:
            log.info("CRICKET_SKIP_REASON | reason=executor_not_ready | %s", match_info)
            return CricketOrderResult(success=False, error="executor not initialized")

        # Don't double-order the same match
        if match_id and match_id in self._match_orders:
            existing_oid = self._match_orders[match_id]
            if existing_oid in self._pending_orders:
                log.info(
                    "CRICKET_SKIP_REASON | reason=pending_order_exists "
                    "| match=%s | order=%s",
                    match_id, existing_oid,
                )
                return CricketOrderResult(success=False, error="pending order exists")

        size_usd = self.order_size
        if size_usd < self.min_order_usd:
            log.info("CRICKET_SKIP_REASON | reason=size_below_minimum | $%.2f", size_usd)
            return CricketOrderResult(success=False, error=f"size ${size_usd:.2f} below minimum")

        # Calculate limit price and side
        if direction == "LONG":
            limit_price = round(mid + self.limit_offset, 2)
            side = BUY
            side_str = "BUY"
        elif direction == "SHORT":
            limit_price = round(mid - self.limit_offset, 2)
            side = SELL
            side_str = "SELL"
        else:
            log.error("CRICKET_SKIP_REASON | reason=invalid_direction | dir=%s", direction)
            return CricketOrderResult(success=False, error=f"invalid direction: {direction}")

        # Sanity: limit price must be in valid range
        if limit_price < 0.01 or limit_price > 0.99:
            log.error(
                "CRICKET_SKIP_REASON | reason=limit_price_out_of_range "
                "| limit=%.4f | mid=%.4f | dir=%s",
                limit_price, mid, direction,
            )
            return CricketOrderResult(success=False, error=f"limit price {limit_price:.4f} out of range")

        log.info(
            "CRICKET_EXECUTION | %s LIMIT | token=%s...%s | $%.2f @ %.4f "
            "| mid=%.4f offset=%.3f | bankroll=$%.2f | %s",
            side_str, token_id[:8], token_id[-8:],
            size_usd, limit_price, mid, self.limit_offset,
            self.bankroll, match_info,
        )

        try:
            order_args = OrderArgs(
                token_id=token_id,
                amount=size_usd,
                price=limit_price,
                side=side,
            )
            signed = self._client.create_order(order_args)
            resp = self._client.post_order(signed, OrderType.GTC)

            order_id = resp.get("orderID", resp.get("id", ""))
            success = resp.get("success", False) or bool(order_id)

            result = CricketOrderResult(
                success=success,
                order_id=order_id,
                filled_size=size_usd if success else 0.0,
                avg_price=limit_price,
                raw_response=resp,
            )

            if success:
                # Reserve bankroll (will be adjusted on fill/cancel)
                self.bankroll -= size_usd
                self._orders_placed += 1
                self._total_spent += size_usd
                self._save_bankroll()

                # Track pending order for timeout management
                pending = PendingOrder(
                    order_id=order_id,
                    match_id=match_id,
                    token_id=token_id,
                    side=side_str,
                    direction=direction,
                    limit_price=limit_price,
                    size_usd=size_usd,
                    placed_ts=time.time(),
                    timeout_s=self.ORDER_TIMEOUT_S,
                    context=match_info,
                    entry_mid=mid,      # v1.1: snapshot for pre-fill validation
                    regime=regime,       # v1.1: regime at placement time
                )
                self._pending_orders[order_id] = pending
                if match_id:
                    self._match_orders[match_id] = order_id
                    self._live_fills.add(match_id)

                log.info(
                    "CRICKET_EXECUTION | ORDER PLACED | order=%s | %s @ %.4f "
                    "| timeout=%ds | bankroll=$%.2f",
                    order_id, side_str, limit_price,
                    int(self.ORDER_TIMEOUT_S), self.bankroll,
                )
            else:
                log.warning("CRICKET_EXECUTION | ORDER FAILED | resp=%s", resp)

            self._log_order(side_str, token_id, size_usd, limit_price, result, match_info)
            return result

        except Exception as e:
            log.error("CRICKET_EXECUTION | ORDER ERROR | %s | %s", match_info, e)
            result = CricketOrderResult(success=False, error=str(e))
            self._log_order(side_str, token_id, size_usd, limit_price, result, match_info)
            return result

    # ═══════════════════════════════════════════════════════════════
    #  EXIT — Limit Sell for position close
    # ═══════════════════════════════════════════════════════════════

    def place_exit(
        self,
        token_id: str,
        mid: float,
        direction: str,
        size_usd: float,
        match_id: str = "",
        match_info: str = "",
    ) -> CricketOrderResult:
        """
        Place a LIMIT exit order.

        Reverses the entry direction:
            LONG position  → SELL at mid - offset
            SHORT position → BUY at mid + offset

        Args:
            token_id: Polymarket condition token ID
            mid: current market mid price
            direction: original entry direction ("LONG" or "SHORT")
            size_usd: amount to close
            match_id: match identifier
            match_info: human-readable description

        Returns:
            CricketOrderResult
        """
        if not self.is_ready:
            return CricketOrderResult(success=False, error="executor not initialized")

        # Exit is REVERSE of entry
        if direction == "LONG":
            # Closing LONG → SELL at mid - offset
            limit_price = round(mid - self.limit_offset, 2)
            side = SELL
            side_str = "SELL"
        elif direction == "SHORT":
            # Closing SHORT → BUY at mid + offset
            limit_price = round(mid + self.limit_offset, 2)
            side = BUY
            side_str = "BUY"
        else:
            return CricketOrderResult(success=False, error=f"invalid direction: {direction}")

        # Sanity
        if limit_price < 0.01 or limit_price > 0.99:
            log.error(
                "CRICKET_SKIP_REASON | reason=exit_limit_out_of_range "
                "| limit=%.4f | mid=%.4f",
                limit_price, mid,
            )
            return CricketOrderResult(success=False, error="exit limit out of range")

        log.info(
            "CRICKET_EXECUTION | EXIT %s LIMIT | token=%s...%s | $%.2f @ %.4f "
            "| mid=%.4f | %s",
            side_str, token_id[:8], token_id[-8:],
            size_usd, limit_price, mid, match_info,
        )

        try:
            order_args = OrderArgs(
                token_id=token_id,
                amount=size_usd,
                price=limit_price,
                side=side,
            )
            signed = self._client.create_order(order_args)
            resp = self._client.post_order(signed, OrderType.GTC)

            order_id = resp.get("orderID", resp.get("id", ""))
            success = resp.get("success", False) or bool(order_id)

            result = CricketOrderResult(
                success=success,
                order_id=order_id,
                filled_size=size_usd if success else 0.0,
                avg_price=limit_price,
                raw_response=resp,
            )

            if success:
                self._total_received += size_usd
                self._save_bankroll()

                # Track exit order for timeout management
                pending = PendingOrder(
                    order_id=order_id,
                    match_id=match_id,
                    token_id=token_id,
                    side=side_str,
                    direction=f"EXIT_{direction}",
                    limit_price=limit_price,
                    size_usd=size_usd,
                    placed_ts=time.time(),
                    timeout_s=self.ORDER_TIMEOUT_S,
                    context=f"EXIT: {match_info}",
                )
                self._pending_orders[order_id] = pending

                log.info(
                    "CRICKET_EXECUTION | EXIT ORDER PLACED | order=%s | %s @ %.4f "
                    "| timeout=%ds",
                    order_id, side_str, limit_price,
                    int(self.ORDER_TIMEOUT_S),
                )
            else:
                log.warning("CRICKET_EXECUTION | EXIT ORDER FAILED | resp=%s", resp)

            self._log_order(f"EXIT_{side_str}", token_id, size_usd, limit_price, result, match_info)
            return result

        except Exception as e:
            log.error("CRICKET_EXECUTION | EXIT ERROR | %s | %s", match_info, e)
            result = CricketOrderResult(success=False, error=str(e))
            self._log_order(f"EXIT_{side_str}", token_id, size_usd, limit_price, result, match_info)
            return result

    # ═══════════════════════════════════════════════════════════════
    #  GTC Order Management — timeout + cancellation
    # ═══════════════════════════════════════════════════════════════

    def check_pending_orders(self) -> list[str]:
        """Check all pending GTC orders for timeout.

        Cancels any order that has been pending longer than its timeout.
        Must be called periodically from the signal loop.

        Returns:
            List of cancelled order IDs.
        """
        if not self.is_ready:
            return []

        now = time.time()
        to_cancel = []

        for order_id, pending in list(self._pending_orders.items()):
            age = now - pending.placed_ts
            if age >= pending.timeout_s:
                to_cancel.append((order_id, pending))

        cancelled = []
        for order_id, pending in to_cancel:
            success = self._cancel_order(order_id)
            if success:
                # Refund reserved bankroll for entry orders
                if not pending.direction.startswith("EXIT_"):
                    self.bankroll += pending.size_usd
                    self._total_spent -= pending.size_usd
                    self._save_bankroll()

                log.info(
                    "CRICKET_EXECUTION | ORDER CANCELLED (timeout) "
                    "| order=%s | age=%.0fs | %s @ %.4f | %s",
                    order_id, age, pending.side, pending.limit_price,
                    pending.context,
                )
                cancelled.append(order_id)

            # Remove from tracking regardless of cancel success
            self._pending_orders.pop(order_id, None)
            if pending.match_id:
                self._match_orders.pop(pending.match_id, None)
                # Remove fill record if entry was cancelled
                if not pending.direction.startswith("EXIT_"):
                    self._live_fills.discard(pending.match_id)

        return cancelled

    # ═══════════════════════════════════════════════════════════════
    #  Pre-Fill Validation (v1.1 safety patch)
    # ═══════════════════════════════════════════════════════════════

    def validate_pending_fills(
        self,
        books: dict,
        regime: str = "",
    ) -> list[str]:
        """Validate all pending entry orders against current market state.

        Cancels orders that would result in stale/invalid fills:
          1. STALE_FILL     — mid drifted > FILL_DRIFT_TOLERANCE from entry
          2. STALE_BOOK_FILL — book age > FILL_MAX_BOOK_AGE_S
          3. SPREAD_TOO_WIDE — spread > FILL_MAX_SPREAD
          4. REGIME_CHANGED  — market regime flipped since order placement

        Only validates ENTRY orders (not EXIT orders).
        Must be called every tick from the signal loop.

        Args:
            books: {token_id: BookState} from PolymarketFeed
            regime: current market regime (e.g. 'CHAOS', 'STRUCTURED')

        Returns:
            List of cancelled order IDs.
        """
        if not self.is_ready:
            return []

        to_cancel: list[tuple[str, PendingOrder, str]] = []  # (oid, pending, reason)

        for order_id, pending in list(self._pending_orders.items()):
            # Skip exit orders — let them fill regardless
            if pending.direction.startswith("EXIT_"):
                continue

            book = books.get(pending.token_id)
            if not book or book.mid <= 0:
                continue  # no book data — can't validate, let timeout handle it

            current_mid = book.mid
            book_age = time.time() - book.timestamp

            # ── CHECK 1: Price validity (drift from entry mid) ──────
            if pending.direction == "LONG":
                # For LONG: current mid should not have run away above entry+tolerance
                if current_mid > pending.entry_mid + self.FILL_DRIFT_TOLERANCE:
                    to_cancel.append((order_id, pending, "STALE_FILL"))
                    log.info(
                        "CRICKET_SKIP_REASON | reason=STALE_FILL "
                        "| order=%s | dir=LONG "
                        "| entry_mid=%.4f | current_mid=%.4f | drift=+%.4f",
                        order_id, pending.entry_mid, current_mid,
                        current_mid - pending.entry_mid,
                    )
                    continue
            elif pending.direction == "SHORT":
                # For SHORT: current mid should not have dropped below entry-tolerance
                if current_mid < pending.entry_mid - self.FILL_DRIFT_TOLERANCE:
                    to_cancel.append((order_id, pending, "STALE_FILL"))
                    log.info(
                        "CRICKET_SKIP_REASON | reason=STALE_FILL "
                        "| order=%s | dir=SHORT "
                        "| entry_mid=%.4f | current_mid=%.4f | drift=%.4f",
                        order_id, pending.entry_mid, current_mid,
                        pending.entry_mid - current_mid,
                    )
                    continue

            # ── CHECK 2: Book freshness ─────────────────────────────
            if book_age > self.FILL_MAX_BOOK_AGE_S:
                to_cancel.append((order_id, pending, "STALE_BOOK_FILL"))
                log.info(
                    "CRICKET_SKIP_REASON | reason=STALE_BOOK_FILL "
                    "| order=%s | book_age=%.1fs | max=%.1fs",
                    order_id, book_age, self.FILL_MAX_BOOK_AGE_S,
                )
                continue

            # ── CHECK 3: Spread check ──────────────────────────────
            if book.spread > self.FILL_MAX_SPREAD:
                to_cancel.append((order_id, pending, "SPREAD_TOO_WIDE"))
                log.info(
                    "CRICKET_SKIP_REASON | reason=SPREAD_TOO_WIDE "
                    "| order=%s | spread=%.4f | max=%.4f",
                    order_id, book.spread, self.FILL_MAX_SPREAD,
                )
                continue

            # ── CHECK 4: Regime consistency ────────────────────────
            if regime and pending.regime and regime != pending.regime:
                to_cancel.append((order_id, pending, "REGIME_CHANGED"))
                log.info(
                    "CRICKET_SKIP_REASON | reason=REGIME_CHANGED "
                    "| order=%s | entry_regime=%s | current_regime=%s",
                    order_id, pending.regime, regime,
                )
                continue

        # Execute cancellations
        cancelled = []
        for order_id, pending, reason in to_cancel:
            success = self._cancel_order(order_id)
            if success:
                # Refund bankroll for entry orders
                self.bankroll += pending.size_usd
                self._total_spent -= pending.size_usd
                self._save_bankroll()

                log.info(
                    "CRICKET_EXECUTION | ORDER CANCELLED (%s) "
                    "| order=%s | %s @ %.4f | entry_mid=%.4f | %s",
                    reason, order_id, pending.side, pending.limit_price,
                    pending.entry_mid, pending.context,
                )
                cancelled.append(order_id)

            # Remove from tracking
            self._pending_orders.pop(order_id, None)
            if pending.match_id:
                self._match_orders.pop(pending.match_id, None)
                self._live_fills.discard(pending.match_id)

        return cancelled

    @staticmethod
    def classify_regime(match_state) -> str:
        """Classify current match regime from cricket state.

        Lightweight regime classification for pre-fill validation.
        Uses overs + wickets to determine CHAOS vs STRUCTURED.

        Args:
            match_state: CricketState or None

        Returns:
            'CHAOS', 'STRUCTURED', or 'UNKNOWN'
        """
        if match_state is None:
            return "UNKNOWN"
        try:
            overs = getattr(match_state, 'overs', 0) or 0
            wickets_fallen = getattr(match_state, 'wickets', 0) or 0

            # CHAOS: powerplay (overs < 6), death (overs > 15),
            # or heavy wicket fall (>= 5)
            if overs < 6 or overs > 15 or wickets_fallen >= 5:
                return "CHAOS"
            return "STRUCTURED"
        except Exception:
            return "UNKNOWN"

    def _cancel_order(self, order_id: str) -> bool:
        """Cancel a single GTC order on Polymarket."""
        if not self._client:
            return False
        try:
            resp = self._client.cancel(order_id)
            log.info("CRICKET_EXECUTION | CANCEL resp=%s", resp)
            return True
        except Exception as e:
            log.error("CRICKET_EXECUTION | CANCEL ERROR | order=%s | %s", order_id, e)
            return False

    def cancel_all_for_match(self, match_id: str) -> bool:
        """Cancel all pending orders for a specific match."""
        order_id = self._match_orders.get(match_id)
        if not order_id or order_id not in self._pending_orders:
            return False

        pending = self._pending_orders[order_id]
        success = self._cancel_order(order_id)

        if success:
            if not pending.direction.startswith("EXIT_"):
                self.bankroll += pending.size_usd
                self._total_spent -= pending.size_usd
                self._save_bankroll()

        self._pending_orders.pop(order_id, None)
        self._match_orders.pop(match_id, None)
        if not pending.direction.startswith("EXIT_"):
            self._live_fills.discard(match_id)

        return success

    # ═══════════════════════════════════════════════════════════════
    #  Bankroll Management
    # ═══════════════════════════════════════════════════════════════

    def record_exit_pnl(self, entry_size: float, exit_price: float, entry_price: float):
        """
        Update bankroll with actual PnL from a closed position.
        Called when the exit manager closes a trade.
        """
        if entry_price > 0:
            shares = entry_size / entry_price
            proceeds = shares * exit_price
            pnl = proceeds - entry_size
            self.bankroll += pnl
            self._save_bankroll()
            log.info(
                "CRICKET_EXECUTION | PNL | entry=$%.2f@%.4f exit@%.4f "
                "| shares=%.2f pnl=$%.2f | bankroll=$%.2f",
                entry_size, entry_price, exit_price, shares, pnl, self.bankroll,
            )

    # ═══════════════════════════════════════════════════════════════
    #  Logging
    # ═══════════════════════════════════════════════════════════════

    def _log_order(self, side: str, token_id: str, size: float,
                   price: float, result: CricketOrderResult, match_info: str):
        """Log order to CSV."""
        if self._csv_writer is None:
            today = time.strftime("%Y%m%d")
            path = self.data_dir / f"cricket_live_orders_{today}.csv"
            write_header = not path.exists()
            self._csv_fh = open(path, "a", newline="", buffering=1)
            self._csv_writer = csv.writer(self._csv_fh)
            if write_header:
                self._csv_writer.writerow([
                    "timestamp", "side", "token_id", "size_usd", "limit_price",
                    "success", "order_id", "filled_size", "error",
                    "bankroll_after", "match_info",
                ])

        self._csv_writer.writerow([
            time.time(), side, token_id, f"{size:.2f}", f"{price:.4f}",
            "1" if result.success else "0", result.order_id,
            f"{result.filled_size:.2f}", result.error,
            f"{self.bankroll:.2f}", match_info,
        ])

    def _save_bankroll(self):
        """Persist bankroll to disk for restart recovery."""
        try:
            with open(self._bankroll_file, "w") as f:
                json.dump({"bankroll": self.bankroll, "updated": time.time()}, f)
        except Exception:
            pass

    def _load_bankroll(self) -> Optional[float]:
        """Load persisted bankroll if available."""
        try:
            if self._bankroll_file.exists():
                with open(self._bankroll_file) as f:
                    data = json.load(f)
                return data.get("bankroll")
        except Exception:
            pass
        return None

    def status_line(self) -> str:
        """Return a short status string for the status printer."""
        if not self.is_ready:
            return "CRICKET LIVE: OFF (no credentials)"
        pending = len(self._pending_orders)
        return (
            f"CRICKET LIVE: ON | bankroll=${self.bankroll:.2f} | "
            f"size=${self.order_size:.2f} | orders={self._orders_placed} | "
            f"pending={pending}"
        )

    def close(self):
        """Clean up — cancel all pending orders and close CSV handle."""
        # Cancel all pending orders on shutdown
        for order_id in list(self._pending_orders.keys()):
            self._cancel_order(order_id)
        self._pending_orders.clear()
        self._match_orders.clear()

        self._save_bankroll()
        if self._csv_fh:
            self._csv_fh.close()
