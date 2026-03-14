"""
Tennis Live Executor — real CLOB order placement on Polymarket.

Wraps py-clob-client for BUY market orders on tennis signals.
Handles bankroll tracking, Kelly sizing, and order logging.

v4.4 — 2026-03-11
"""
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("tennis.live_executor")

# ─── Safe import of py-clob-client ────────────────────────────────────
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import MarketOrderArgs, OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
    import py_clob_client.http_helpers.helpers as _clob_helpers
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False
    log.warning("py-clob-client not installed — live trading disabled")


@dataclass
class OrderResult:
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


class LiveExecutor:
    """
    Manages live Polymarket execution for tennis trades.

    - Initializes CLOB client from private key + API creds
    - Tracks bankroll with compounding
    - Places market BUY orders (FOK)
    - Logs every order to CSV
    """

    def __init__(
        self,
        private_key: str = "",
        funder_address: str = "",
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        proxy_url: str = "",
        initial_bankroll: float = 24.0,
        kelly_pct: float = 0.30,
        min_order_usd: float = 1.0,
        data_dir: Path = Path("sports_data"),
    ):
        self.kelly_pct = kelly_pct
        self.min_order_usd = min_order_usd
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._bankroll_file = self.data_dir / "tennis_bankroll.json"

        # Load persisted bankroll or use initial
        saved = self._load_bankroll()
        if saved is not None:
            self.bankroll = saved
            log.info("BANKROLL: loaded persisted $%.2f", self.bankroll)
        else:
            self.bankroll = initial_bankroll
            log.info("BANKROLL: starting fresh $%.2f", self.bankroll)

        self._client: Optional[ClobClient] = None
        self._csv_writer = None
        self._csv_fh = None
        self._orders_placed = 0
        self._total_spent = 0.0
        self._total_received = 0.0
        self._live_fills: set = set()  # match_ids with confirmed live fills

        if not HAS_CLOB:
            log.error("py-clob-client not available — cannot trade live")
            return

        if not private_key:
            log.error("POLY_PRIVATE_KEY not set — live trading disabled")
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
                log.info("LIVE EXECUTOR: using pre-existing API credentials")
            else:
                self._client.set_api_creds(self._client.create_or_derive_api_creds())
                log.info("LIVE EXECUTOR: derived API credentials from private key")

            log.info("LIVE EXECUTOR: initialized (bankroll=$%.2f, kelly=%.0f%%)",
                     self.bankroll, self.kelly_pct * 100)

            # Monkey-patch py-clob-client's httpx client with proxy
            if proxy_url and HAS_CLOB:
                try:
                    import httpx
                    proxied = httpx.Client(
                        http2=True,
                        proxy=proxy_url,
                        timeout=10.0,
                    )
                    _clob_helpers._http_client = proxied
                    log.info("LIVE EXECUTOR: proxy set → %s", proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url)
                except Exception as pe:
                    log.error("LIVE EXECUTOR: failed to set proxy: %s", pe)

        except Exception as e:
            log.error("LIVE EXECUTOR: failed to init ClobClient: %s", e)
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

    def buy(self, token_id: str, price: float, match_info: str = "") -> OrderResult:
        """
        Place a BUY market order for the given token.

        Args:
            token_id: Polymarket condition token ID
            price: current best ask / market price
            match_info: human-readable match description for logging

        Returns:
            OrderResult with fill details
        """
        if not self.is_ready:
            return OrderResult(success=False, error="executor not initialized")

        size_usd = self.order_size
        if size_usd < self.min_order_usd:
            return OrderResult(success=False, error=f"size ${size_usd:.2f} below minimum")

        log.info("LIVE BUY | token=%s...%s | $%.2f @ %.4f | bankroll=$%.2f | %s",
                 token_id[:8], token_id[-8:], size_usd, price, self.bankroll, match_info)

        try:
            # Use market order (FOK = fill or kill)
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=size_usd,
                side=BUY,
                price=price,
            )
            signed = self._client.create_market_order(order_args)
            resp = self._client.post_order(signed, OrderType.FOK)

            order_id = resp.get("orderID", resp.get("id", ""))
            success = resp.get("success", False) or bool(order_id)

            result = OrderResult(
                success=success,
                order_id=order_id,
                filled_size=size_usd if success else 0.0,
                avg_price=price,
                raw_response=resp,
            )

            if success:
                self.bankroll -= size_usd
                self._orders_placed += 1
                self._total_spent += size_usd
                self._save_bankroll()
                log.info("LIVE BUY FILLED | order=%s | $%.2f @ %.4f | bankroll=$%.2f",
                         order_id, size_usd, price, self.bankroll)
            else:
                log.warning("LIVE BUY FAILED | resp=%s", resp)

            self._log_order("BUY", token_id, size_usd, price, result, match_info)
            return result

        except Exception as e:
            log.error("LIVE BUY ERROR | %s | %s", match_info, e)
            result = OrderResult(success=False, error=str(e))
            self._log_order("BUY", token_id, size_usd, price, result, match_info)
            return result

    def sell(self, token_id: str, size_usd: float, price: float,
             match_info: str = "") -> OrderResult:
        """
        Place a SELL market order to close a position.

        Args:
            token_id: Polymarket condition token ID
            size_usd: amount to sell (in USD terms)
            price: current best bid / market price
            match_info: human-readable match description

        Returns:
            OrderResult with fill details
        """
        if not self.is_ready:
            return OrderResult(success=False, error="executor not initialized")

        log.info("LIVE SELL | token=%s...%s | $%.2f @ %.4f | %s",
                 token_id[:8], token_id[-8:], size_usd, price, match_info)

        try:
            # Sell at market (FOK)
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=size_usd,
                side=SELL,
                price=price,
            )
            signed = self._client.create_market_order(order_args)
            resp = self._client.post_order(signed, OrderType.FOK)

            order_id = resp.get("orderID", resp.get("id", ""))
            success = resp.get("success", False) or bool(order_id)

            result = OrderResult(
                success=success,
                order_id=order_id,
                filled_size=size_usd if success else 0.0,
                avg_price=price,
                raw_response=resp,
            )

            if success:
                proceeds = size_usd  # approximate — actual may differ
                self.bankroll += proceeds
                self._total_received += proceeds
                self._save_bankroll()
                log.info("LIVE SELL FILLED | order=%s | $%.2f @ %.4f | bankroll=$%.2f",
                         order_id, size_usd, price, self.bankroll)
            else:
                log.warning("LIVE SELL FAILED | resp=%s", resp)

            self._log_order("SELL", token_id, size_usd, price, result, match_info)
            return result

        except Exception as e:
            log.error("LIVE SELL ERROR | %s | %s", match_info, e)
            result = OrderResult(success=False, error=str(e))
            self._log_order("SELL", token_id, size_usd, price, result, match_info)
            return result

    def record_exit_pnl(self, entry_size: float, exit_price: float, entry_price: float):
        """
        Update bankroll with actual PnL from a closed position.
        Called when the exit manager closes a trade.
        """
        # Shares bought = entry_size / entry_price
        # Proceeds = shares * exit_price
        if entry_price > 0:
            shares = entry_size / entry_price
            proceeds = shares * exit_price
            pnl = proceeds - entry_size
            self.bankroll += pnl  # note: entry_size was already subtracted on buy
            self._save_bankroll()
            log.info("LIVE PNL | entry=$%.2f@%.4f exit@%.4f | shares=%.2f pnl=$%.2f | bankroll=$%.2f",
                     entry_size, entry_price, exit_price, shares, pnl, self.bankroll)

    def _log_order(self, side: str, token_id: str, size: float,
                   price: float, result: OrderResult, match_info: str):
        """Log order to CSV."""
        if self._csv_writer is None:
            today = time.strftime("%Y%m%d")
            path = self.data_dir / f"tennis_live_orders_{today}.csv"
            write_header = not path.exists()
            self._csv_fh = open(path, "a", newline="", buffering=1)
            self._csv_writer = csv.writer(self._csv_fh)
            if write_header:
                self._csv_writer.writerow([
                    "timestamp", "side", "token_id", "size_usd", "price",
                    "success", "order_id", "filled_size", "error",
                    "bankroll_after", "match_info",
                ])

        self._csv_writer.writerow([
            time.time(), side, token_id, f"{size:.2f}", f"{price:.4f}",
            "1" if result.success else "0", result.order_id,
            f"{result.filled_size:.2f}", result.error,
            f"{self.bankroll:.2f}", match_info,
        ])

    def status_line(self) -> str:
        """Return a short status string for the status printer."""
        if not self.is_ready:
            return "LIVE: OFF (no credentials)"
        return (f"LIVE: ON | bankroll=${self.bankroll:.2f} | "
                f"size=${self.order_size:.2f} | orders={self._orders_placed}")

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

    def close(self):
        """Clean up CSV handle."""
        self._save_bankroll()
        if self._csv_fh:
            self._csv_fh.close()
