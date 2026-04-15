"""
Telegram notifications for sports trading system.
Sends alerts for: system startup, edge signals, paper trades,
game summaries, and session reports.
"""
import asyncio
import logging
import time
from typing import Optional

import aiohttp

from sports.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger("sports.telegram")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


class TelegramNotifier:
    """Async Telegram bot for trading alerts."""

    def __init__(self):
        self.enabled = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_until = 0.0
        self._msg_count = 0
        if not self.enabled:
            log.warning("Telegram not configured — notifications disabled")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send(self, text: str, parse_mode: str = "HTML"):
        """Send a message to the configured chat."""
        if not self.enabled:
            return

        now = time.time()
        if now < self._rate_limit_until:
            return

        try:
            session = await self._get_session()
            async with session.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 429:
                    # Rate limited
                    data = await resp.json()
                    wait = data.get("parameters", {}).get("retry_after", 30)
                    self._rate_limit_until = now + wait
                    log.warning("Telegram rate limited, waiting %ds", wait)
                elif resp.status != 200:
                    log.warning("Telegram send failed: %d", resp.status)
                else:
                    self._msg_count += 1
        except Exception as e:
            log.error("Telegram error: %s", e)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Formatted messages ────────────────────────────────────────

    async def notify_startup(self, n_markets: int, n_links: int, n_tokens: int):
        await self.send(
            f"🚀 <b>Sports System Started (v8.3)</b>\n"
            f"├ Markets: {n_markets}\n"
            f"├ Linked games: {n_links}\n"
            f"├ WS tokens: {n_tokens}\n"
            f"└ Time: {time.strftime('%H:%M UTC')}"
        )

    async def notify_edge_signal(self, direction: str, outcome: str,
                                  edge: float, model_p: float,
                                  market_p: float, game_state: str,
                                  market_title: str):
        emoji = "🟢" if direction == "BUY" else "🔴"
        await self.send(
            f"{emoji} <b>EDGE: {direction} {outcome.upper()}</b>\n"
            f"├ Market: {market_title}\n"
            f"├ Edge: {edge:+.3f} ({abs(edge)*100:.1f}¢)\n"
            f"├ Model: {model_p:.3f} vs Mkt: {market_p:.3f}\n"
            f"└ {game_state}"
        )

    async def notify_paper_entry(self, pos_id: str, direction: str,
                                  outcome: str, price: float,
                                  size: float, edge: float,
                                  game_state: str):
        await self.send(
            f"📝 <b>PAPER ENTRY {pos_id}</b>\n"
            f"├ {direction} {outcome.upper()} @ {price:.3f}\n"
            f"├ Size: ${size:.0f} | Edge: {edge:+.3f}\n"
            f"└ {game_state}"
        )

    async def notify_paper_exit(self, pos_id: str, outcome: str,
                                 entry: float, exit_price: float,
                                 pnl: float, reason: str,
                                 daily_pnl: float):
        emoji = "💰" if pnl > 0 else "💸"
        await self.send(
            f"{emoji} <b>PAPER EXIT {pos_id}</b>\n"
            f"├ {outcome.upper()}: {entry:.3f} → {exit_price:.3f}\n"
            f"├ PnL: ${pnl:+.2f} ({reason})\n"
            f"└ Daily: ${daily_pnl:+.2f}"
        )

    async def notify_game_summary(self, home: str, away: str,
                                    h_score: int, a_score: int,
                                    league: str, trades: int,
                                    game_pnl: float, signals: int):
        result = f"{h_score}-{a_score}"
        emoji = "⚽" if league != "NBA" else "🏀"
        await self.send(
            f"{emoji} <b>GAME OVER: {home} {result} {away}</b>\n"
            f"├ League: {league}\n"
            f"├ Signals: {signals} | Trades: {trades}\n"
            f"└ Game PnL: ${game_pnl:+.2f}"
        )

    async def notify_session_summary(self, summary: dict):
        total = summary.get("total_trades", 0)
        wins = summary.get("wins", 0)
        wr = summary.get("win_rate", 0)
        pnl = summary.get("total_pnl", 0)
        avg_edge = summary.get("avg_edge_at_entry", 0)
        best = summary.get("best_trade", 0)
        worst = summary.get("worst_trade", 0)
        open_pos = summary.get("open_positions", 0)

        emoji = "📊"
        await self.send(
            f"{emoji} <b>SESSION SUMMARY</b>\n"
            f"├ Trades: {total} (W:{wins} L:{total-wins})\n"
            f"├ Win Rate: {wr*100:.1f}%\n"
            f"├ PnL: ${pnl:+.2f}\n"
            f"├ Avg Edge: {avg_edge:.3f}\n"
            f"├ Best: ${best:+.2f} | Worst: ${worst:+.2f}\n"
            f"└ Open: {open_pos}"
        )

    async def notify_status(self, live_football: int, live_nba: int,
                             ws_connected: bool, ws_msgs: int,
                             links: int, trades: int, daily_pnl: float):
        ws_status = "✅" if ws_connected else "❌"
        await self.send(
            f"📡 <b>STATUS</b>\n"
            f"├ Live: ⚽{live_football} 🏀{live_nba}\n"
            f"├ WS: {ws_status} ({ws_msgs} msgs)\n"
            f"├ Links: {links} | Trades: {trades}\n"
            f"└ Daily PnL: ${daily_pnl:+.2f}"
        )

    # ── Cricket Readiness Alerts (v7.0) ───────────────────────────

    async def notify_cricket_ready(self, match_name: str, spread: float,
                                    tick_rate: float, price_range: float):
        await self.send(
            f"🏏 <b>CRICKET READY ✅</b>\n"
            f"Match: {match_name}\n"
            f"Spread: {spread:.4f}\n"
            f"Ticks: {tick_rate:.0f}/min\n"
            f"Movement: {price_range:.4f}\n"
            f"Market: ACTIVE"
        )

    async def notify_cricket_not_ready(self, match_name: str, reason: str,
                                        issues: list[str], spread: float,
                                        tick_rate: float, price_range: float):
        issues_str = "\n".join(f"  • {i}" for i in issues) if issues else "  (none)"
        await self.send(
            f"🏏 <b>CRICKET NOT READY ❌</b>\n"
            f"Match: {match_name}\n"
            f"Reason: {reason}\n"
            f"Issues:\n{issues_str}\n"
            f"Spread: {spread:.4f}\n"
            f"Ticks: {tick_rate:.0f}\n"
            f"Movement: {price_range:.4f}"
        )

    async def notify_cricket_status_change(self, match_name: str,
                                            old_status: str, new_status: str,
                                            spread: float, tick_rate: float,
                                            price_range: float):
        emoji = "✅" if new_status == "READY" else "❌"
        await self.send(
            f"🏏 <b>CRICKET STATUS CHANGE {emoji}</b>\n"
            f"Match: {match_name}\n"
            f"{old_status} → {new_status}\n"
            f"Spread: {spread:.4f}\n"
            f"Ticks: {tick_rate:.0f}/min\n"
            f"Movement: {price_range:.4f}"
        )
