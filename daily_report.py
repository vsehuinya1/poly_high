"""
Daily Report — sends automated Telegram summary of today's trading performance.

Run via cron at midnight:
    0 0 * * * cd /root/poly_high_sports && python3 daily_report.py

Or run manually:
    python daily_report.py                # today's report
    python daily_report.py --date 20260318  # specific date
"""
import argparse
import csv
import glob
import os
import sys
import requests
from collections import defaultdict
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send_telegram(text: str):
    """Send HTML message via Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("No Telegram creds — printing instead:")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }, timeout=10)


def load_paper_trades(date_str: str, data_dir: str = "sports_data") -> list[dict]:
    """Load paper trades for a specific date."""
    path = os.path.join(data_dir, f"paper_trades_{date_str}.csv")
    if not os.path.exists(path):
        return []
    trades = []
    with open(path) as f:
        for row in csv.DictReader(f):
            trades.append(row)
    return trades


def load_tennis_trades(date_str: str, data_dir: str = "sports_data") -> list[dict]:
    """Load tennis lifecycle trades for a specific date."""
    path = os.path.join(data_dir, f"tennis_trade_lifecycle_{date_str}.csv")
    if not os.path.exists(path):
        return []
    trades = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("match_id"):
                trades.append(row)
    return trades


def classify_sport(row: dict) -> str:
    """Classify sport from paper trade row."""
    sport = row.get("sport", "")
    if sport:
        return sport
    gid = row.get("game_id", "")
    if gid.startswith("00"):
        return "nba"
    elif gid.startswith("4"):
        return "football"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Date YYYYMMDD (default: today)")
    parser.add_argument("--data-dir", default="sports_data")
    args = parser.parse_args()

    if args.date:
        date_str = args.date
    else:
        date_str = datetime.utcnow().strftime("%Y%m%d")

    paper = load_paper_trades(date_str, args.data_dir)
    tennis = load_tennis_trades(date_str, args.data_dir)

    # Parse paper trades by sport
    sports = defaultdict(lambda: {"entries": 0, "exits": 0, "wins": 0, "pnl": 0.0,
                                   "reasons": defaultdict(int), "blocks": 0})

    for row in paper:
        sport = classify_sport(row)
        event = row.get("event", "")
        if event == "ENTRY":
            sports[sport]["entries"] += 1
        elif event == "EXIT":
            sports[sport]["exits"] += 1
            pnl = float(row.get("pnl", 0) or 0)
            sports[sport]["pnl"] += pnl
            if pnl > 0:
                sports[sport]["wins"] += 1
            reason = row.get("exit_reason", "?")
            sports[sport]["reasons"][reason] += 1

    # Tennis stats
    t_wins = 0
    t_losses = 0
    t_total_r = 0.0
    t_best = ("", 0.0)
    t_worst = ("", 0.0)
    for t in tennis:
        r = float(t.get("R_multiple", "0").replace("+", "") or 0)
        t_total_r += r
        if r > 0:
            t_wins += 1
        else:
            t_losses += 1
        if r > t_best[1]:
            t_best = (t.get("player", "?"), r)
        if r < t_worst[1]:
            t_worst = (t.get("player", "?"), r)

    # Build message
    lines = [f"📊 <b>Daily Report — {date_str}</b>"]
    lines.append("")

    for sport in ["nba", "football"]:
        s = sports.get(sport)
        if not s or s["exits"] == 0:
            lines.append(f"{'🏀' if sport == 'nba' else '⚽'} <b>{sport.upper()}</b>: no trades")
            continue
        wr = s["wins"] / s["exits"] * 100 if s["exits"] else 0
        emoji = "🏀" if sport == "nba" else "⚽"
        lines.append(f"{emoji} <b>{sport.upper()}</b>: {s['exits']} trades, "
                      f"{s['wins']}W-{s['exits']-s['wins']}L, "
                      f"WR {wr:.0f}%, PnL ${s['pnl']:.2f}")
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(s["reasons"].items()))
        lines.append(f"  Exits: {reasons}")

    if tennis:
        wr = t_wins / (t_wins + t_losses) * 100 if (t_wins + t_losses) else 0
        lines.append(f"🎾 <b>Tennis</b>: {t_wins + t_losses} trades, "
                      f"{t_wins}W-{t_losses}L, "
                      f"WR {wr:.0f}%, ΣR {t_total_r:+.4f}")
        if t_best[0]:
            lines.append(f"  Best: {t_best[0]} R={t_best[1]:+.4f}")
        if t_worst[0]:
            lines.append(f"  Worst: {t_worst[0]} R={t_worst[1]:+.4f}")
    else:
        lines.append("🎾 <b>Tennis</b>: no trades")

    # Overall
    total_pnl = sum(s["pnl"] for s in sports.values())
    lines.append("")
    lines.append(f"💰 <b>Total PnL</b>: ${total_pnl:+.2f} (paper)")

    msg = "\n".join(lines)
    send_telegram(msg)
    print(msg)


if __name__ == "__main__":
    main()
