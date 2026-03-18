"""
Replay Backtest Engine — test strategy parameter changes against historical data.

Usage:
    python replay_backtest.py                    # Compare current vs proposed params
    python replay_backtest.py --min-hold 90      # Test specific min-hold
    python replay_backtest.py --edge 0.15        # Test specific edge threshold
    python replay_backtest.py --timeout 600      # Test specific timeout (seconds)
    python replay_backtest.py --price-floor 0.30 # Tennis price floor

Reads all paper_trades and tennis_trade_lifecycle CSVs from sports_data/,
replays each trade through the given filters, and shows a side-by-side comparison.
"""
import argparse
import csv
import glob
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TradeResult:
    sport: str
    game_id: str
    entry_price: float
    exit_price: float
    edge: float
    pnl: float
    exit_reason: str
    hold_s: float
    direction: str
    game_state: str


@dataclass
class TennisResult:
    match_id: str
    player: str
    entry_price: float
    exit_price: float
    edge: float
    R_multiple: float
    exit_reason: str
    duration_s: float
    mfe: float
    mae: float
    spread: float


def load_paper_trades(data_dir: str = "sports_data") -> list[TradeResult]:
    """Load all NBA/football paper trades."""
    trades = []
    for f in sorted(glob.glob(os.path.join(data_dir, "paper_trades_*.csv"))):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if row.get("event") != "EXIT":
                    continue
                sport = row.get("sport", "")
                if not sport:
                    # Infer from game_id format
                    gid = row.get("game_id", "")
                    if gid.startswith("00"):
                        sport = "nba"
                    elif gid.startswith("4"):
                        sport = "football"
                    else:
                        sport = "unknown"
                trades.append(TradeResult(
                    sport=sport,
                    game_id=row.get("game_id", ""),
                    entry_price=float(row.get("entry_price", 0) or 0),
                    exit_price=float(row.get("exit_price", 0) or 0),
                    edge=abs(float(row.get("entry_edge", 0) or 0)),
                    pnl=float(row.get("pnl", 0) or 0),
                    exit_reason=row.get("exit_reason", ""),
                    hold_s=float(row.get("hold_duration_s", 0) or 0),
                    direction=row.get("direction", "SELL"),
                    game_state=row.get("game_state", ""),
                ))
    return trades


def load_tennis_trades(data_dir: str = "sports_data") -> list[TennisResult]:
    """Load all tennis trade lifecycle data."""
    trades = []
    for f in sorted(glob.glob(os.path.join(data_dir, "tennis_trade_lifecycle_*.csv"))):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if not row.get("match_id"):
                    continue
                trades.append(TennisResult(
                    match_id=row.get("match_id", ""),
                    player=row.get("player", ""),
                    entry_price=float(row.get("entry_price", 0) or 0),
                    exit_price=float(row.get("exit_price", 0) or 0),
                    edge=float(row.get("edge_entry", "").replace("+", "") or 0),
                    R_multiple=float(row.get("R_multiple", "").replace("+", "") or 0),
                    exit_reason=row.get("exit_reason", ""),
                    duration_s=float(row.get("duration_seconds", 0) or 0),
                    mfe=float(row.get("mfe", 0) or 0),
                    mae=float(row.get("mae", 0) or 0),
                    spread=float(row.get("spread", 0) or 0),
                ))
    return trades


def replay_paper(trades: list[TradeResult], min_hold: float, edge_thresh: float,
                 timeout: float, sport_filter: str = "") -> dict:
    """Replay paper trades with new parameters."""
    kept = []
    filtered = []

    for t in trades:
        if sport_filter and t.sport != sport_filter:
            continue

        # Apply filters
        reasons = []
        if t.edge < edge_thresh:
            reasons.append(f"edge {t.edge:.3f} < {edge_thresh}")
        # Check if trade would have been exited differently with new min_hold
        # If exit_reason is edge_flip and hold < min_hold, it would have stayed open
        would_survive_hold = (t.exit_reason == "edge_flip" and t.hold_s < min_hold)
        # If timeout is shorter and hold > timeout, exit earlier
        would_timeout_earlier = (t.hold_s > timeout and t.exit_reason == "timeout")

        if reasons:
            filtered.append((t, reasons))
            continue

        # Adjust PnL for modified parameters
        adj_pnl = t.pnl
        if would_survive_hold:
            # Would have held longer — we don't know final outcome, mark as "survived"
            adj_pnl = 0.0  # conservative: assume breakeven
        if would_timeout_earlier:
            # Would have exited earlier — scale PnL by time ratio
            ratio = timeout / t.hold_s if t.hold_s > 0 else 1
            adj_pnl = t.pnl * ratio

        kept.append((t, adj_pnl))

    wins = sum(1 for _, p in kept if p > 0)
    losses = sum(1 for _, p in kept if p <= 0)
    total_pnl = sum(p for _, p in kept)
    return {
        "total": len(kept),
        "wins": wins,
        "losses": losses,
        "wr": wins / len(kept) * 100 if kept else 0,
        "pnl": total_pnl,
        "filtered_out": len(filtered),
        "avg_pnl": total_pnl / len(kept) if kept else 0,
        "edge_flip_survived": sum(1 for t, _ in kept if t.exit_reason == "edge_flip" and t.hold_s < min_hold),
    }


def replay_tennis(trades: list[TennisResult], price_floor: float,
                   min_edge: float, max_spread: float = 1.0) -> dict:
    """Replay tennis trades with new parameters."""
    kept = []
    filtered = []

    for t in trades:
        reasons = []
        if t.entry_price < price_floor:
            reasons.append(f"price {t.entry_price:.2f} < floor {price_floor}")
        if t.edge < min_edge:
            reasons.append(f"edge {t.edge:.3f} < {min_edge}")
        if t.spread > max_spread:
            reasons.append(f"spread {t.spread:.2f} > {max_spread}")

        if reasons:
            filtered.append((t, reasons))
            continue

        kept.append(t)

    wins = sum(1 for t in kept if t.R_multiple > 0)
    losses = sum(1 for t in kept if t.R_multiple <= 0)
    total_r = sum(t.R_multiple for t in kept)
    return {
        "total": len(kept),
        "wins": wins,
        "losses": losses,
        "wr": wins / len(kept) * 100 if kept else 0,
        "total_R": total_r,
        "avg_R": total_r / len(kept) if kept else 0,
        "filtered_out": len(filtered),
        "best": max((t.R_multiple for t in kept), default=0),
        "worst": min((t.R_multiple for t in kept), default=0),
        "filtered_trades": [(t.player, t.entry_price, t.R_multiple, r)
                            for t, r in filtered],
    }


def fmt(d: dict, sport: str = "") -> str:
    """Format results dict for display."""
    lines = []
    if "total_R" in d:  # Tennis
        lines.append(f"  Trades: {d['total']} | Wins: {d['wins']} | WR: {d['wr']:.1f}%")
        lines.append(f"  ΣR: {d['total_R']:+.4f} | Avg R: {d['avg_R']:+.4f}")
        lines.append(f"  Best: {d['best']:+.4f} | Worst: {d['worst']:+.4f}")
        lines.append(f"  Filtered out: {d['filtered_out']} trades")
        if d.get("filtered_trades"):
            for player, price, r, reasons in d["filtered_trades"][:5]:
                lines.append(f"    ✂ {player} (entry={price:.2f}, R={r:+.4f}) — {', '.join(reasons)}")
    else:  # NBA/Football
        lines.append(f"  Trades: {d['total']} | Wins: {d['wins']} | WR: {d['wr']:.1f}%")
        lines.append(f"  PnL: ${d['pnl']:.2f} | Avg: ${d['avg_pnl']:.2f}/trade")
        lines.append(f"  Filtered out: {d['filtered_out']} trades")
        if d.get("edge_flip_survived", 0) > 0:
            lines.append(f"  Edge-flip trades that would survive new hold: {d['edge_flip_survived']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Replay backtest engine")
    parser.add_argument("--data-dir", default="sports_data", help="Data directory")
    parser.add_argument("--min-hold", type=float, default=None, help="Min hold seconds (NBA)")
    parser.add_argument("--edge", type=float, default=None, help="Min edge threshold")
    parser.add_argument("--timeout", type=float, default=None, help="Timeout seconds")
    parser.add_argument("--price-floor", type=float, default=None, help="Tennis price floor")
    parser.add_argument("--tennis-edge", type=float, default=None, help="Tennis min edge")
    args = parser.parse_args()

    # Load data
    paper = load_paper_trades(args.data_dir)
    tennis = load_tennis_trades(args.data_dir)

    nba = [t for t in paper if t.sport == "nba"]
    football = [t for t in paper if t.sport == "football"]

    print(f"\n{'='*60}")
    print(f"  REPLAY BACKTEST ENGINE")
    print(f"  Data: {len(paper)} NBA/FB trades, {len(tennis)} tennis trades")
    print(f"{'='*60}\n")

    # ── Current params (baseline) ──
    curr_hold_nba = 5
    curr_hold_fb = 5
    curr_edge = 0.10
    curr_fb_edge = 0.10
    curr_timeout_nba = 1800
    curr_timeout_fb = 900
    curr_t_floor = 0.05
    curr_t_edge = 0.08

    # ── Proposed params ──
    new_hold_nba = args.min_hold or 90
    new_hold_fb = 30
    new_edge = args.edge or 0.10
    new_fb_edge = args.edge or 0.15
    new_timeout_nba = args.timeout or 1200
    new_timeout_fb = args.timeout or 600
    new_t_floor = args.price_floor or 0.30
    new_t_edge = args.tennis_edge or 0.08

    # ── NBA ──
    if nba:
        print("🏀 NBA")
        print(f"  Current: hold={curr_hold_nba}s, edge={curr_edge}, timeout={curr_timeout_nba}s")
        old = replay_paper(nba, curr_hold_nba, curr_edge, curr_timeout_nba)
        print(fmt(old))
        print()
        print(f"  Proposed: hold={new_hold_nba}s, edge={new_edge}, timeout={new_timeout_nba}s")
        new = replay_paper(nba, new_hold_nba, new_edge, new_timeout_nba)
        print(fmt(new))
        pnl_delta = new["pnl"] - old["pnl"]
        print(f"\n  → PnL delta: ${pnl_delta:+.2f} | WR: {old['wr']:.0f}% → {new['wr']:.0f}%")
    else:
        print("🏀 NBA: no trades found")

    print()

    # ── Football ──
    if football:
        print("⚽ Football")
        print(f"  Current: hold={curr_hold_fb}s, edge={curr_fb_edge}, timeout={curr_timeout_fb}s")
        old = replay_paper(football, curr_hold_fb, curr_fb_edge, curr_timeout_fb)
        print(fmt(old))
        print()
        print(f"  Proposed: hold={new_hold_fb}s, edge={new_fb_edge}, timeout={new_timeout_fb}s")
        new = replay_paper(football, new_hold_fb, new_fb_edge, new_timeout_fb)
        print(fmt(new))
        pnl_delta = new["pnl"] - old["pnl"]
        print(f"\n  → PnL delta: ${pnl_delta:+.2f} | WR: {old['wr']:.0f}% → {new['wr']:.0f}%")
    else:
        print("⚽ Football: no trades found")

    print()

    # ── Tennis ──
    if tennis:
        print("🎾 Tennis")
        print(f"  Current: floor={curr_t_floor}, edge={curr_t_edge}")
        old = replay_tennis(tennis, curr_t_floor, curr_t_edge)
        print(fmt(old))
        print()
        print(f"  Proposed: floor={new_t_floor}, edge={new_t_edge}")
        new = replay_tennis(tennis, new_t_floor, new_t_edge)
        print(fmt(new))
        r_delta = new["total_R"] - old["total_R"]
        print(f"\n  → ΣR delta: {r_delta:+.4f} | WR: {old['wr']:.0f}% → {new['wr']:.0f}%")
        if new["filtered_out"] > old["filtered_out"]:
            print(f"  → Would have filtered {new['filtered_out'] - old['filtered_out']} more bad trades")
    else:
        print("🎾 Tennis: no trades found")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
