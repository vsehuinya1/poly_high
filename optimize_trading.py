import json
import os
import statistics
import datetime

def solve():
    path = "/Users/MartinOile/Desktop/poly_high/sports_data/extracted_trades_ticks.json"
    if not os.path.exists(path):
        print(json.dumps({"error": f"File not found: {path}"}))
        return

    with open(path, 'r') as f:
        data = json.load(f)

    # Flatten trades
    all_trades = []
    trades_obj = data.get("trades", {})
    for key, trades_list in trades_obj.items():
        if isinstance(trades_list, list):
            all_trades.extend(trades_list)

    total_trades_before = len(all_trades)
    
    # Process signal_timestamp_raw
    for t in all_trades:
        ts = t.get("signal_timestamp_raw")
        if ts is None:
            # Try parsing signal_timestamp
            st = t.get("signal_timestamp")
            if st:
                try:
                    if st.endswith('Z'): st = st[:-1]
                    t["signal_timestamp_raw"] = datetime.datetime.fromisoformat(st).timestamp()
                except:
                    t["signal_timestamp_raw"] = 0
            else:
                t["signal_timestamp_raw"] = 0

    # 1) FILTER RECENT DATA
    timestamps = [t["signal_timestamp_raw"] for t in all_trades if t["signal_timestamp_raw"] > 0]
    if timestamps:
        latest_timestamp = max(timestamps)
        threshold = latest_timestamp - (3 * 24 * 3600)
        recent_trades = [t for t in all_trades if t["signal_timestamp_raw"] >= threshold]
        
        # Fallback if too few
        if len(recent_trades) < len(all_trades) * 0.1:
            all_trades.sort(key=lambda x: x["signal_timestamp_raw"])
            recent_trades = all_trades[-max(1, int(len(all_trades)*0.3)):]
    else:
        # Fallback if no timestamps
        recent_trades = all_trades[-max(1, int(len(all_trades)*0.3)):]

    total_trades_after = len(recent_trades)

    # 2) NORMALIZE RETURNS
    def get_return_pct(trade):
        r_mult = trade.get('R_multiple')
        if r_mult is not None:
            return float(r_mult)
        pnl = trade.get('pnl_absolute')
        size = trade.get('size_usd')
        if pnl is not None and size is not None and float(size) > 0:
            return float(pnl) / float(size)
        pnl_p = trade.get('pnl_pct')
        if pnl_p is not None:
            return float(pnl_p) / 100.0
        # Price fallback
        entry = trade.get('entry_price')
        exit_p = trade.get('exit_price')
        if entry is not None and exit_p is not None and float(entry) > 0:
            direction = str(trade.get('direction', 'BUY')).upper()
            if direction == 'BUY':
                return (float(exit_p) - float(entry)) / float(entry)
            else:
                return (float(entry) - float(exit_p)) / float(entry)
        return 0.0

    for t in recent_trades:
        t["return_pct"] = get_return_pct(t)

    # 3) PER-SPORT METRICS
    sports = {}
    for t in recent_trades:
        s = t.get("sport", "unknown").lower()
        if s not in sports: sports[s] = []
        sports[s].append(t)

    def calc_metrics(trade_list):
        if not trade_list: return None
        rets = [t["return_pct"] for t in trade_list]
        total = len(rets)
        wins = len([r for r in rets if r > 0])
        avg = statistics.mean(rets)
        median = statistics.median(rets)
        stdev = statistics.stdev(rets) if len(rets) > 1 else 0.0
        sharpe = avg / stdev if stdev != 0 else 0.0
        
        mfes = [t["max_favorable_excursion"] for t in trade_list if t.get("max_favorable_excursion") is not None]
        maes = [t["max_adverse_excursion"] for t in trade_list if t.get("max_adverse_excursion") is not None]
        
        return {
            "total_trades": total,
            "win_rate": wins / total,
            "avg_return": avg,
            "median_return": median,
            "std_dev_return": stdev,
            "sharpe_like": sharpe,
            "avg_MFE": statistics.mean(mfes) if mfes else 0.0,
            "avg_MAE": statistics.mean(maes) if maes else 0.0
        }

    per_sport_metrics = {s: calc_metrics(ts) for s, ts in sports.items()}

    # 4) TRADE FILTERING (RECENT DATA ONLY)
    filtering_results = {}
    for s, ts in sports.items():
        sorted_ts = sorted(ts, key=lambda x: x["return_pct"])
        
        def subset_metrics(pct_to_remove):
            count_to_remove = int(len(sorted_ts) * pct_to_remove)
            subset = sorted_ts[count_to_remove:]
            if not subset: return None
            rets = [t["return_pct"] for t in subset]
            return {
                "avg_return": statistics.mean(rets),
                "total_return": sum(rets),
                "win_rate": len([r for r in rets if r > 0]) / len(rets)
            }
            
        filtering_results[s] = {
            "remove_worst_20_pct": subset_metrics(0.2),
            "remove_worst_30_pct": subset_metrics(0.3)
        }

    # 5) PORTFOLIO SIMULATION
    # Combine football + tennis
    # Exclude nba if avg_return < 0
    nba_avg = per_sport_metrics.get("nba", {}).get("avg_return", -1) if per_sport_metrics.get("nba") else -1
    included_sports = ["football", "tennis"]
    if nba_avg >= 0: included_sports.append("nba")
    
    portfolio_trades = [t for t in recent_trades if t.get("sport", "unknown").lower() in included_sports]
    portfolio_trades.sort(key=lambda x: x["signal_timestamp_raw"])

    def simulate(sizing):
        bankroll = 100.0
        peak = 100.0
        max_dd = 0.0
        daily_returns = []
        for t in portfolio_trades:
            pnl = bankroll * sizing * t["return_pct"]
            bankroll += pnl
            if bankroll > peak: peak = bankroll
            dd = (peak - bankroll) / peak
            if dd > max_dd: max_dd = dd
            daily_returns.append(t["return_pct"] * sizing)
        
        vol = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.0
        return {
            "final_bankroll": bankroll,
            "max_drawdown": max_dd,
            "volatility": vol
        }

    portfolio_simulation = {
        "sizing_5_pct": simulate(0.05),
        "sizing_10_pct": simulate(0.10),
        "sizing_20_pct": simulate(0.20)
    }

    # 6) CAPITAL ALLOCATION
    def simulate_allocation(fb_weight, tn_weight):
        bankroll = 100.0
        peak = 100.0
        max_dd = 0.0
        pnl_series = []
        
        # Sort and filter
        fb_trades = sorted(sports.get("football", []), key=lambda x: x["signal_timestamp_raw"])
        tn_trades = sorted(sports.get("tennis", []), key=lambda x: x["signal_timestamp_raw"])
        
        # Alternate or follow timeline
        combined = []
        for t in fb_trades: combined.append((t, "fb"))
        for t in tn_trades: combined.append((t, "tn"))
        combined.sort(key=lambda x: x[0]["signal_timestamp_raw"])
        
        sizing = 0.10 # Base sizing for comparison
        for t, kind in combined:
            weight = fb_weight if kind == "fb" else tn_weight
            pnl = bankroll * sizing * weight * t["return_pct"]
            bankroll += pnl
            if bankroll > peak: peak = bankroll
            dd = (peak - bankroll) / peak
            if dd > max_dd: max_dd = dd
            pnl_series.append(sizing * weight * t["return_pct"])
            
        avg_pnl = statistics.mean(pnl_series) if pnl_series else 0.0
        std_pnl = statistics.stdev(pnl_series) if len(pnl_series) > 1 else 0.0
        sharpe = avg_pnl / std_pnl if std_pnl != 0 else 0.0
        
        return {
            "final_bankroll": bankroll,
            "max_drawdown": max_dd,
            "sharpe_like_portfolio": sharpe
        }

    allocation_results = {
        "100_football": simulate_allocation(1.0, 0.0),
        "70_football_30_tennis": simulate_allocation(0.7, 0.3),
        "50_football_50_tennis": simulate_allocation(0.5, 0.5)
    }

    output = {
        "data_filter": {
            "total_trades_before": total_trades_before,
            "total_trades_after": total_trades_after
        },
        "per_sport_metrics": per_sport_metrics,
        "filtering_results": filtering_results,
        "portfolio_simulation": portfolio_simulation,
        "allocation_results": allocation_results
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    solve()
