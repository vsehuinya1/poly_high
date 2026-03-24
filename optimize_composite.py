import json
import statistics
import datetime
import math
import itertools
from typing import List, Dict, Any

FILE_PATH = "/Users/MartinOile/Desktop/poly_high/sports_data/extracted_trades_ticks.json"

def get_return_pct(trade: Dict) -> float:
    ret = trade.get('R_multiple')
    if ret is not None:
        return float(ret)
    
    pnl = trade.get('pnl_absolute')
    size = trade.get('size_usd')
    if pnl is not None and size is not None and float(size) > 0:
        return float(pnl) / float(size)
        
    pnl_p = trade.get('pnl_pct')
    if pnl_p is not None:
        return float(pnl_p) / 100.0
        
    entry = trade.get('entry_price')
    exit_p = trade.get('exit_price')
    if entry is not None and exit_p is not None and float(entry) > 0:
        direction = str(trade.get('direction', 'BUY')).upper()
        if direction == 'BUY':
            return (float(exit_p) - float(entry)) / float(entry)
        else:
            return (float(entry) - float(exit_p)) / float(entry)
    return 0.0

def process_ticks_for_features(trade: Dict) -> Dict:
    ticks = trade.get('ticks', trade.get('ticks_before_entry', []))
    if not ticks:
        return {
            "spread_at_entry": 0.0,
            "volatility": 0.0,
            "trend": 0.0
        }
        
    prices = []
    spreads = []
    for t in ticks:
        if isinstance(t, dict):
            p = t.get('mid_price', t.get('price'))
            s = t.get('spread')
            if p is not None: prices.append(float(p))
            if s is not None: spreads.append(float(s))
            
    avg_s = sum(spreads)/len(spreads) if spreads else 0.0
    vol = statistics.stdev(prices) if len(prices) > 1 else 0.0
    trend = 0.0
    if len(prices) >= 2:
        trend = prices[-1] - prices[0]
        
    return {
        "spread_at_entry": avg_s,
        "volatility": vol,
        "trend": trend
    }

def pearson_corr(x, y):
    if len(x) != len(y) or len(x) < 2: return 0.0
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((xi - mean_x)*(yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x)**2 for xi in x)
    den_y = sum((yi - mean_y)**2 for yi in y)
    if den_x == 0 or den_y == 0: return 0.0
    return num / math.sqrt(den_x * den_y)

def main():
    try:
        with open(FILE_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(json.dumps({"error": "File not found"}))
        return

    raw_trades = []
    if isinstance(data, dict) and "trades" in data:
        for k, v in data["trades"].items():
            if isinstance(v, list):
                raw_trades.extend(v)
    elif isinstance(data, list):
        raw_trades = data
        
    cleaned = []
    for t in raw_trades:
        ts = t.get('signal_timestamp_raw')
        if not ts:
            st = t.get('signal_timestamp')
            if st:
                try:
                    if st.endswith('Z'): st = st[:-1]
                    ts = datetime.datetime.fromisoformat(st).timestamp()
                except:
                    ts = 0
            else:
                ts = 0
        t['_ts'] = float(ts)
        t['return_pct'] = get_return_pct(t)
        
        feats = process_ticks_for_features(t)
        t.update(feats)
        cleaned.append(t)
        
    cleaned.sort(key=lambda x: x['_ts'])
    
    valid_ts = [t['_ts'] for t in cleaned if t['_ts'] > 0]
    filtered = []
    if valid_ts:
        latest = max(valid_ts)
        threshold = latest - (3 * 24 * 3600)
        filtered = [t for t in cleaned if t['_ts'] >= threshold]
        
    if len(filtered) < 0.1 * len(cleaned):
        filtered = cleaned[-int(len(cleaned)*0.3):] if cleaned else []
        
    if not filtered:
        print(json.dumps({"error": "No trades found"}))
        return

    # Extract raw features
    for t in filtered:
        # We want higher to be 'better' for the composite sum test.
        # Absolute edge magnitude (wait, or just standard formulation?)
        # Let's define the base features exactly
        t['f_edge'] = abs(t.get('edge_at_entry', 0.0))
        t['f_conf'] = t.get('confirmation_ticks_count', 0.0)
        t['f_delay'] = t.get('delay_seconds_applied', 0.0)
        t['f_spread'] = t.get('spread_at_entry', 0.0)
        t['f_vol'] = t.get('volatility', 0.0)
        
        direction = str(t.get('direction', 'BUY')).upper()
        t_val = t.get('trend', 0.0)
        t['f_trend'] = t_val if direction == 'BUY' else -t_val

    # Normalize (Z-Score)
    features_list = ['f_edge', 'f_conf', 'f_delay', 'f_spread', 'f_vol', 'f_trend']
    for f in features_list:
        vals = [t[f] for t in filtered]
        mean_v = sum(vals)/len(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
        
        for t in filtered:
            if std_v > 0.0001:
                t[f + '_z'] = (t[f] - mean_v) / std_v
            else:
                t[f + '_z'] = 0.0

    # Build Composite Signals Combinations
    # To test combinations efficiently, we assign fixed weights based on logic:
    # + edge (higher edge mag is good)
    # + conf (higher conf is good)
    # - delay (lower delay is good)
    # - spread (lower spread is good)
    # - vol (lower volatility is good)
    # + trend (trend matching direction is good)
    
    weights = {
        'f_edge': 1,
        'f_conf': 1,
        'f_delay': -1,
        'f_spread': -1,
        'f_vol': -1,
        'f_trend': 1
    }
    
    # We will test all subsets of length 1 to 6
    all_signals_tested = []
    
    combinations = []
    for L in range(1, len(features_list)+1):
        for subset in itertools.combinations(features_list, L):
            combinations.append(subset)
            
    for combo in combinations:
        formula_name = " + ".join([(f"{weights[f]}*{f.replace('f_', '')}") for f in combo])
        
        # Calculate score for each trade
        for t in filtered:
            score = 0.0
            for f in combo:
                score += weights[f] * t[f + '_z']
            t['_composite_score'] = score
            
        # Rank trades
        sorted_trades = sorted(filtered, key=lambda x: x['_composite_score'])
        n_total = len(sorted_trades)
        n_20 = int(n_total * 0.2)
        n_20 = max(1, n_20)
        
        bottom_20 = sorted_trades[:n_20]
        top_20 = sorted_trades[-n_20:]
        middle_60 = sorted_trades[n_20:-n_20] if n_total > n_20*2 else []
        
        def g_ret(grp):
            if not grp: return 0.0
            r = [x['return_pct'] for x in grp]
            return sum(r)/len(r)
            
        bot_ret = g_ret(bottom_20)
        top_ret = g_ret(top_20)
        mid_ret = g_ret(middle_60)
        
        tot_ret = sum(x['return_pct'] for x in sorted_trades)
        
        scores = [x['_composite_score'] for x in sorted_trades]
        rets = [x['return_pct'] for x in sorted_trades]
        corr = pearson_corr(scores, rets)
        
        res = {
            "formula": formula_name,
            "correlation": corr,
            "top20_avg_return": top_ret,
            "middle60_avg_return": mid_ret,
            "bottom20_avg_return": bot_ret,
            "top_vs_bottom_separation": top_ret - bot_ret,
            "total_return": tot_ret,
            "combo_keys": combo
        }
        all_signals_tested.append(res)
        
    # Sort to find best. We want max top20_avg_return, provided separation > 0
    all_signals_tested.sort(key=lambda x: (x['top20_avg_return'], x['correlation']), reverse=True)
    
    best_signal = all_signals_tested[0]
    
    # Format output keys exactly as requested
    out = {
        "best_signal": {
            "formula": best_signal["formula"],
            "correlation": best_signal["correlation"],
            "top20_avg_return": best_signal["top20_avg_return"],
            "bottom20_avg_return": best_signal["bottom20_avg_return"]
        },
        "all_signals_tested": [
            {
                "formula": s["formula"],
                "correlation": s["correlation"],
                "top20_avg_return": s["top20_avg_return"],
                "bottom20_avg_return": s["bottom20_avg_return"],
                "separation": s["top_vs_bottom_separation"]
            }
            for s in all_signals_tested
        ]
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
