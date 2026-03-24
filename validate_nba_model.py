import json
import statistics
import datetime
import math
import csv
import sys
from typing import List, Dict, Any

FILE_PATH = "/Users/MartinOile/Desktop/poly_high/sports_data/extracted_trades_ticks.json"
CSV_PATH = "/Users/MartinOile/Desktop/poly_high/sports_data/nba_model_predictions.csv"

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

def load_predictions(csv_path: str) -> Dict[str, List[Dict]]:
    preds = {}
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gid = row.get('game_id')
                if not gid: continue
                ts = float(row.get('timestamp', 0))
                p_home = float(row.get('predicted_prob_home', 0))
                p_away = float(row.get('predicted_prob_away', 0))
                
                if gid not in preds:
                    preds[gid] = []
                preds[gid].append({
                    "timestamp": ts,
                    "prob_home": p_home,
                    "prob_away": p_away
                })
                
        # Sort each game's predictions by timestamp
        for gid in preds:
            preds[gid].sort(key=lambda x: x["timestamp"])
            
    except FileNotFoundError:
        pass
    return preds

def find_closest_pred(preds_list: List[Dict], target_ts: float) -> Dict:
    # Find closest BEFORE entry
    best = None
    for p in preds_list:
        if p["timestamp"] <= target_ts:
            best = p
        else:
            break
    return best

def safe_mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v)/len(v) if v else 0.0

def build_buckets(sim_trades: List[Dict], score_key: str):
    sim_trades.sort(key=lambda x: x[score_key])
    n = len(sim_trades)
    q_size = n // 5
    if q_size == 0:
        return None
        
    for i, t in enumerate(sim_trades):
        if i < q_size: t[f'_{score_key}_quintile'] = "Q1"
        elif i < q_size*2: t[f'_{score_key}_quintile'] = "Q2"
        elif i < q_size*3: t[f'_{score_key}_quintile'] = "Q3"
        elif i < q_size*4: t[f'_{score_key}_quintile'] = "Q4"
        else: t[f'_{score_key}_quintile'] = "Q5"
        
    buckets = {"Q1":[], "Q2":[], "Q3":[], "Q4":[], "Q5":[]}
    for t in sim_trades:
        buckets[t[f'_{score_key}_quintile']].append(t)
        
    metrics = {}
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        trds = buckets[q]
        if not trds: 
            metrics[q] = {"number_of_trades": 0, "avg_return": 0.0, "total_return": 0.0, "win_rate": 0.0}
            continue
        rets = [x['return_pct'] for x in trds]
        wins = len([r for r in rets if r > 0])
        metrics[q] = {
            "number_of_trades": len(trds),
            "avg_return": safe_mean(rets),
            "total_return": sum(rets),
            "win_rate": wins / len(trds)
        }
    return metrics, buckets

def zscore_normalize(trades: List[Dict], key: str, out_key: str):
    vals = [t[key] for t in trades]
    m = sum(vals)/len(vals) if vals else 0.0
    s = statistics.stdev(vals) if len(vals)>1 else 0.0001
    s = max(s, 0.0001)
    for t in trades:
        t[out_key] = (t[key] - m) / s

def main():
    try:
        with open(FILE_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {FILE_PATH}"}))
        return

    predictions = load_predictions(CSV_PATH)
    if not predictions:
        print(json.dumps({"error": f"Predictions CSV not found or empty: {CSV_PATH}"}))
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
        cleaned.append(t)
        
    cleaned.sort(key=lambda x: x['_ts'])
    
    # 1. Filter Data
    valid_ts = [t['_ts'] for t in cleaned if t['_ts'] > 0]
    filtered = []
    if valid_ts:
        latest = max(valid_ts)
        threshold = latest - (3 * 24 * 3600)
        filtered = [t for t in cleaned if t['_ts'] >= threshold]
        
    nba_trades = [t for t in filtered if t.get('sport', '').lower() == 'nba']
    if not nba_trades:
        print(json.dumps({"error": "No NBA trades found in recent 3 days"}))
        return
        
    # Match trades to predictions
    matched_trades = []
    for t in nba_trades:
        gid = str(t.get('market_id')) # Assuming market_id corresponds to game_id
        if gid not in predictions:
            # Fallback exact string matching or other logic if needed
            gid = str(t.get('game_id'))
        if gid not in predictions:
            continue
            
        entry_ts = float(t.get('entry_timestamp_raw', t['_ts']))
        pred = find_closest_pred(predictions[gid], entry_ts)
        
        if pred:
            t['_pred_prob_home'] = pred['prob_home']
            t['_pred_prob_away'] = pred['prob_away']
            matched_trades.append(t)
            
    if not matched_trades:
        print(json.dumps({"error": "Failed to map any NBA trades to predictions"}))
        return

    # 2. Compute Model Edge
    for t in matched_trades:
        direction = str(t.get('direction', 'BUY')).upper()
        selection = str(t.get('selection', '')).lower()
        
        model_prob = t['_pred_prob_home'] if selection == 'home' else t['_pred_prob_away']
        
        # Get market implied prob
        ticks = t.get('ticks', t.get('ticks_before_entry', []))
        mid_p = None
        if ticks:
            last_tick = ticks[-1]
            if isinstance(last_tick, dict):
                mid_p = last_tick.get('mid_price', last_tick.get('price'))
        
        if mid_p is None:
            mid_p = t.get('entry_price')
            
        implied_prob = float(mid_p) if mid_p is not None else 0.5
        
        if direction == 'SELL':
            # Selling means buying the opposite side
            model_prob = 1.0 - model_prob
            implied_prob = 1.0 - implied_prob
            
        t['edge_at_entry_model'] = model_prob - implied_prob
        t['edge_at_entry_old'] = t.get('edge_at_entry', 0.0)
        t['confirmation'] = t.get('confirmation_ticks_count', 0.0)

    # 3. Build Scores (Normalize first)
    # Use absolute magnitude for both edges 
    for t in matched_trades:
        t['abs_edge_model'] = abs(t['edge_at_entry_model'])
        t['abs_edge_old'] = abs(t['edge_at_entry_old'])

    zscore_normalize(matched_trades, 'abs_edge_model', 'z_edge_model')
    zscore_normalize(matched_trades, 'abs_edge_old', 'z_edge_old')
    zscore_normalize(matched_trades, 'confirmation', 'z_conf')
    
    for t in matched_trades:
        t['score_old'] = t['z_edge_old'] + t['z_conf']
        t['score_new'] = t['z_edge_model'] + t['z_conf']

    # 4 & 5. Bucketing
    metrics_old, buckets_old = build_buckets(matched_trades, 'score_old')
    metrics_new, buckets_new = build_buckets(matched_trades, 'score_new')
    
    if not metrics_old or not metrics_new:
        print(json.dumps({"error": "Not enough trades to form quintiles"}))
        return

    # 6. Compare Signal Quality
    q5_old_avg = metrics_old["Q5"]["avg_return"]
    q1_old_avg = metrics_old["Q1"]["avg_return"]
    sep_old = q5_old_avg - q1_old_avg
    
    q5_new_avg = metrics_new["Q5"]["avg_return"]
    q1_new_avg = metrics_new["Q1"]["avg_return"]
    sep_new = q5_new_avg - q1_new_avg
    
    q5_improvement = q5_new_avg - q5_old_avg
    separation_improvement = sep_new - sep_old
    is_better = q5_new_avg > q5_old_avg and sep_new > sep_old

    # 7. Filter Simulation
    def sim_port(subset):
        br = 100.0
        rets = []
        for t in sorted(subset, key=lambda x: x['_ts']):
            pnl = br * 0.10 * t['return_pct']
            br += pnl
            rets.append(t['return_pct'])
        return {
            "trades_remaining": len(subset),
            "avg_return": safe_mean(rets),
            "total_return": sum(rets)
        }
        
    sim_old = sim_port(buckets_old["Q5"])
    sim_new = sim_port(buckets_new["Q5"])

    out = {
        "old_signal": metrics_old,
        "new_signal": metrics_new,
        "comparison": {
            "q5_improvement": q5_improvement,
            "separation_improvement": separation_improvement,
            "is_new_better": is_better
        },
        "filter_simulation": {
            "OLD_trade_only_Q5": sim_old,
            "NEW_trade_only_Q5": sim_new
        },
        "decision": "replace_model" if is_better else "keep_current",
        "key_findings": [
            "Evaluated Elo/Efficiency-based NBA model against market-implied baseline.",
            f"New model Q5 delta: {q5_improvement*100:.2f}%",
            f"Replacement decision generated via bucket priority constraints."
        ]
    }
    
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
