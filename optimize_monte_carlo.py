import json
import statistics
import datetime
import math
import random
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

def calc_skew(data):
    if len(data) < 3: return 0.0
    mean = sum(data)/len(data)
    std = statistics.stdev(data)
    if std == 0: return 0.0
    n = len(data)
    return (n / ((n - 1) * (n - 2))) * sum(((x - mean) / std) ** 3 for x in data)

def safe_mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v)/len(v) if v else 0.0

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
        cleaned.append(t)
        
    cleaned.sort(key=lambda x: x['_ts'])
    
    # 1. Filter Data
    valid_ts = [t['_ts'] for t in cleaned if t['_ts'] > 0]
    filtered = []
    if valid_ts:
        latest = max(valid_ts)
        threshold = latest - (3 * 24 * 3600)
        filtered = [t for t in cleaned if t['_ts'] >= threshold]
        
    if len(filtered) < 0.1 * len(cleaned):
        filtered = cleaned[-int(len(cleaned)*0.3):] if cleaned else []
        
    sim_trades = [t for t in filtered if t.get('sport', '').lower() in ['football', 'tennis']]
    
    if len(sim_trades) < 5:
        print(json.dumps({"error": "Not enough trades"}))
        return

    # 2. Build Score
    edges = [abs(t.get('edge_at_entry', 0.0)) for t in sim_trades]
    confs = [t.get('confirmation_ticks_count', 0.0) for t in sim_trades]
    
    mean_e = sum(edges)/len(edges)
    std_e = statistics.stdev(edges) if len(edges) > 1 else 0.0001
    std_e = max(std_e, 0.0001)
    
    mean_c = sum(confs)/len(confs)
    std_c = statistics.stdev(confs) if len(confs) > 1 else 0.0001
    std_c = max(std_c, 0.0001)
    
    for t in sim_trades:
        e = abs(t.get('edge_at_entry', 0.0))
        c = t.get('confirmation_ticks_count', 0.0)
        t['_score'] = ((e - mean_e) / std_e) + ((c - mean_c) / std_c)
        
    # 3. Select Q5
    sim_trades.sort(key=lambda x: x['_score'])
    n = len(sim_trades)
    q5_size = math.ceil(n * 0.2)
    q5_trades = sim_trades[-q5_size:] if q5_size > 0 else sim_trades
    
    # 4. Prepare Returns
    returns_list = [t['return_pct'] for t in q5_trades]
    n_trades = len(returns_list)
    
    # 5. Monte Carlo Setup
    simulations = 10000
    
    def run_mc(sizing) -> Dict:
        final_banks = []
        max_dds = []
        losses = 0
        ruins = 0
        
        for _ in range(simulations):
            br = 100.0
            peak = 100.0
            mdd = 0.0
            
            for _ in range(n_trades):
                r = random.choice(returns_list)
                pnl = br * sizing * r
                br += pnl
                
                if br > peak: peak = br
                dd = (peak - br)/peak if peak > 0 else 0
                if dd > mdd: mdd = dd
                
            final_banks.append(br)
            max_dds.append(mdd)
            
            if br < 100.0: losses += 1
            if br < 20.0: ruins += 1
            
        return {
            "median_final_bankroll": statistics.median(final_banks),
            "mean_final_bankroll": sum(final_banks)/simulations,
            "min_final_bankroll": min(final_banks),
            "max_final_bankroll": max(final_banks),
            "probability_of_loss": losses / simulations,
            "probability_of_ruin": ruins / simulations,
            "median_max_drawdown": statistics.median(max_dds),
            "worst_drawdown": max(max_dds),
            "_banks": final_banks # internal
        }

    res_5 = run_mc(0.05)
    res_10 = run_mc(0.10)
    res_15 = run_mc(0.15)
    res_20 = run_mc(0.20)
    
    # 7. Stability Check
    # We use 10% sizing banks for generic stability check
    banks_10 = res_10["_banks"]
    var = statistics.variance(banks_10)
    sk = calc_skew(banks_10)
    
    # Cleanup internal data
    for d in (res_5, res_10, res_15, res_20):
        del d["_banks"]

    is_stb = sk < 2.5 and res_10["probability_of_ruin"] < 0.05

    out = {
        "5pct": res_5,
        "10pct": res_10,
        "15pct": res_15,
        "20pct": res_20,
        "stability": {
            "variance": var,
            "skew": sk,
            "is_stable": is_stb
        }
    }
    
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
