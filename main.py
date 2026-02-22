"""
main.py — Orchestrator for Polymarket Microstructure Diagnostic System.

Runs all 5 analysis phases + lead-lag cross-correlation, prints final report,
saves CSV & plot outputs. Deterministic with fixed random seed.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from data_loader import load_data
from event_detector import detect_significant_moves
from lag_analysis import measure_reaction_lags, compute_lag_stats, plot_lag_histogram
from lead_lag import compute_cross_correlation, find_peak_lag, plot_cross_correlation
from spread_analysis import compute_spread_summary
from adverse_selection import simulate_passive_fills, compute_adverse_selection_summary
from transition_matrix import compute_transition_summary
from volume_clustering import compute_volume_summary


def print_header(text, width=70):
    print('\n' + '═' * width)
    print(f'  {text}')
    print('═' * width)


def print_section(text):
    print(f'\n── {text} ' + '─' * max(1, 50 - len(text)))


def run_pipeline(mode='synthetic', days=14, assets=None, output_dir='output'):
    """Execute the full diagnostic pipeline."""
    if assets is None:
        assets = ['BTC', 'SOL']

    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, 'data')

    # ─── Data Loading ──────────────────────────────────────────
    print_header('POLYMARKET MICROSTRUCTURE DIAGNOSTIC')
    print(f'  Mode: {mode}')
    print(f'  Days: {days}')
    print(f'  Assets: {", ".join(assets)}')
    print(f'  Output: {output_dir}/')

    print_section('Loading Data')
    data = load_data(mode=mode, data_dir=data_dir, days=days, assets=assets)

    # Store results per asset for comparison
    all_results = {}

    for asset in assets:
        asset_dir = os.path.join(output_dir, asset.lower())
        os.makedirs(asset_dir, exist_ok=True)

        binance_df = data[asset]['binance']
        poly_df = data[asset]['polymarket']

        print_header(f'{asset} ANALYSIS')
        print(f'  Binance trades: {len(binance_df):,}')
        print(f'  Polymarket snapshots: {len(poly_df):,}')
        print(f'  Time range: {binance_df["timestamp"].min()} → '
              f'{binance_df["timestamp"].max()}')

        results = {}

        # ─── Event Detection ───────────────────────────────────
        print_section(f'{asset} — Event Detection')
        events_010 = detect_significant_moves(binance_df, poly_df, 0.10)
        events_020 = detect_significant_moves(binance_df, poly_df, 0.20)
        print(f'  Events ≥0.10%: {len(events_010):,}')
        print(f'  Events ≥0.20%: {len(events_020):,}')

        # Use 0.10% threshold for most analyses (more events)
        events = events_010
        events.to_csv(os.path.join(asset_dir, 'events.csv'), index=False)

        # ─── Phase 1: Reaction Lag ─────────────────────────────
        print_section(f'{asset} — Phase 1: Reaction Lag (Fair-Prob Anchored)')
        lag_df = measure_reaction_lags(events, poly_df,
                                        prob_threshold=0.02, max_window_s=5.0)
        lag_stats = compute_lag_stats(lag_df)
        lag_df.to_csv(os.path.join(asset_dir, f'lag_results_{asset.lower()}.csv'),
                      index=False)
        plot_lag_histogram(lag_df, asset,
                          os.path.join(asset_dir, f'lag_histogram_{asset.lower()}.png'))

        print(f'  Total events analyzed: {lag_stats["count"]}')
        print(f'  True reactions: {lag_stats["true_reactions"]} '
              f'({lag_stats["true_reaction_rate"]:.1%})')
        print(f'  False reactions: {lag_stats["false_reactions"]} '
              f'({lag_stats["false_reaction_rate"]:.1%})')
        print(f'  No reaction (5s cap): {lag_stats["no_reactions"]} '
              f'({lag_stats["no_reaction_rate"]:.1%})')
        print(f'  Mean lag: {lag_stats["mean_lag_ms"]:.0f} ms')
        print(f'  Median lag: {lag_stats["median_lag_ms"]:.0f} ms')
        print(f'  P90 lag: {lag_stats["p90_lag_ms"]:.0f} ms')
        results['lag'] = lag_stats

        # ─── Lead-Lag Cross-Correlation ────────────────────────
        print_section(f'{asset} — Lead-Lag Cross-Correlation')
        ccf_df = compute_cross_correlation(binance_df, poly_df,
                                            max_lag_s=5, resolution_ms=100)
        peak = find_peak_lag(ccf_df)
        ccf_df.to_csv(os.path.join(asset_dir,
                      f'cross_correlation_{asset.lower()}.csv'), index=False)
        plot_cross_correlation(ccf_df, asset,
                               os.path.join(asset_dir,
                               f'cross_correlation_{asset.lower()}.png'))

        print(f'  Peak lag: {peak["peak_lag_ms"]} ms')
        print(f'  Peak correlation: {peak["peak_correlation"]:.4f}')
        print(f'  → {peak["interpretation"]}')
        results['lead_lag'] = peak

        # ─── Phase 2: Spread Persistence ───────────────────────
        print_section(f'{asset} — Phase 2: Spread Persistence')
        spread_stats = compute_spread_summary(poly_df, events, binance_df,
                                              asset_dir)
        print(f'  Avg spread: {spread_stats["avg_spread"]:.4f}')
        print(f'  P95 spread: {spread_stats["p95_spread"]:.4f}')
        print(f'  Mean persistence (>0.02): {spread_stats["mean_persistence_ms"]:.0f} ms')
        print(f'  Wide spread episodes: {spread_stats["n_wide_spread_episodes"]}')
        print(f'  Spread-vol correlation: {spread_stats["spread_vol_correlation"]:.4f} '
              f'(p={spread_stats["spread_vol_pvalue"]:.4f})')
        results['spread'] = spread_stats

        # ─── Phase 3: Adverse Selection ────────────────────────
        print_section(f'{asset} — Phase 3: Adverse Selection (Prob-Space)')
        fills = simulate_passive_fills(poly_df)
        print(f'  Detected fills: {len(fills):,}')

        # Subsample fills if too many (keep analysis tractable)
        if len(fills) > 50000:
            fills = fills.sample(50000, random_state=42).reset_index(drop=True)
            print(f'  Subsampled to: {len(fills):,}')

        adv_stats = compute_adverse_selection_summary(
            fills, poly_df, binance_df, asset_dir)

        for h in [1, 3, 10]:
            key = f'mean_adv_sel_{h}s'
            if key in adv_stats:
                print(f'  Mean adverse selection ({h}s): '
                      f'{adv_stats[key]:.6f} prob')
                print(f'  P95 adverse selection ({h}s): '
                      f'{adv_stats.get(f"p95_adv_sel_{h}s", 0):.6f} prob')
        results['adverse_selection'] = adv_stats

        # ─── Phase 4: Transition Matrix ────────────────────────
        print_section(f'{asset} — Phase 4: Probability Transitions')
        trans_stats = compute_transition_summary(poly_df, asset_dir)
        print(f'  Mean self-transition (30s): '
              f'{trans_stats["mean_self_transition_30s"]:.4f}')
        print(f'  Mean self-transition (60s): '
              f'{trans_stats["mean_self_transition_60s"]:.4f}')
        print(f'  Sweet spot (0.2-0.4, 0.6-0.8) vol: '
              f'{trans_stats["sweet_spot_mean_vol"]:.6f}')
        print(f'  Other buckets vol: {trans_stats["other_mean_vol"]:.6f}')
        print(f'  Sweet spot ratio: {trans_stats["sweet_spot_ratio"]:.2f}x')
        results['transition'] = trans_stats

        # ─── Phase 5: Volume Clustering ────────────────────────
        print_section(f'{asset} — Phase 5: Volume & Fill Clustering')
        vol_stats = compute_volume_summary(events, poly_df, asset_dir)
        print(f'  Avg fills/min: {vol_stats["avg_fills_per_min"]:.2f}')
        print(f'  P95 fills/min: {vol_stats["p95_fills_per_min"]:.2f}')
        print(f'  Burst ratio (event/baseline): {vol_stats["burst_ratio"]:.2f}x')
        if 'expiry_concentration_ratio' in vol_stats:
            print(f'  Expiry concentration (0-30s / 180-300s): '
                  f'{vol_stats["expiry_concentration_ratio"]:.2f}x')
        results['volume'] = vol_stats

        all_results[asset] = results

    # ─── Final Report ──────────────────────────────────────────
    print_final_report(all_results, assets)

    return all_results


def print_final_report(all_results, assets):
    """Print the consolidated final report."""
    print_header('FINAL DIAGNOSTIC REPORT')

    for asset in assets:
        if asset not in all_results:
            continue
        r = all_results[asset]

        print(f'\n┌─ {asset} ─────────────────────────────────────────────┐')

        # Lag
        lag = r.get('lag', {})
        print(f'│  Median reaction lag:     {lag.get("median_lag_ms", "N/A"):>8} ms')
        print(f'│  P90 reaction lag:        {lag.get("p90_lag_ms", "N/A"):>8} ms')
        print(f'│  True reaction rate:      {lag.get("true_reaction_rate", 0):>8.1%}')
        print(f'│  False reaction rate:     {lag.get("false_reaction_rate", 0):>8.1%}')

        # Lead-lag
        ll = r.get('lead_lag', {})
        print(f'│  Lead-lag peak:           {ll.get("peak_lag_ms", "N/A"):>8} ms')
        print(f'│  Peak correlation:        {ll.get("peak_correlation", "N/A"):>8}')

        # Spread
        sp = r.get('spread', {})
        print(f'│  Avg spread:              {sp.get("avg_spread", 0):>8.4f}')
        print(f'│  Spread-vol corr:         {sp.get("spread_vol_correlation", 0):>8.4f}')

        # Adverse selection
        adv = r.get('adverse_selection', {})
        print(f'│  Mean adv sel (1s):       {adv.get("mean_adv_sel_1s", 0):>8.6f} prob')
        print(f'│  Mean adv sel (3s):       {adv.get("mean_adv_sel_3s", 0):>8.6f} prob')
        print(f'│  Mean adv sel (10s):      {adv.get("mean_adv_sel_10s", 0):>8.6f} prob')

        # Transitions
        tr = r.get('transition', {})
        print(f'│  Sweet spot vol ratio:    {tr.get("sweet_spot_ratio", 0):>8.2f}x')

        # Volume
        vol = r.get('volume', {})
        print(f'│  Avg fills/min:           {vol.get("avg_fills_per_min", 0):>8.2f}')
        print(f'│  Burst ratio:             {vol.get("burst_ratio", 0):>8.2f}x')

        print(f'└──────────────────────────────────────────────────────┘')

    # Comparison summary
    if len(assets) > 1 and all(a in all_results for a in assets):
        print_section('BTC vs SOL Comparison')
        headers = ['Metric', *assets]
        rows = [
            ['Median lag (ms)',
             *[f'{all_results[a].get("lag", {}).get("median_lag_ms", "N/A"):.0f}'
               for a in assets]],
            ['P90 lag (ms)',
             *[f'{all_results[a].get("lag", {}).get("p90_lag_ms", "N/A"):.0f}'
               for a in assets]],
            ['True reaction rate',
             *[f'{all_results[a].get("lag", {}).get("true_reaction_rate", 0):.1%}'
               for a in assets]],
            ['Lead-lag peak (ms)',
             *[f'{all_results[a].get("lead_lag", {}).get("peak_lag_ms", "N/A")}'
               for a in assets]],
            ['Avg spread',
             *[f'{all_results[a].get("spread", {}).get("avg_spread", 0):.4f}'
               for a in assets]],
            ['Mean adv sel 1s',
             *[f'{all_results[a].get("adverse_selection", {}).get("mean_adv_sel_1s", 0):.6f}'
               for a in assets]],
            ['Mean adv sel 3s',
             *[f'{all_results[a].get("adverse_selection", {}).get("mean_adv_sel_3s", 0):.6f}'
               for a in assets]],
            ['Mean adv sel 10s',
             *[f'{all_results[a].get("adverse_selection", {}).get("mean_adv_sel_10s", 0):.6f}'
               for a in assets]],
            ['Sweet spot ratio',
             *[f'{all_results[a].get("transition", {}).get("sweet_spot_ratio", 0):.2f}x'
               for a in assets]],
            ['Burst ratio',
             *[f'{all_results[a].get("volume", {}).get("burst_ratio", 0):.2f}x'
               for a in assets]],
        ]

        col_widths = [max(len(str(r[i])) for r in [headers] + rows)
                      for i in range(len(headers))]
        fmt = '  '.join(f'{{:<{w}}}' for w in col_widths)
        print(fmt.format(*headers))
        print('  '.join('─' * w for w in col_widths))
        for row in rows:
            print(fmt.format(*row))

    print('\n' + '═' * 70)
    print('  Diagnostic complete. No strategy recommendations.')
    print('  All outputs saved to output/ directory.')
    print('═' * 70 + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Polymarket Microstructure Diagnostic System')
    parser.add_argument('--mode', type=str, default='synthetic',
                        choices=['synthetic', 'real'],
                        help='Data mode: synthetic or real')
    parser.add_argument('--days', type=int, default=14,
                        help='Number of days of data')
    parser.add_argument('--assets', type=str, nargs='+',
                        default=['BTC', 'SOL'],
                        help='Assets to analyze (e.g. BTC SOL)')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory')
    args = parser.parse_args()

    run_pipeline(mode=args.mode, days=args.days, assets=args.assets,
                 output_dir=args.output)


if __name__ == '__main__':
    main()
