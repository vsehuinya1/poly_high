#!/usr/bin/env python3
"""
Polymarket Market Scanner — Phase 1
Queries Gamma API for ALL active markets, filters by volume/liquidity,
excludes crypto updown markets, ranks by 24h volume.
"""

import requests
import json
import sys
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

# Filters
MIN_VOLUME_24H = 50_000       # $50k minimum 24h volume
MIN_LIQUIDITY  = 10_000       # $10k minimum liquidity
LIMIT_PER_PAGE = 100
MAX_PAGES      = 20           # up to 2000 events scanned

# Slug patterns to EXCLUDE (crypto updown noise)
CRYPTO_EXCLUDE_PATTERNS = [
    "updown", "btc-", "eth-", "sol-", "xrp-", "doge-",
    "bitcoin-price", "ethereum-price", "solana-price",
    "crypto-price", "bnb-", "ada-", "avax-", "dot-",
    "matic-", "link-", "ltc-", "shib-",
]


def is_crypto_updown(event: dict) -> bool:
    """Check if an event looks like a crypto updown / price market."""
    slug = event.get("slug", "").lower()
    title = event.get("title", "").lower()
    combined = slug + " " + title

    for pattern in CRYPTO_EXCLUDE_PATTERNS:
        if pattern in combined:
            return True
    return False


def fetch_all_events() -> list[dict]:
    """Paginate through Gamma API events endpoint, sorted by 24h volume."""
    all_events = []

    for page in range(MAX_PAGES):
        offset = page * LIMIT_PER_PAGE
        params = {
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "limit": str(LIMIT_PER_PAGE),
            "offset": str(offset),
        }

        try:
            resp = requests.get(GAMMA_EVENTS_URL, params=params, timeout=30)
            resp.raise_for_status()
            events = resp.json()
        except Exception as e:
            print(f"  [!] API error on page {page}: {e}")
            break

        if not events:
            break

        all_events.extend(events)
        print(f"  Fetched page {page+1}: {len(events)} events (total: {len(all_events)})")

        if len(events) < LIMIT_PER_PAGE:
            break

    return all_events


def extract_market_data(event: dict) -> dict:
    """Extract key metrics from an event-level record."""
    markets = event.get("markets", [])
    n_outcomes = len(markets) if markets else 0

    # Aggregate per-event
    total_volume_24h = float(event.get("volume24hr", 0) or 0)
    total_volume     = float(event.get("volume", 0) or 0)
    total_liquidity  = float(event.get("liquidity", 0) or 0)
    volume_1wk       = float(event.get("volume1wk", 0) or 0)
    volume_1mo       = float(event.get("volume1mo", 0) or 0)

    # Best spread from sub-markets
    spreads = []
    best_bids = []
    best_asks = []
    token_ids = []

    for m in markets:
        spread = float(m.get("spread", 1) or 1)
        spreads.append(spread)
        bb = float(m.get("bestBid", 0) or 0)
        ba = float(m.get("bestAsk", 0) or 0)
        best_bids.append(bb)
        best_asks.append(ba)

        # Collect clobTokenIds
        raw_ids = m.get("clobTokenIds", "[]")
        if isinstance(raw_ids, str):
            try:
                ids = json.loads(raw_ids)
            except json.JSONDecodeError:
                ids = []
        else:
            ids = raw_ids
        token_ids.extend(ids)

    avg_spread = sum(spreads) / len(spreads) if spreads else 1.0
    min_spread = min(spreads) if spreads else 1.0

    # End date
    end_date_str = event.get("endDate") or ""
    if end_date_str:
        try:
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_to_resolution = (end_dt - now).days
        except Exception:
            days_to_resolution = -1
    else:
        days_to_resolution = -1

    # Competitive score (0-1, higher = more competitive/active book)
    competitive = float(event.get("competitive", 0) or 0)

    return {
        "event_id":          event.get("id", ""),
        "title":             event.get("title", "")[:80],
        "slug":              event.get("slug", ""),
        "n_outcomes":        n_outcomes,
        "volume_total":      total_volume,
        "volume_24h":        total_volume_24h,
        "volume_1wk":        volume_1wk,
        "volume_1mo":        volume_1mo,
        "liquidity":         total_liquidity,
        "avg_spread":        avg_spread,
        "min_spread":        min_spread,
        "days_to_resolution": days_to_resolution,
        "competitive":       competitive,
        "end_date":          end_date_str[:10],
        "neg_risk":          event.get("negRisk", False),
        "n_token_ids":       len(token_ids),
        "token_ids_sample":  token_ids[:4],  # first 4 for reference
    }


def main():
    print("=" * 80)
    print("  POLYMARKET MARKET SCANNER — Phase 1")
    print(f"  Filters: 24h Vol ≥ ${MIN_VOLUME_24H:,.0f} | Liquidity ≥ ${MIN_LIQUIDITY:,.0f}")
    print(f"  Excluding: crypto updown / price markets")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 80)

    # ── Fetch ──────────────────────────────────────────────────────────
    print("\n[1] Fetching events from Gamma API...")
    raw_events = fetch_all_events()
    print(f"    Total events fetched: {len(raw_events)}")

    # ── Filter ─────────────────────────────────────────────────────────
    print("\n[2] Filtering...")
    filtered = []
    crypto_excluded = 0
    volume_excluded = 0
    liquidity_excluded = 0

    for event in raw_events:
        if is_crypto_updown(event):
            crypto_excluded += 1
            continue

        vol_24h = float(event.get("volume24hr", 0) or 0)
        liq     = float(event.get("liquidity", 0) or 0)

        if vol_24h < MIN_VOLUME_24H:
            volume_excluded += 1
            continue
        if liq < MIN_LIQUIDITY:
            liquidity_excluded += 1
            continue

        data = extract_market_data(event)
        filtered.append(data)

    print(f"    Crypto updown excluded: {crypto_excluded}")
    print(f"    Below volume threshold: {volume_excluded}")
    print(f"    Below liquidity threshold: {liquidity_excluded}")
    print(f"    → Qualifying events: {len(filtered)}")

    # ── Sort by 24h volume ─────────────────────────────────────────────
    filtered.sort(key=lambda x: x["volume_24h"], reverse=True)

    # ── Display Top 20 ─────────────────────────────────────────────────
    top_n = 20
    print(f"\n{'=' * 120}")
    print(f"  TOP {min(top_n, len(filtered))} ACTIVE POLYMARKET EVENTS (by 24h Volume)")
    print(f"{'=' * 120}")

    print(f"\n{'#':>3}  {'Title':<55} {'Vol 24h':>12} {'Vol Total':>14} {'Liquidity':>12} {'Spread':>8} {'Outcomes':>8} {'Days':>6}")
    print(f"{'─'*3}  {'─'*55} {'─'*12} {'─'*14} {'─'*12} {'─'*8} {'─'*8} {'─'*6}")

    for i, m in enumerate(filtered[:top_n], 1):
        days_str = str(m['days_to_resolution']) if m['days_to_resolution'] >= 0 else "N/A"
        print(
            f"{i:>3}  {m['title']:<55} "
            f"${m['volume_24h']:>10,.0f} "
            f"${m['volume_total']:>12,.0f} "
            f"${m['liquidity']:>10,.0f} "
            f"{m['min_spread']:>7.3f} "
            f"{m['n_outcomes']:>8} "
            f"{days_str:>6}"
        )

    # ── Detailed breakdown of top 20 ───────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print(f"  DETAILED BREAKDOWN")
    print(f"{'=' * 120}")

    for i, m in enumerate(filtered[:top_n], 1):
        days_str = str(m['days_to_resolution']) if m['days_to_resolution'] >= 0 else "N/A"
        print(f"\n{'─' * 80}")
        print(f"  #{i}: {m['title']}")
        print(f"  Slug:       {m['slug']}")
        print(f"  Event ID:   {m['event_id']}")
        print(f"  Outcomes:   {m['n_outcomes']} | Token IDs: {m['n_token_ids']}")
        print(f"  Volume:     24h=${m['volume_24h']:,.0f}  |  1wk=${m['volume_1wk']:,.0f}  |  1mo=${m['volume_1mo']:,.0f}  |  total=${m['volume_total']:,.0f}")
        print(f"  Liquidity:  ${m['liquidity']:,.0f}")
        print(f"  Spread:     avg={m['avg_spread']:.4f}  |  min={m['min_spread']:.4f}")
        print(f"  Resolution: {m['end_date']}  ({days_str} days)")
        print(f"  Competitive: {m['competitive']:.4f}")
        print(f"  NegRisk:    {m['neg_risk']}")

        # Volume velocity
        if m['volume_1wk'] > 0:
            daily_avg = m['volume_1wk'] / 7
            velocity = m['volume_24h'] / daily_avg if daily_avg > 0 else 0
            print(f"  Vol Velocity: {velocity:.2f}x (24h vs 7d avg)")

    # ── Summary stats ──────────────────────────────────────────────────
    if filtered:
        print(f"\n\n{'=' * 80}")
        print(f"  SUMMARY")
        print(f"{'=' * 80}")

        total_vol_24h = sum(m['volume_24h'] for m in filtered)
        total_liq     = sum(m['liquidity'] for m in filtered)
        avg_spread_all = sum(m['min_spread'] for m in filtered) / len(filtered)

        print(f"  Qualifying events:    {len(filtered)}")
        print(f"  Total 24h volume:     ${total_vol_24h:,.0f}")
        print(f"  Total liquidity:      ${total_liq:,.0f}")
        print(f"  Avg min spread:       {avg_spread_all:.4f}")

        # Category breakdown by slug keywords
        categories = {}
        for m in filtered:
            slug = m['slug'].lower()
            if any(k in slug for k in ['president', 'election', 'senate', 'governor', 'congress', 'nominee', 'gop', 'democrat', 'republican']):
                cat = 'Politics'
            elif any(k in slug for k in ['nba', 'nfl', 'mlb', 'nhl', 'cbb', 'cfb', 'epl', 'soccer', 'tennis', 'ufc', 'boxing', 'mma', 'f1', 'nascar', 'ncaa', 'champions-league', 'premier-league', 'sport', 'match', 'game']):
                cat = 'Sports'
            elif any(k in slug for k in ['fed', 'rate', 'gdp', 'inflation', 'cpi', 'unemployment', 'recession', 'tariff', 'trade', 'economy', 'treasury']):
                cat = 'Macro/Econ'
            elif any(k in slug for k in ['war', 'ukraine', 'russia', 'china', 'taiwan', 'military', 'nato', 'ceasefire', 'conflict']):
                cat = 'Geopolitical'
            elif any(k in slug for k in ['elon', 'trump', 'celebrity', 'twitter', 'tweet', 'spotify', 'youtube', 'tiktok', 'oscar', 'grammy', 'emmy', 'super-bowl']):
                cat = 'Pop/Culture'
            else:
                cat = 'Other'
            categories.setdefault(cat, []).append(m)

        print(f"\n  Category Breakdown:")
        for cat, items in sorted(categories.items(), key=lambda x: -sum(i['volume_24h'] for i in x[1])):
            cat_vol = sum(i['volume_24h'] for i in items)
            cat_liq = sum(i['liquidity'] for i in items)
            print(f"    {cat:<15} {len(items):>4} events  |  24h vol: ${cat_vol:>12,.0f}  |  liq: ${cat_liq:>12,.0f}")

        # Market quality tiers
        print(f"\n  Quality Tiers:")
        tier_a = [m for m in filtered if m['volume_24h'] >= 500_000 and m['liquidity'] >= 100_000]
        tier_b = [m for m in filtered if 100_000 <= m['volume_24h'] < 500_000 and m['liquidity'] >= 50_000]
        tier_c = [m for m in filtered if m not in tier_a and m not in tier_b]
        print(f"    Tier A (Vol≥$500k, Liq≥$100k): {len(tier_a)} events")
        print(f"    Tier B (Vol≥$100k, Liq≥$50k):  {len(tier_b)} events")
        print(f"    Tier C (rest):                  {len(tier_c)} events")

        if tier_a:
            print(f"\n  Tier A Markets (best candidates for data collection):")
            for m in tier_a:
                print(f"    • {m['title'][:60]:60s} vol24h=${m['volume_24h']:>10,.0f}  liq=${m['liquidity']:>10,.0f}  spread={m['min_spread']:.3f}")

    print(f"\n{'=' * 80}")
    print(f"  Scan complete.")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
