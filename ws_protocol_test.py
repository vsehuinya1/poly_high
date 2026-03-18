#!/usr/bin/env python3
"""Three controlled WS protocol tests to isolate the subscription mismatch."""
import asyncio
import websockets
import json
import aiohttp
import ast
import sys

WS_URI = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POLITICAL_TOKEN = "51338236787729560681434534660841415073585974762690814047670810862722808070955"

async def ws_test(label, sub_payload, timeout=8):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    print(f"  OUT: {json.dumps(sub_payload)[:300]}")
    try:
        async with websockets.connect(WS_URI, ping_interval=20, close_timeout=5) as ws:
            await ws.send(json.dumps(sub_payload))
            for i in range(5):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    print(f"  IN[{i}]: {msg[:400]}")
                    if "INVALID" in str(msg):
                        print(f"  >>> REJECTED")
                        return "REJECTED"
                    if msg.strip() == "[]":
                        continue
                    try:
                        data = json.loads(msg)
                        if isinstance(data, list) and len(data) > 0:
                            print(f"  >>> SUCCESS - {len(data)} items, keys={list(data[0].keys())[:8]}")
                            return "SUCCESS"
                    except:
                        pass
                except asyncio.TimeoutError:
                    print(f"  IN[{i}]: TIMEOUT")
                    return "TIMEOUT"
    except Exception as e:
        print(f"  ERROR: {e}")
        return "ERROR"

async def main():
    # TEST 1: Political token
    r1 = await ws_test("1. POLITICAL TOKEN (control)", {"assets_ids": [POLITICAL_TOKEN]})

    # Fetch ONE NBA market from Gamma
    print(f"\n\n{'#'*60}")
    print("Fetching NBA market from Gamma API...")
    nba_market = None
    async with aiohttp.ClientSession() as s:
        url = "https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=50&order=volume24hr&ascending=false"
        async with s.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            all_markets = await r.json()

        nba_kw = ["spurs","pistons","kings","grizzlies","rockets","jazz",
                   "pacers","lakers","celtics","knicks","warriors","76ers",
                   "bulls","heat","nets","hawks","magic","clippers",
                   "nuggets","suns","mavs","bucks","cavaliers","timberwolves"]
        for m in all_markets:
            q = (m.get("question","") + " " + m.get("groupItemTitle","")).lower()
            if any(w in q for w in nba_kw):
                nba_market = m
                break

    if not nba_market:
        print("No NBA market found!")
        sys.exit(1)

    # DUMP full object
    print(f"\nFULL MARKET OBJECT:")
    for k in sorted(nba_market.keys()):
        v = str(nba_market[k])
        if len(v) > 150: v = v[:150] + "..."
        print(f"  {k}: {v}")

    # Extract every ID
    clob_raw = nba_market.get("clobTokenIds", "")
    if isinstance(clob_raw, str):
        try: clob_tokens = ast.literal_eval(clob_raw) if clob_raw.startswith("[") else [clob_raw]
        except: clob_tokens = [clob_raw]
    else:
        clob_tokens = clob_raw or []

    ids = {
        "clobTokenIds[0]": clob_tokens[0] if clob_tokens else "",
        "conditionId": nba_market.get("conditionId", ""),
        "questionID": nba_market.get("questionID", ""),
        "negRiskMarketID": nba_market.get("negRiskMarketID", ""),
    }

    print(f"\nEXTRACTED IDs:")
    for k, v in ids.items():
        print(f"  {k}: {v}")

    # TEST 2: Single clobTokenId
    if clob_tokens:
        await ws_test(f"2. SINGLE clobTokenId", {"assets_ids": [clob_tokens[0]]})

    # TEST 3: Try every ID field
    for field_name, val in ids.items():
        if val and val != ids.get("clobTokenIds[0]", ""):  # skip duplicate
            await ws_test(f"3. field={field_name}", {"assets_ids": [val]})

    # TEST 3x: conditionId with type=market
    cond = ids.get("conditionId", "")
    if cond:
        await ws_test("3x. conditionId + type=market", {"type": "market", "assets_ids": [cond]})

    # SUMMARY
    print(f"\n\n{'='*60}")
    print("DONE — check results above")
    print(f"{'='*60}")

asyncio.run(main())
