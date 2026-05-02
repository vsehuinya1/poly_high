"""
TSLB Corrected Backtest — Realistic Constraints

Fixes from naive version:
  1. Enter at ASK (mid + spread/2), not mid
  2. Max 10 concurrent positions
  3. No overlapping entries on same token (must exit before re-entering)
  4. Results broken down by entry price bucket
  5. Separate match-ending catastrophic losses (exit <= 0.005)
  6. Hold = 600s, trailing stop = 2c, 2R cap
"""
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

conn = sqlite3.connect("sports_data/tick_history.db")
cur = conn.cursor()

MAX_PRICE = 0.20
MAX_SPREAD = 0.03
MIN_PRICE = 0.03
DEDUP_S = 90.0
HOLD_S = 600.0
TRAILING_STOP = 0.02
MAX_CONCURRENT = 10

print("=" * 70)
print("TSLB CORRECTED BACKTEST — Realistic Constraints")
print("=" * 70)

# Load candidates
cur.execute("""
    SELECT t.token_id, t.timestamp, t.mid, t.spread, tl.market_title
    FROM ticks t JOIN token_labels tl ON t.token_id = tl.token_id
    WHERE tl.sport = 'tennis' AND t.mid > ? AND t.mid < ?
      AND t.spread > 0 AND t.spread <= ?
    ORDER BY t.timestamp ASC
""", (MIN_PRICE, MAX_PRICE, MAX_SPREAD))
candidates = cur.fetchall()
print(f"  Raw candidates: {len(candidates):,}")

# Simulate with realistic constraints
last_entry_ts = {}      # token_id → last entry ts (dedup)
active_tokens = {}      # token_id → exit_ts (no overlap)
active_count = 0        # current concurrent
trades = []
skipped = {"dedup": 0, "concurrent": 0, "overlap": 0, "no_path": 0}

for tid, ts, mid, spread, title in candidates:
    # Dedup: same token within 90s
    if ts - last_entry_ts.get(tid, 0) < DEDUP_S:
        skipped["dedup"] += 1
        continue

    # No overlapping: if this token has an active trade, skip
    if tid in active_tokens and ts < active_tokens[tid]:
        skipped["overlap"] += 1
        continue

    # Concurrent cap
    # Clean expired positions
    active_tokens = {t: exp for t, exp in active_tokens.items() if ts < exp}
    if len(active_tokens) >= MAX_CONCURRENT:
        skipped["concurrent"] += 1
        continue

    last_entry_ts[tid] = ts

    # Get forward path
    cur.execute("""
        SELECT timestamp - ?, mid FROM ticks
        WHERE token_id = ? AND timestamp > ? AND timestamp <= ?
        ORDER BY timestamp
    """, (ts, tid, ts, ts + HOLD_S + 30))
    path = cur.fetchall()
    if len(path) < 3:
        skipped["no_path"] += 1
        continue

    # REALISTIC ENTRY: buy at ask = mid + spread/2
    entry = mid + spread / 2.0

    peak = mid  # track mid for trailing stop
    exit_price = mid
    exit_reason = "TIMEOUT"
    exit_elapsed = HOLD_S
    catastrophic = False

    for dt, price in path:
        if price > peak:
            peak = price
        # Trailing stop (based on mid movement)
        if peak > mid and peak - price >= TRAILING_STOP:
            exit_price = price
            exit_reason = "TRAIL"
            exit_elapsed = dt
            break
        # 2R hit (mid doubled from our entry)
        if price >= entry * 2:
            exit_price = price
            exit_reason = "2R"
            exit_elapsed = dt
            break
        # Timeout
        if dt >= HOLD_S:
            exit_price = price
            exit_reason = "TIMEOUT"
            exit_elapsed = dt
            break

    # REALISTIC EXIT: sell at bid = exit_mid - spread/2
    # Approximate: use half the entry spread as exit slippage
    exit_price_real = max(0.001, exit_price - spread / 2.0)

    if exit_price_real <= 0.005:
        catastrophic = True

    pnl = exit_price_real - entry
    r_pct = pnl / entry if entry > 0 else 0

    # Mark token as occupied until exit
    active_tokens[tid] = ts + exit_elapsed

    trades.append({
        "ts": ts, "token": tid, "title": title,
        "entry_mid": mid, "entry_ask": entry, "spread": spread,
        "exit_mid": exit_price, "exit_bid": exit_price_real,
        "pnl": pnl, "r_pct": r_pct, "peak": peak,
        "reason": exit_reason, "hold_s": exit_elapsed,
        "catastrophic": catastrophic,
    })

print(f"  Trades executed: {len(trades)}")
print(f"  Skipped: {skipped}")

if not trades:
    print("NO TRADES"); conn.close(); exit()

# ── Overall Stats ──
wins = [t for t in trades if t["r_pct"] > 0.005]
losses = [t for t in trades if t["r_pct"] < -0.005]
flat = [t for t in trades if abs(t["r_pct"]) <= 0.005]
cats = [t for t in trades if t["catastrophic"]]

all_r = [t["r_pct"] for t in trades]
avg_r = sum(all_r) / len(all_r)

print(f"\n{'='*70}")
print(f"OVERALL ({len(trades)} trades)")
print(f"{'='*70}")
print(f"  Wins:   {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
print(f"  Flat:   {len(flat)} ({len(flat)/len(trades)*100:.1f}%)")
print(f"  Catastrophic (→0): {len(cats)} ({len(cats)/len(trades)*100:.1f}%)")
print(f"  Avg R:  {avg_r:+.2%}")
print(f"  Avg slippage cost: {sum(t['spread'] for t in trades)/len(trades):.4f}")

if wins:
    print(f"  Avg win R:  {sum(t['r_pct'] for t in wins)/len(wins):+.2%}")
if losses:
    print(f"  Avg loss R: {sum(t['r_pct'] for t in losses)/len(losses):+.2%}")

# By exit reason
print(f"\n  By exit reason:")
by_reason = defaultdict(list)
for t in trades:
    by_reason[t["reason"]].append(t["r_pct"])
for reason, rs in sorted(by_reason.items()):
    avg = sum(rs) / len(rs)
    wr = sum(1 for r in rs if r > 0.005) / len(rs) if rs else 0
    print(f"    {reason:>8s}: n={len(rs):>5} avg_R={avg:+.2%} wr={wr:.0%}")

# ── Price Bucket Breakdown ──
print(f"\n{'='*70}")
print(f"BY ENTRY PRICE BUCKET")
print(f"{'='*70}")
buckets = [(0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 0.20)]
for lo, hi in buckets:
    bt = [t for t in trades if lo <= t["entry_mid"] < hi]
    if not bt:
        print(f"  [{lo:.2f}, {hi:.2f}): no trades")
        continue
    avg = sum(t["r_pct"] for t in bt) / len(bt)
    w = sum(1 for t in bt if t["r_pct"] > 0.005)
    l = sum(1 for t in bt if t["r_pct"] < -0.005)
    wr = w / (w + l) if (w + l) else 0
    hits_2r = sum(1 for t in bt if t["reason"] == "2R")
    cat_n = sum(1 for t in bt if t["catastrophic"])
    sp_pct = sum(t["spread"]/t["entry_mid"] for t in bt) / len(bt)
    print(f"  [{lo:.2f}, {hi:.2f}): n={len(bt):>5} | avg_R={avg:+.2%} | wr={wr:.0%} | 2R={hits_2r} | cat={cat_n} | spread_as_%={sp_pct:.0%}")

# ── Excluding Catastrophic ──
print(f"\n{'='*70}")
print(f"EXCLUDING CATASTROPHIC LOSSES (match endings)")
print(f"{'='*70}")
clean = [t for t in trades if not t["catastrophic"]]
if clean:
    avg_clean = sum(t["r_pct"] for t in clean) / len(clean)
    w_c = sum(1 for t in clean if t["r_pct"] > 0.005)
    l_c = sum(1 for t in clean if t["r_pct"] < -0.005)
    wr_c = w_c / (w_c + l_c) if (w_c + l_c) else 0
    print(f"  Trades: {len(clean)} | Avg R: {avg_clean:+.2%} | WR: {wr_c:.0%}")

# ── Kelly ──
if wins and losses:
    p = len(wins) / (len(wins) + len(losses))
    avg_w = sum(t["r_pct"] for t in wins) / len(wins)
    avg_l = abs(sum(t["r_pct"] for t in losses) / len(losses))
    b = avg_w / avg_l if avg_l > 0 else 1
    kelly = max(0, (p * b - (1 - p)) / b)

    print(f"\n{'='*70}")
    print(f"KELLY (with slippage)")
    print(f"{'='*70}")
    print(f"  Win rate:       {p:.2%}")
    print(f"  Avg win:        {avg_w:+.2%}")
    print(f"  Avg loss:       {avg_l:+.2%}")
    print(f"  W/L ratio:      {b:.2f}")
    print(f"  Full Kelly:     {kelly:.2%}")
    print(f"  Half Kelly:     {kelly/2:.2%}")

    # Equity
    print(f"\n{'='*70}")
    print(f"EQUITY CURVE — $100 start")
    print(f"{'='*70}")
    for frac, label in [(kelly/2, "HALF"), (kelly/4, "QUARTER"), (kelly/8, "EIGHTH")]:
        if frac <= 0: continue
        eq = 100.0
        eq_hi = 100.0
        max_dd = 0.0
        for t in trades:
            eq += frac * eq * t["r_pct"]
            if eq > eq_hi: eq_hi = eq
            dd = (eq_hi - eq) / eq_hi if eq_hi > 0 else 0
            if dd > max_dd: max_dd = dd
            if eq <= 0: eq = 0; break
        print(f"  {label} KELLY ({frac:.2%}): ${eq:,.2f} | max_dd={max_dd:.1%}")

    # Monthly
    print(f"\n{'='*70}")
    print(f"MONTHLY (Quarter Kelly)")
    print(f"{'='*70}")
    qk = kelly / 4
    eq = 100.0
    by_month = defaultdict(list)
    for t in trades:
        m = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%Y-%m")
        by_month[m].append(t)
    for m in sorted(by_month.keys()):
        ms = eq
        for t in by_month[m]:
            eq += qk * eq * t["r_pct"]
            if eq <= 0: eq = 0; break
        mr = (eq - ms) / ms if ms > 0 else 0
        n = len(by_month[m])
        wr = sum(1 for t in by_month[m] if t["r_pct"] > 0.005) / n if n else 0
        print(f"  {m}: {n:>4} trades | ${eq:>12,.2f} | R={mr:+.1%} | wr={wr:.0%}")

conn.close()
print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
