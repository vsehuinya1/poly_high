"""
Flashscore vs Polymarket Timing Analysis

Uses EXISTING data to answer: does Polymarket reprice BEFORE or AFTER
the Flashscore feed detects a score change?

Approach:
  - Flashscore polls every 3s. Score changes are detected at poll boundaries.
  - Polymarket ticks record continuous price updates.
  - For each big Polymarket price move (>5c) = score change event:
    - Find the FIRST tick where the move started
    - This is when Polymarket market makers had the information
  - Compare this to Flashscore poll timing to estimate if we're faster or slower

Also: check if there are PREDICTIVE price moves (Polymarket starting to move
BEFORE the big jump), which would indicate informed trading.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

conn = sqlite3.connect("sports_data/tick_history.db")
cur = conn.cursor()

print("=" * 70)
print("POLYMARKET REPRICING DYNAMICS — WHO MOVES FIRST?")
print("=" * 70)

# Find all tokens with enough ticks for analysis
cur.execute("""
    SELECT t.token_id, COUNT(*) as tick_count, tl.market_title
    FROM ticks t JOIN token_labels tl ON t.token_id=tl.token_id
    WHERE tl.sport='tennis' AND t.mid > 0.10 AND t.mid < 0.90
    GROUP BY t.token_id HAVING tick_count > 100
    ORDER BY tick_count DESC LIMIT 200
""")
active_tokens = cur.fetchall()
print(f"  Active tokens with >100 ticks: {len(active_tokens)}")

# For each token, find big moves (score changes)
big_moves = []
for tid, cnt, title in active_tokens:
    cur.execute("""
        SELECT timestamp, mid, spread FROM ticks
        WHERE token_id=? ORDER BY timestamp
    """, (tid,))
    ticks = cur.fetchall()
    
    for i in range(5, len(ticks)-10):
        ts, mid, sp = ticks[i]
        prev_ts, prev_mid, _ = ticks[i-1]
        
        delta = mid - prev_mid
        if abs(delta) < 0.05:
            continue
        
        # Skip if too close to last big move
        if big_moves and big_moves[-1]["tid"] == tid and ts - big_moves[-1]["ts"] < 60:
            continue
        
        # Pre-move analysis: what happened in the 30s BEFORE this big tick?
        pre_ticks = [(t, m) for t, m, s in ticks[max(0,i-20):i] if ts - t <= 30]
        
        # Was price already drifting in the direction of the big move?
        pre_drift = 0
        if len(pre_ticks) >= 3:
            pre_drift = pre_ticks[-1][1] - pre_ticks[0][1]
            if delta < 0:
                pre_drift = -pre_drift  # normalize: positive = drifting in move direction
        
        # Post-move: how much further does it go?
        post_ticks = [(t, m) for t, m, s in ticks[i:i+20] if t - ts <= 60]
        post_continuation = 0
        if len(post_ticks) >= 2:
            post_continuation = post_ticks[-1][1] - mid
            if delta < 0:
                post_continuation = -post_continuation  # normalize
        
        # Time gap between consecutive ticks around the big move
        tick_gap = ts - prev_ts
        
        # Spread at moment of big move
        spread_at_move = sp
        
        big_moves.append({
            "tid": tid, "ts": ts, "mid": mid, "delta": delta,
            "abs_delta": abs(delta), "pre_drift": pre_drift,
            "post_continuation": post_continuation,
            "tick_gap": tick_gap, "spread": spread_at_move,
            "title": title,
        })

print(f"  Score-change events found: {len(big_moves)}")

if not big_moves:
    print("NO BIG MOVES FOUND"); conn.close(); exit()

# Analysis 1: Pre-drift (informed trading before the big move?)
print(f"\n{'='*70}")
print("1. PRE-DRIFT ANALYSIS — Is price moving BEFORE the big jump?")
print(f"{'='*70}")
pre_drift_positive = sum(1 for m in big_moves if m["pre_drift"] > 0.005)
pre_drift_negative = sum(1 for m in big_moves if m["pre_drift"] < -0.005)
pre_drift_flat = len(big_moves) - pre_drift_positive - pre_drift_negative
avg_pre = sum(m["pre_drift"] for m in big_moves) / len(big_moves)

print(f"  Events: {len(big_moves)}")
print(f"  Price drifting toward move: {pre_drift_positive} ({pre_drift_positive/len(big_moves)*100:.0f}%)")
print(f"  Price drifting against move: {pre_drift_negative} ({pre_drift_negative/len(big_moves)*100:.0f}%)")
print(f"  Flat pre-move: {pre_drift_flat} ({pre_drift_flat/len(big_moves)*100:.0f}%)")
print(f"  Avg pre-drift (normalized): {avg_pre:+.4f}")
if avg_pre > 0.005:
    print(f"  → INFORMED TRADING DETECTED — price starts moving before the big jump")
    print(f"  → This means market makers have faster data than the tick feed")
elif avg_pre < -0.005:
    print(f"  → COUNTER-MOVE before big jump — possible fake-out pattern")
else:
    print(f"  → No pre-drift — big moves arrive as sudden jumps")

# Analysis 2: Post-continuation
print(f"\n{'='*70}")
print("2. POST-CONTINUATION — Does the move keep going after the jump?")
print(f"{'='*70}")
post_pos = sum(1 for m in big_moves if m["post_continuation"] > 0.005)
post_neg = sum(1 for m in big_moves if m["post_continuation"] < -0.005)
avg_post = sum(m["post_continuation"] for m in big_moves) / len(big_moves)
print(f"  Continues in direction: {post_pos} ({post_pos/len(big_moves)*100:.0f}%)")
print(f"  Reverses: {post_neg} ({post_neg/len(big_moves)*100:.0f}%)")
print(f"  Avg post-continuation: {avg_post:+.4f}")
if avg_post > 0.005:
    print(f"  → Edge: BUY after the jump, price keeps going")
elif avg_post < -0.005:
    print(f"  → Edge: FADE the jump, price reverses")
else:
    print(f"  → No edge after the jump — price reaches fair value instantly")

# Analysis 3: Tick gap (how suddenly does the big move arrive?)
print(f"\n{'='*70}")
print("3. TICK GAP — How long between last normal tick and big-move tick?")
print(f"{'='*70}")
gaps = [m["tick_gap"] for m in big_moves]
avg_gap = sum(gaps)/len(gaps)
for threshold in [1, 3, 5, 10, 30]:
    n = sum(1 for g in gaps if g <= threshold)
    print(f"  Gap <= {threshold}s: {n} ({n/len(gaps)*100:.0f}%)")
print(f"  Avg gap: {avg_gap:.1f}s")
print(f"  Median gap: {sorted(gaps)[len(gaps)//2]:.1f}s")
if avg_gap < 5:
    print(f"  → Big moves arrive within seconds — consistent with 3s polling feed")
else:
    print(f"  → Longer gaps — suggests price updates are batched or delayed")

# Analysis 4: Spread dynamics during big moves
print(f"\n{'='*70}")
print("4. SPREAD AT BIG MOVES — Is liquidity present when you need it?")
print(f"{'='*70}")
spreads = [m["spread"] for m in big_moves if m["spread"] > 0]
if spreads:
    avg_sp = sum(spreads)/len(spreads)
    for threshold in [0.01, 0.02, 0.03, 0.05, 0.10]:
        n = sum(1 for s in spreads if s <= threshold)
        print(f"  Spread <= {threshold}: {n} ({n/len(spreads)*100:.0f}%)")
    print(f"  Avg spread at big move: {avg_sp:.4f}")

# Analysis 5: Simulated $100 — buy at big move tick, hold 30s/60s
print(f"\n{'='*70}")
print("5. SIMULATED $100 — RIDE THE BIG MOVE")
print(f"{'='*70}")
print("   If we could detect the big move AS it happens (0s latency):")
print("   Entry at the big-move tick price + spread/2 (ask)")
print("   Exit at +30s/+60s at mid - spread/2 (bid)")

for hold in [10, 30, 60]:
    trades_r = []
    for m in big_moves:
        if m["delta"] < 0:
            continue  # only buy UP moves (can't short on Poly)
        entry = m["mid"] + m["spread"]/2
        
        cur.execute("""
            SELECT mid, spread FROM ticks WHERE token_id=?
            AND timestamp BETWEEN ? AND ?
            ORDER BY ABS(timestamp-?) LIMIT 1
        """, (m["tid"], m["ts"]+hold-3, m["ts"]+hold+3, m["ts"]+hold))
        r = cur.fetchone()
        if not r: continue
        exit_bid = r[0] - r[1]/2
        pnl_r = (exit_bid - entry) / entry
        trades_r.append(pnl_r)
    
    if not trades_r: continue
    avg = sum(trades_r)/len(trades_r)
    wins = [r for r in trades_r if r > 0.001]
    losses = [r for r in trades_r if r < -0.001]
    wr = len(wins)/(len(wins)+len(losses)) if (wins or losses) else 0
    
    # Kelly
    kelly = 0
    if wins and losses:
        p = len(wins)/(len(wins)+len(losses))
        aw = sum(wins)/len(wins)
        al = abs(sum(losses)/len(losses))
        b = aw/al if al > 0 else 1
        kelly = max(0, (p*b-(1-p))/b)
    
    # Equity
    qk = kelly/4
    eq = 100.0
    eq_hi = 100.0
    max_dd = 0.0
    for r in trades_r:
        eq += qk * eq * r
        if eq > eq_hi: eq_hi = eq
        dd = (eq_hi-eq)/eq_hi if eq_hi > 0 else 0
        if dd > max_dd: max_dd = dd
        if eq <= 0: eq = 0; break
    
    print(f"\n  Hold {hold}s: n={len(trades_r)} avg_R={avg:+.2%} wr={wr:.0%} kelly={kelly:.1%} | $100→${eq:,.2f} (DD {max_dd:.0%})")

conn.close()
print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")
