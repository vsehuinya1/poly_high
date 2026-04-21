"""
Cricket Tick-Based Momentum Strategy — v2.0

Replaces mean-reversion (v1.0, proven ΣR -2.75) with momentum continuation.

Two signal types:
  1. SPIKE_CONTINUATION  — ride confirmed spikes (≥0.04 in ≤60s)
  2. MOMENTUM_DRIFT      — ride sustained drift (≥0.04 over 120-300s)

IPL-only market filtering. No ESPN, no external feeds.
Only uses: price (mid), spread, timestamp.

Exit logic:
  - STOP_LOSS: -0.06
  - EXIT_RUNNER: MFE ≥ 0.03, trail 0.02
  - EXIT_MOMENTUM_FAIL: no new high/low for 45s after entry
  - TIMEOUT: 300s

v2.0 — 2026-04-06  Continuation model (replaces reversion)
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("cricket.tick_strategy")


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

# Ring buffer
MAX_TICK_HISTORY = 600  # ~10 min at 1 tick/s

# Guards
PRICE_FLOOR = 0.20
PRICE_CEIL = 0.80
STALE_TICK_S = 60.0

# Spike Continuation
SPIKE_MIN_MOVE = 0.02       # minimum price move for spike
SPIKE_WINDOW_S = 120.0      # v8.5: widened from 60s for REST polling cadence (~72s)
SPIKE_CONFIRM_MIN_S = 5.0   # wait at least 5s after spike
SPIKE_CONFIRM_MAX_S = 10.0  # don't wait longer than 10s
SPIKE_MAX_RETRACE = 0.33    # retrace must be < 33% of spike
SPIKE_MAX_SPREAD = 0.04     # spread must be ≤ 0.04 at entry
SPIKE_MIN_CONTRACTION = 2   # spread must contract for ≥ 2 consecutive ticks

# Momentum Drift
DRIFT_MIN_MOVE = 0.015      # net move over window
DRIFT_WINDOW_MIN_S = 120.0  # minimum lookback
DRIFT_WINDOW_MAX_S = 300.0  # maximum lookback
DRIFT_SMOOTH_RATIO = 0.60   # net/range > 0.60
DRIFT_MAX_SPREAD = 0.03     # spread at entry
DRIFT_CONFIRM_TICKS = 3     # last N ticks same direction

# Edge Calculation
MIN_EDGE = 0.01             # reject if edge < this

# Exits
STOP_LOSS = 0.06            # initial stop distance
TIMEOUT_S = 300.0           # 5 minutes
RUNNER_ACTIVATION = 0.03    # MFE to activate runner
RUNNER_TRAIL = 0.02         # trail distance once active
MOMENTUM_FAIL_S = 45.0      # exit if no new extreme for this long

# Cooldown
COOLDOWN_S = 120.0

# v8.2: Pullback Entry
PULLBACK_OFFSET = 0.008     # enter 0.8¢ below spike peak
PULLBACK_WINDOW_S = 30.0    # must fill within 30s
PULLBACK_MAX_SPREAD = 0.03  # spread check at fill

# v8.2: Continuation Clock
CONTINUATION_S = 90.0       # must see +0.02 within 90s of entry
CONTINUATION_MIN_MOVE = 0.02

# v8.2: Regime Filter
REGIME_LOOKBACK_S = 60.0    # look back 60s for momentum
REGIME_MIN_MOVE = 0.02      # require 2¢ move in last 60s

# v8.2: Match Cooldown
MATCH_COOLDOWN_S = 180.0    # 3 minutes between trades per match

# Market filter
IPL_KEYWORDS = ["indian premier league"]


# ═══════════════════════════════════════════════════════════════════════
#  Data Types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TickPoint:
    timestamp: float
    mid: float
    spread: float


@dataclass
class CricketTickSignal:
    """Output from the tick-based detector."""
    signal_type: str        # SPIKE_CONTINUATION or MOMENTUM_DRIFT
    match_id: str
    direction: str          # LONG or SHORT — same as move direction
    move: float             # observed move magnitude
    entry_price: float      # current mid at signal time
    fair_price: float       # projected fair value (continuation)
    edge: float             # projected_move - spread/2
    spread: float           # current spread


@dataclass
class PendingPullback:
    """v8.2: Pending pullback entry waiting for fill."""
    signal: CricketTickSignal
    spike_peak: float           # price at spike detection
    limit_price: float          # entry limit = peak - PULLBACK_OFFSET
    created_ts: float           # when pending was created
    direction: str              # LONG or SHORT


@dataclass
class TickTrade:
    """Active trade with runner tracking."""
    match_id: str
    signal_type: str
    entry_price: float
    entry_timestamp: float
    direction: str          # LONG or SHORT
    stop: float             # absolute stop price
    timeout_s: float
    # Runner state
    mfe: float = 0.0                # best favorable excursion
    mfe_timestamp: float = 0.0      # when MFE was set
    runner_active: bool = False
    trail_price: float = 0.0        # trailing stop price
    last_extreme_ts: float = 0.0    # last new high/low timestamp
    tick_count: int = 0


# ═══════════════════════════════════════════════════════════════════════
#  Detector
# ═══════════════════════════════════════════════════════════════════════

class CricketTickDetector:
    """Momentum continuation detector for cricket markets.

    Detects spike continuation and momentum drift patterns.
    Manages runner-style exits with momentum-fail protection.
    """

    def __init__(self):
        self._ticks: dict[str, deque[TickPoint]] = {}
        self._trades: dict[str, TickTrade] = {}
        self._last_signal_ts: dict[str, float] = {}
        self._market_titles: dict[str, str] = {}  # match_id → title
        self._pending: dict[str, PendingPullback] = {}  # v8.2: pending pullback entries
        self._last_trade_exit_ts: dict[str, float] = {}  # v8.2: match cooldown
        # v8.5: Stateful mid tracking for sparse REST data
        self._last_mid: dict[str, float] = {}     # match_id → last distinct mid
        self._last_mid_ts: dict[str, float] = {}  # match_id → timestamp of last change
        # v9.4: Continuation confirmation — pending spikes awaiting next-tick verify
        self._pending_confirm: dict[str, dict] = {}  # match_id → {signal, peak, dir, ts}
        self._skip_counts: dict[str, int] = {
            "no_move": 0, "spread": 0, "band": 0, "cooldown": 0,
            "not_ipl": 0, "small_spike": 0, "no_contraction": 0,
            "low_edge": 0, "retrace": 0, "too_early": 0,
            "low_momentum": 0, "match_cooldown": 0, "pullback_expired": 0,
            "score_filter": 0, "continuation_fail": 0,
        }
        self._last_skip_log = 0.0

    # ── Tick Processing ─────────────────────────────────────────────

    def on_tick(
        self,
        match_id: str,
        mid: float,
        spread: float,
        timestamp: float,
        market_title: str = "",
        match_state: object = None,
        observe_only: bool = False,
    ) -> Optional[CricketTickSignal]:
        """Process a new tick. Returns a signal if one is detected.

        Args:
            observe_only: If True, run spike detection for diagnostics
                but do NOT create entries or pullbacks.  Used when
                match_state is None (no score context).
        """
        # Store tick
        if match_id not in self._ticks:
            self._ticks[match_id] = deque(maxlen=MAX_TICK_HISTORY)
        buf = self._ticks[match_id]
        buf.append(TickPoint(timestamp=timestamp, mid=mid, spread=spread))

        # v8.5: Track distinct mid changes for sparse REST data
        prev_mid = self._last_mid.get(match_id)
        if prev_mid is None or abs(mid - prev_mid) >= 0.001:
            self._last_mid[match_id] = mid
            self._last_mid_ts[match_id] = timestamp

        # Store title for IPL filtering
        if market_title:
            self._market_titles[match_id] = market_title

        # ── Guards ────────────────────────────────────────────────

        # IPL filter
        title = self._market_titles.get(match_id, "")
        if not any(kw in title.lower() for kw in IPL_KEYWORDS):
            self._skip("not_ipl")
            return None

        # Active trade for this market
        if match_id in self._trades:
            return None

        # v8.2: Check pending pullback fill
        if match_id in self._pending:
            pending = self._pending[match_id]
            age = timestamp - pending.created_ts
            if age > PULLBACK_WINDOW_S:
                log.info("CRICKET_PULLBACK_EXPIRED | %s | age=%.1fs | peak=%.4f limit=%.4f mid=%.4f",
                         match_id, age, pending.spike_peak, pending.limit_price, mid)
                self._skip("pullback_expired")
                del self._pending[match_id]
            elif spread <= PULLBACK_MAX_SPREAD:
                # Check fill: for LONG, price must dip to limit; for SHORT, rise to limit
                filled = False
                if pending.direction == "LONG" and mid <= pending.limit_price:
                    filled = True
                elif pending.direction == "SHORT" and mid >= pending.limit_price:
                    filled = True
                if filled:
                    # Adjust entry price to actual fill level
                    fill_signal = pending.signal
                    fill_signal.entry_price = mid
                    log.info("CRICKET_PULLBACK_ENTRY | %s | peak=%.4f limit=%.4f fill=%.4f | spread=%.4f",
                             match_id, pending.spike_peak, pending.limit_price, mid, spread)
                    del self._pending[match_id]
                    return fill_signal
            return None

        # v9.4: Check pending continuation confirmation
        if match_id in self._pending_confirm:
            pc = self._pending_confirm[match_id]
            age = timestamp - pc['ts']
            peak = pc['peak']
            direction = pc['dir']
            # Timeout: no confirmation tick within 15s → reject
            if age > 15.0:
                log.info(
                    "CRICKET_SPIKE_REJECT | %s | reason=CONTINUATION_TIMEOUT "
                    "| dir=%s | peak=%.4f | age=%.1fs",
                    match_id, direction, peak, age,
                )
                self._skip("continuation_fail")
                del self._pending_confirm[match_id]
                return None
            # Check continuation on this tick
            if direction == "LONG":
                drop = peak - mid
                passed = mid >= peak - 0.003
            else:  # SHORT
                drop = mid - peak
                passed = mid <= peak + 0.003
            if not passed:
                log.info(
                    "CRICKET_SPIKE_REJECT | %s | reason=CONTINUATION_FAIL "
                    "| dir=%s | peak=%.4f | next=%.4f | drop=%.4f",
                    match_id, direction, peak, mid, drop,
                )
                self._skip("continuation_fail")
                del self._pending_confirm[match_id]
                return None
            # ── v9.5: Continuation confirmed → MARKET ENTRY (bypass pullback) ──
            log.info(
                "CRICKET_CONTINUATION_PASS | %s | dir=%s | peak=%.4f | next=%.4f",
                match_id, direction, peak, mid,
            )
            sig = pc['signal']
            sig.entry_price = mid
            log.info(
                "CRICKET_MARKET_ENTRY | %s | dir=%s | entry=%.4f "
                "| peak=%.4f | confirm_price=%.4f | reason=POST_CONTINUATION",
                match_id, direction, mid, peak, mid,
            )
            del self._pending_confirm[match_id]
            return sig

        # Price band
        if mid < PRICE_FLOOR or mid > PRICE_CEIL:
            self._skip("band")
            return None

        # Staleness
        if len(buf) >= 2:
            last_change_ts = timestamp
            for i in range(len(buf) - 2, -1, -1):
                if abs(buf[i].mid - mid) > 0.001:
                    last_change_ts = buf[i].timestamp
                    break
            if timestamp - last_change_ts > STALE_TICK_S:
                self._skip("no_move")
                return None

        # Cooldown
        last_sig = self._last_signal_ts.get(match_id, 0)
        if timestamp - last_sig < COOLDOWN_S:
            self._skip("cooldown")
            return None

        # v8.2: Match cooldown — 3min between trades per match
        last_exit = self._last_trade_exit_ts.get(match_id, 0)
        if timestamp - last_exit < MATCH_COOLDOWN_S:
            self._skip("match_cooldown")
            return None

        # Need enough data
        if len(buf) < 10:
            return None

        # ── v8.4: Detection FIRST, quality gates AFTER ─────────────
        # Run spike detection on ALL ticks (regardless of spread/regime)
        signal = self._check_spike_continuation(match_id, buf, mid, spread, timestamp)
        if signal:
            # ── SPIKE CANDIDATE FOUND — apply quality gates ────────
            log.info(
                "CRICKET_SPIKE_CANDIDATE | %s | dir=%s | move=%.4f "
                "| spread=%.4f | mid=%.4f | age=%.1fs",
                match_id, signal.direction, signal.move,
                spread, mid, timestamp - buf[0].timestamp,
            )

            # Quality gate 1: Score filter (v8.4)
            if signal and match_state is not None:
                _is_live = getattr(match_state, 'is_live', False)
                _overs = getattr(match_state, 'overs', 0.0)
                _recent_wicket = getattr(match_state, 'had_recent_wicket', False)
                if not _is_live:
                    log.info(
                        "CRICKET_SPIKE_REJECT | %s | reason=score_filter_not_live "
                        "| move=%.4f | spread=%.4f",
                        match_id, signal.move, spread,
                    )
                    self._skip("score_filter")
                    signal = None
                elif _overs < 2:
                    # v8.6: Early shock regime — allow big SHORT spikes even < 2 overs
                    _early_shock = (
                        _overs >= 0
                        and abs(signal.move) >= 0.05
                        and spread <= 0.02
                        and signal.direction == "SHORT"
                    )
                    if not _early_shock:
                        log.info(
                            "CRICKET_SPIKE_REJECT | %s | reason=score_filter_early_overs "
                            "| overs=%.1f | move=%.4f",
                            match_id, _overs, signal.move,
                        )
                        self._skip("score_filter")
                        signal = None
                    else:
                        log.info(
                            "CRICKET_EARLY_SHOCK_ACCEPTED | %s "
                            "| overs=%.1f | move=%.4f | spread=%.4f",
                            match_id, _overs, signal.move, spread,
                        )
                elif not (_overs >= 16 or _recent_wicket):
                    # v8.6: Early shock regime for overs 2–6
                    _early_shock = (
                        _overs < 6
                        and abs(signal.move) >= 0.05
                        and spread <= 0.02
                        and signal.direction == "SHORT"
                    )
                    if not _early_shock:
                        log.info(
                            "CRICKET_SPIKE_REJECT | %s | reason=score_filter_low_pressure "
                            "| overs=%.1f | wicket=%s | move=%.4f",
                            match_id, _overs, _recent_wicket, signal.move,
                        )
                        self._skip("score_filter")
                        signal = None
                    else:
                        log.info(
                            "CRICKET_EARLY_SHOCK_ACCEPTED | %s "
                            "| overs=%.1f | move=%.4f | spread=%.4f",
                            match_id, _overs, signal.move, spread,
                        )

            # Quality gate 2: Spread
            if signal and spread > SPIKE_MAX_SPREAD:
                log.info(
                    "CRICKET_SPIKE_REJECT | %s | reason=spread "
                    "| spread=%.4f > %.4f | move=%.4f",
                    match_id, spread, SPIKE_MAX_SPREAD, signal.move,
                )
                self._skip("spread")
                signal = None

            # Quality gate 3: Regime filter — require momentum in last 60s
            if signal:
                regime_ok = False
                for i in range(len(buf) - 1, -1, -1):
                    dt = timestamp - buf[i].timestamp
                    if dt > REGIME_LOOKBACK_S:
                        break
                    if abs(mid - buf[i].mid) >= REGIME_MIN_MOVE:
                        regime_ok = True
                        break
                if not regime_ok:
                    log.info(
                        "CRICKET_SPIKE_REJECT | %s | reason=low_momentum "
                        "| move=%.4f | no %.3f move in last %.0fs",
                        match_id, signal.move, REGIME_MIN_MOVE,
                        REGIME_LOOKBACK_S,
                    )
                    self._skip("low_momentum")
                    signal = None

            # ── v8.7: Observe-only mode — log but do NOT act ──────
            if observe_only and signal:
                log.info(
                    "CRICKET_BLOCK_NO_STATE | %s | dir=%s | move=%.4f "
                    "| mid=%.4f | reason=observe_only_no_match_state",
                    match_id, signal.direction, signal.move, mid,
                )
                return None

            # ── v9.4: Signal passed all gates → store for continuation confirm ──
            if signal:
                log.info(
                    "CRICKET_SPIKE_ACCEPTED | %s | dir=%s | move=%.4f "
                    "| spread=%.4f | mid=%.4f",
                    match_id, signal.direction, signal.move,
                    spread, mid,
                )
                # Store spike and wait for NEXT tick to confirm continuation
                self._pending_confirm[match_id] = {
                    'signal': signal,
                    'peak': mid,
                    'dir': signal.direction,
                    'ts': timestamp,
                }
                self._last_signal_ts[match_id] = timestamp
                return None  # will confirm on next tick
        else:
            # No spike — check spread for diag counting only
            if spread > SPIKE_MAX_SPREAD:
                self._skip("spread")
            # Check regime for diag counting only
            else:
                regime_ok = False
                for i in range(len(buf) - 1, -1, -1):
                    dt = timestamp - buf[i].timestamp
                    if dt > REGIME_LOOKBACK_S:
                        break
                    if abs(mid - buf[i].mid) >= REGIME_MIN_MOVE:
                        regime_ok = True
                        break
                if not regime_ok:
                    self._skip("low_momentum")

        # Check drift (only when no spike and spread ok)
        if spread <= DRIFT_MAX_SPREAD:
            drift_signal = self._check_momentum_drift(
                match_id, buf, mid, spread, timestamp,
            )
            if drift_signal:
                return drift_signal

        # Periodic diagnostic (every 60s)
        if timestamp - self._last_skip_log > 60.0:
            total_ticks = sum(len(b) for b in self._ticks.values())
            log.info(
                "CRICKET_TICK_DIAG | markets=%d | total_ticks=%d | "
                "skips: not_ipl=%d small_spike=%d no_contraction=%d "
                "low_edge=%d spread=%d band=%d retrace=%d cooldown=%d "
                "low_momentum=%d match_cd=%d pullback_exp=%d cont_fail=%d",
                len(self._ticks), total_ticks,
                self._skip_counts.get("not_ipl", 0),
                self._skip_counts.get("small_spike", 0),
                self._skip_counts.get("no_contraction", 0),
                self._skip_counts.get("low_edge", 0),
                self._skip_counts.get("spread", 0),
                self._skip_counts.get("band", 0),
                self._skip_counts.get("retrace", 0),
                self._skip_counts.get("cooldown", 0),
                self._skip_counts.get("low_momentum", 0),
                self._skip_counts.get("match_cooldown", 0),
                self._skip_counts.get("pullback_expired", 0),
                self._skip_counts.get("continuation_fail", 0),
            )
            self._skip_counts = {k: 0 for k in self._skip_counts}
            self._last_skip_log = timestamp

        return None

    # ── Pattern: Spike Continuation ───────────────────────────────

    def _check_spike_continuation(
        self, match_id: str, buf: deque, mid: float,
        spread: float, now: float,
    ) -> Optional[CricketTickSignal]:
        """Detect spike ≥0.02 within ≤120s and ride continuation.

        v8.5: Uses both buffer lookback AND stateful mid tracking
        to handle sparse REST polling data.
        """

        # ── Method 1: Buffer lookback (works with dense WS ticks) ──
        best_move = 0.0
        best_origin_idx = -1

        for i in range(len(buf) - 2, -1, -1):
            dt = now - buf[i].timestamp
            if dt > SPIKE_WINDOW_S:
                break
            move = mid - buf[i].mid
            if abs(move) > abs(best_move):
                best_move = move
                best_origin_idx = i

        # ── Method 2: Stateful mid delta (v8.5 — handles sparse REST) ──
        prev_mid = self._last_mid.get(match_id)
        prev_ts = self._last_mid_ts.get(match_id, 0)
        if prev_mid is not None and abs(mid - prev_mid) >= 0.001:
            stateful_dt = now - prev_ts
            stateful_move = mid - prev_mid
            if stateful_dt <= SPIKE_WINDOW_S and abs(stateful_move) > abs(best_move):
                best_move = stateful_move
                # Use the closest buffer tick at the old price as origin
                for i in range(len(buf) - 2, -1, -1):
                    if abs(buf[i].mid - prev_mid) < 0.001:
                        best_origin_idx = i
                        break

        # Minimum spike size
        if abs(best_move) < SPIKE_MIN_MOVE:
            return None

        # Need a valid origin
        if best_origin_idx < 0:
            return None

        origin = buf[best_origin_idx]
        spike_age = now - origin.timestamp

        # Confirmation delay: must be 5-10s after spike origin
        # v8.5: Skip confirmation for REST-detected spikes (age > 30s)
        if spike_age < SPIKE_CONFIRM_MIN_S and spike_age < 30.0:
            self._skip("too_early")
            return None

        # Check retrace: price must not have reverted > 33% of spike
        peak_idx = best_origin_idx
        if best_move > 0:
            for j in range(best_origin_idx, len(buf)):
                if buf[j].mid > buf[peak_idx].mid:
                    peak_idx = j
        else:
            for j in range(best_origin_idx, len(buf)):
                if buf[j].mid < buf[peak_idx].mid:
                    peak_idx = j
        peak_price = buf[peak_idx].mid

        retrace = 0.0
        for j in range(peak_idx, len(buf)):
            if best_move > 0:
                retrace = max(retrace, peak_price - buf[j].mid)
            else:
                retrace = max(retrace, buf[j].mid - peak_price)

        if retrace > abs(best_move) * SPIKE_MAX_RETRACE:
            self._skip("retrace")
            return None

        # Spread contraction check
        if len(buf) >= SPIKE_MIN_CONTRACTION + 1:
            contracting = 0
            for k in range(len(buf) - SPIKE_MIN_CONTRACTION, len(buf)):
                if buf[k].spread <= buf[k - 1].spread:
                    contracting += 1
            if contracting < SPIKE_MIN_CONTRACTION:
                self._skip("no_contraction")
                return None
        else:
            self._skip("no_contraction")
            return None

        # ── SIGNAL CONFIRMED ──────────────────────────────────────
        direction = "LONG" if best_move > 0 else "SHORT"

        # Edge calculation (v2.0)
        projected_move = max(0.02, min(0.05, abs(best_move) * 0.6))
        if direction == "LONG":
            fair = mid + projected_move
        else:
            fair = mid - projected_move
        edge = projected_move - spread / 2.0

        if edge < MIN_EDGE:
            self._skip("low_edge")
            log.info(
                "CRICKET_SKIP_REASON | reason=low_edge | "
                "projected=%.4f spread=%.4f edge=%.4f < %.4f | %s",
                projected_move, spread, edge, MIN_EDGE, match_id,
            )
            return None

        self._last_signal_ts[match_id] = now
        log.info(
            "CRICKET_SIGNAL_CONTINUATION | type=SPIKE_CONTINUATION | "
            "spike_size=%.4f | delay_s=%.1f | retrace_pct=%.0f%% | "
            "spread=%.4f | edge=%.4f | projected=%.4f | "
            "dir=%s | mid=%.4f | %s",
            abs(best_move), spike_age,
            (retrace / abs(best_move) * 100) if abs(best_move) > 0 else 0,
            spread, edge, projected_move,
            direction, mid, match_id,
        )
        return CricketTickSignal(
            signal_type="SPIKE_CONTINUATION",
            match_id=match_id,
            direction=direction,
            move=abs(best_move),
            entry_price=mid,
            fair_price=fair,
            edge=edge,
            spread=spread,
        )

    # ── Pattern: Momentum Drift ───────────────────────────────────

    def _check_momentum_drift(
        self, match_id: str, buf: deque, mid: float,
        spread: float, now: float,
    ) -> Optional[CricketTickSignal]:
        """Detect sustained directional drift ≥0.04 over 120-300s."""
        if spread > DRIFT_MAX_SPREAD:
            return None

        # Find tick from 120-300s ago
        old_tick = None
        for i in range(len(buf)):
            dt = now - buf[i].timestamp
            if DRIFT_WINDOW_MIN_S <= dt <= DRIFT_WINDOW_MAX_S:
                old_tick = buf[i]
                break

        if old_tick is None:
            return None

        net_move = mid - old_tick.mid
        if abs(net_move) < DRIFT_MIN_MOVE:
            return None

        # Smoothness check
        prices_in_window = [
            t.mid for t in buf
            if now - t.timestamp <= DRIFT_WINDOW_MAX_S
        ]
        if not prices_in_window:
            return None
        total_range = max(prices_in_window) - min(prices_in_window)
        if total_range <= 0:
            return None
        smoothness = abs(net_move) / total_range
        if smoothness < DRIFT_SMOOTH_RATIO:
            return None

        # Confirmation: last 3 ticks same direction
        if len(buf) >= DRIFT_CONFIRM_TICKS + 1:
            all_same = True
            for k in range(len(buf) - DRIFT_CONFIRM_TICKS, len(buf)):
                delta = buf[k].mid - buf[k - 1].mid
                if net_move > 0 and delta < 0:
                    all_same = False
                    break
                if net_move < 0 and delta > 0:
                    all_same = False
                    break
            if not all_same:
                return None
        else:
            return None

        # ── SIGNAL CONFIRMED ──────────────────────────────────────
        direction = "LONG" if net_move > 0 else "SHORT"
        projected_move = 0.02  # fixed for drift
        if direction == "LONG":
            fair = mid + projected_move
        else:
            fair = mid - projected_move
        edge = projected_move - spread / 2.0

        if edge < MIN_EDGE:
            self._skip("low_edge")
            return None

        self._last_signal_ts[match_id] = now
        log.info(
            "CRICKET_SIGNAL_CONTINUATION | type=MOMENTUM_DRIFT | "
            "net_move=%.4f | smooth=%.2f | spread=%.4f | "
            "edge=%.4f | dir=%s | mid=%.4f | %s",
            net_move, smoothness, spread, edge, direction, mid, match_id,
        )
        return CricketTickSignal(
            signal_type="MOMENTUM_DRIFT",
            match_id=match_id,
            direction=direction,
            move=abs(net_move),
            entry_price=mid,
            fair_price=fair,
            edge=edge,
            spread=spread,
        )

    # ── Trade Registration ────────────────────────────────────────

    def register_trade(self, signal: CricketTickSignal) -> None:
        """Register an active trade for exit tracking."""
        # ═══ GLOBAL EDGE GUARD (v7.0 — inside function, non-bypassable) ═══
        from sports.guards import validate_trade_execution, circuit_breaker, STRAT_CRICKET_MOM
        can_exec, block_reason = validate_trade_execution(
            edge=signal.edge,
            price=signal.entry_price,
            sport="cricket_tick",
            context=f"{signal.signal_type} {signal.direction} | {signal.match_id}",
            strategy=STRAT_CRICKET_MOM,
        )
        if not can_exec:
            circuit_breaker.record_signal_result(
                was_blocked=True, sport="cricket_tick", strategy=STRAT_CRICKET_MOM,
            )
            return
        circuit_breaker.record_signal_result(
            was_blocked=False, sport="cricket_tick", strategy=STRAT_CRICKET_MOM,
        )

        now = time.time()
        if signal.direction == "LONG":
            stop = signal.entry_price - STOP_LOSS
        else:
            stop = signal.entry_price + STOP_LOSS

        self._trades[signal.match_id] = TickTrade(
            match_id=signal.match_id,
            signal_type=signal.signal_type,
            entry_price=signal.entry_price,
            entry_timestamp=now,
            direction=signal.direction,
            stop=stop,
            timeout_s=TIMEOUT_S,
            mfe=0.0,
            mfe_timestamp=now,
            last_extreme_ts=now,
        )
        log.info(
            "CRICKET_TICK_TRADE | %s | %s | entry=%.4f | "
            "stop=%.4f | timeout=%ds | %s",
            signal.signal_type, signal.direction,
            signal.entry_price, stop,
            int(TIMEOUT_S), signal.match_id,
        )

    # ── Exit Checks ───────────────────────────────────────────────

    def check_exits(
        self, get_price: callable,
    ) -> list[tuple[str, str, float, float, str]]:
        """Check all active trades for exits.

        Returns:
            List of (match_id, signal_type, entry_price, exit_price, reason)
        """
        exits = []
        now = time.time()

        for match_id in list(self._trades.keys()):
            trade = self._trades[match_id]
            result = get_price(match_id)
            if result is None:
                continue
            mid, spread = result
            if mid <= 0:
                continue

            trade.tick_count += 1
            elapsed = now - trade.entry_timestamp
            reason = None

            # ── Update MFE and runner state ───────────────────────
            if trade.direction == "LONG":
                favorable = mid - trade.entry_price
            else:
                favorable = trade.entry_price - mid

            if favorable > trade.mfe:
                trade.mfe = favorable
                trade.mfe_timestamp = now
                trade.last_extreme_ts = now

                # Activate runner once MFE threshold reached
                if not trade.runner_active and trade.mfe >= RUNNER_ACTIVATION:
                    trade.runner_active = True
                    log.info(
                        "CRICKET_RUNNER_ACTIVE | mfe=%.4f | "
                        "entry=%.4f mid=%.4f | %s",
                        trade.mfe, trade.entry_price, mid, match_id,
                    )

                # Update trailing stop
                if trade.runner_active:
                    if trade.direction == "LONG":
                        trade.trail_price = mid - RUNNER_TRAIL
                    else:
                        trade.trail_price = mid + RUNNER_TRAIL

            # ── Exit hierarchy: STOP → RUNNER → MOMENTUM_FAIL → TIMEOUT ──

            # 1. Stop loss
            if trade.direction == "LONG" and mid <= trade.stop:
                reason = "EXIT_STOP_LOSS"
            elif trade.direction == "SHORT" and mid >= trade.stop:
                reason = "EXIT_STOP_LOSS"

            # 2. Runner trailing stop (only when active)
            if not reason and trade.runner_active:
                if trade.direction == "LONG" and mid <= trade.trail_price:
                    reason = "EXIT_RUNNER"
                elif trade.direction == "SHORT" and mid >= trade.trail_price:
                    reason = "EXIT_RUNNER"

            # 3. v8.2: Continuation clock — exit if no +0.02 in 90s
            if not reason and not trade.runner_active:
                if elapsed >= CONTINUATION_S and trade.mfe < CONTINUATION_MIN_MOVE:
                    reason = "EXIT_EARLY"
                    log.info(
                        "CRICKET_EXIT_EARLY | reason=NO_CONTINUATION | "
                        "duration=%.0fs | mfe=%.4f < %.4f | "
                        "entry=%.4f mid=%.4f | %s",
                        elapsed, trade.mfe, CONTINUATION_MIN_MOVE,
                        trade.entry_price, mid, match_id,
                    )

            # 4. Momentum fail: no new high/low for 45s after entry
            if not reason and not trade.runner_active:
                time_since_extreme = now - trade.last_extreme_ts
                if time_since_extreme >= MOMENTUM_FAIL_S and elapsed >= MOMENTUM_FAIL_S:
                    reason = "EXIT_MOMENTUM_FAIL"
                    log.warning(
                        "CRICKET_EXIT_MOMENTUM_FAIL | duration=%.0fs | "
                        "mfe=%.4f | entry=%.4f mid=%.4f | "
                        "time_since_extreme=%.0fs | %s",
                        elapsed, trade.mfe, trade.entry_price, mid,
                        time_since_extreme, match_id,
                    )

            # 5. Timeout
            if not reason and elapsed >= trade.timeout_s:
                reason = "EXIT_TIMEOUT"

            # ── Execute exit ──────────────────────────────────────
            if reason:
                if trade.direction == "LONG":
                    pnl = mid - trade.entry_price
                else:
                    pnl = trade.entry_price - mid

                r_mult = pnl / STOP_LOSS if STOP_LOSS > 0 else 0

                log.info(
                    "CRICKET_TICK_EXIT | %s | %s | entry=%.4f "
                    "exit=%.4f | pnl=%+.4f R=%+.3f | mfe=%.4f | "
                    "runner=%s | hold=%.0fs | %s",
                    trade.signal_type, reason,
                    trade.entry_price, mid, pnl, r_mult,
                    trade.mfe, trade.runner_active,
                    elapsed, match_id,
                )

                # Feed circuit breaker
                from sports.guards import circuit_breaker, STRAT_CRICKET_MOM
                circuit_breaker.record_trade_outcome(
                    r_mult,
                    sport="cricket_tick",
                    strategy=STRAT_CRICKET_MOM,
                )

                exits.append((
                    match_id, trade.signal_type,
                    trade.entry_price, mid, reason,
                ))
                del self._trades[match_id]
                # v8.2: record exit time for match cooldown
                self._last_trade_exit_ts[match_id] = now

        return exits

    # ── Internals ─────────────────────────────────────────────────

    def _skip(self, reason: str) -> None:
        self._skip_counts[reason] = self._skip_counts.get(reason, 0) + 1

    @property
    def active_trade_count(self) -> int:
        return len(self._trades)

    @property
    def has_trades(self) -> bool:
        return bool(self._trades)
