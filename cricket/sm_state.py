"""
Cricket Scoreboard-Delta State Machine.

Derives synthetic events from consecutive Sportmonks polls:
    WICKET   — wickets_now > wickets_prev
    BOUNDARY — runs_delta >= 4
    SURGE    — runs_delta >= 6
    DOT      — runs_delta == 0

Regime classification:
    CHAOS      — overs < 6 OR > 15 OR wickets >= 5 OR pressure > 2.2
    STRUCTURED — everything else

Tail Collapse Override:
    WICKET + wickets >= 6 → force CHAOS + SHORT

v1.0.1 — 2026-04-25
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from cricket.sm_feeds import ScoreboardSnapshot

log = logging.getLogger("cricket.sm_state")


# ═════════════════════════════════════════════════════════════════════
#  Events + Regime
# ═════════════════════════════════════════════════════════════════════

class CricketEvent(Enum):
    WICKET   = "WICKET"
    BOUNDARY = "BOUNDARY"
    SURGE    = "SURGE"
    DOT      = "DOT"
    NONE     = "NONE"


class CricketRegime(Enum):
    CHAOS      = "CHAOS"
    STRUCTURED = "STRUCTURED"


@dataclass
class EventResult:
    """Output of event derivation for a single poll delta."""
    event: CricketEvent
    regime: CricketRegime
    fixture_id: int
    innings: int

    # Delta values
    runs_delta: int = 0
    wickets_delta: int = 0
    overs_delta: float = 0.0

    # Current state
    runs: int = 0
    wickets: int = 0
    overs: float = 0.0
    run_rate: float = 0.0
    required_run_rate: float = 0.0
    pressure: float = 0.0
    wickets_in_hand: int = 10
    target: int = 0

    # Tail collapse
    tail_collapse: bool = False
    forced_direction: str = ""  # "SHORT" if tail collapse

    # Timestamps
    timestamp: float = 0.0


# ═════════════════════════════════════════════════════════════════════
#  Debounce Config
# ═════════════════════════════════════════════════════════════════════

DEBOUNCE_WINDOW_S = 8.0  # minimum seconds between events per fixture


# ═════════════════════════════════════════════════════════════════════
#  State Machine
# ═════════════════════════════════════════════════════════════════════

class SmCricketState:
    """Maintains per-fixture state and derives events from scoreboard deltas.

    Stores the previous snapshot for each fixture. On each new poll,
    computes deltas and classifies events + regime.
    """

    def __init__(self):
        self._prev: dict[int, ScoreboardSnapshot] = {}  # fixture_id → last snapshot
        self._last_event_ts: dict[int, float] = {}       # fixture_id → last event ts
        self._event_counts: dict[int, dict[str, int]] = {}  # fixture_id → {event: count}

    def update(self, snapshot: ScoreboardSnapshot) -> EventResult | None:
        """Process a new scoreboard snapshot and derive events.

        Returns EventResult if a meaningful event occurred, None otherwise.
        Implements debouncing (8s minimum between events per fixture).
        """
        fid = snapshot.fixture_id
        now = snapshot.timestamp

        prev = self._prev.get(fid)
        self._prev[fid] = snapshot

        # First poll — no delta possible
        if prev is None:
            log.info(
                "CRICKET_SM_STATE_INIT | fixture=%d | inn=%d | "
                "%d/%d (%.1f ov)",
                fid, snapshot.innings, snapshot.runs,
                snapshot.wickets, snapshot.overs,
            )
            return None

        # Same innings check
        if snapshot.innings != prev.innings and snapshot.innings < prev.innings:
            return None  # shouldn't happen, but guard

        # Innings change (1→2) — reset state for new innings
        if snapshot.innings != prev.innings:
            log.info(
                "CRICKET_SM_INNINGS_CHANGE | fixture=%d | %d → %d",
                fid, prev.innings, snapshot.innings,
            )
            # Don't generate event on innings change, just store
            return None

        # ── Compute deltas ────────────────────────────────────────
        runs_delta = snapshot.runs - prev.runs
        wickets_delta = snapshot.wickets - prev.wickets
        overs_delta = snapshot.overs - prev.overs

        # No change at all — skip
        if runs_delta == 0 and wickets_delta == 0 and overs_delta == 0:
            return None

        # ── Debounce check ────────────────────────────────────────
        last_evt = self._last_event_ts.get(fid, 0)
        if now - last_evt < DEBOUNCE_WINDOW_S:
            log.debug(
                "CRICKET_SM_DEBOUNCE | fixture=%d | age=%.1fs < %.1fs",
                fid, now - last_evt, DEBOUNCE_WINDOW_S,
            )
            return None

        # ── Event classification ──────────────────────────────────
        event = self._classify_event(runs_delta, wickets_delta)

        if event == CricketEvent.NONE:
            return None

        self._last_event_ts[fid] = now

        # ── Pressure + Regime ─────────────────────────────────────
        pressure = self._compute_pressure(snapshot)
        regime = self._classify_regime(snapshot, pressure)

        # ── Tail Collapse Override ────────────────────────────────
        tail_collapse = False
        forced_direction = ""
        if event == CricketEvent.WICKET and snapshot.wickets >= 6:
            regime = CricketRegime.CHAOS
            tail_collapse = True
            forced_direction = "SHORT"
            log.info(
                "CRICKET_TAIL_COLLAPSE | fixture=%d | wickets=%d | "
                "forced=CHAOS+SHORT",
                fid, snapshot.wickets,
            )

        # ── Track event counts ────────────────────────────────────
        if fid not in self._event_counts:
            self._event_counts[fid] = {}
        self._event_counts[fid][event.value] = (
            self._event_counts[fid].get(event.value, 0) + 1
        )

        result = EventResult(
            event=event,
            regime=regime,
            fixture_id=fid,
            innings=snapshot.innings,
            runs_delta=runs_delta,
            wickets_delta=wickets_delta,
            overs_delta=overs_delta,
            runs=snapshot.runs,
            wickets=snapshot.wickets,
            overs=snapshot.overs,
            run_rate=snapshot.run_rate,
            required_run_rate=snapshot.required_run_rate,
            pressure=pressure,
            wickets_in_hand=snapshot.wickets_in_hand,
            target=snapshot.target,
            tail_collapse=tail_collapse,
            forced_direction=forced_direction,
            timestamp=now,
        )

        # ── Logging ───────────────────────────────────────────────
        log.info(
            "CRICKET_EVENT_DETECTED | fixture=%d | event=%s | "
            "runs_delta=%d wickets_delta=%d | %d/%d (%.1f ov) | "
            "RR=%.1f RRR=%.1f",
            fid, event.value, runs_delta, wickets_delta,
            snapshot.runs, snapshot.wickets, snapshot.overs,
            snapshot.run_rate, snapshot.required_run_rate,
        )
        log.info(
            "CRICKET_REGIME | fixture=%d | regime=%s | "
            "pressure=%.2f | tail_collapse=%s",
            fid, regime.value, pressure, tail_collapse,
        )
        log.info(
            "CRICKET_PRESSURE | fixture=%d | CRR=%.1f | RRR=%.1f | "
            "WIH=%d | pressure=%.2f",
            fid, snapshot.run_rate, snapshot.required_run_rate,
            snapshot.wickets_in_hand, pressure,
        )

        return result

    def _classify_event(
        self, runs_delta: int, wickets_delta: int
    ) -> CricketEvent:
        """Classify event from scoreboard deltas.

        Priority order:
            1. WICKET (highest priority)
            2. SURGE (runs_delta >= 6)
            3. BOUNDARY (runs_delta >= 4)
            4. DOT (runs_delta == 0)
        """
        if wickets_delta > 0:
            return CricketEvent.WICKET

        if runs_delta >= 6:
            return CricketEvent.SURGE

        if runs_delta >= 4:
            return CricketEvent.BOUNDARY

        if runs_delta == 0:
            return CricketEvent.DOT

        # Small run change (1-3) — not a tradeable event
        return CricketEvent.NONE

    def _compute_pressure(self, snap: ScoreboardSnapshot) -> float:
        """Compute chase pressure index.

        Pressure = (RRR / CRR) * (10 / (WIH + 1))

        Higher = more pressure on batting team.
        Only meaningful in 2nd innings.
        """
        if snap.innings != 2:
            return 0.0

        crr = snap.run_rate
        rrr = snap.required_run_rate
        wih = snap.wickets_in_hand

        if crr <= 0:
            return 10.0  # max pressure if no scoring

        return (rrr / crr) * (10 / (wih + 1))

    def _classify_regime(
        self, snap: ScoreboardSnapshot, pressure: float
    ) -> CricketRegime:
        """Classify match regime.

        CHAOS:
            overs < 6 (powerplay)
            OR overs > 15 (death)
            OR wickets >= 5 (collapse)
            OR pressure > 2.2

        STRUCTURED:
            everything else
        """
        if snap.overs < 6:
            return CricketRegime.CHAOS
        if snap.overs > 15:
            return CricketRegime.CHAOS
        if snap.wickets >= 5:
            return CricketRegime.CHAOS
        if pressure > 2.2:
            return CricketRegime.CHAOS

        return CricketRegime.STRUCTURED

    def get_event_counts(self, fixture_id: int) -> dict[str, int]:
        """Return event counts for a fixture (for diagnostics)."""
        return self._event_counts.get(fixture_id, {})
