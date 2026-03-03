"""
Tennis player name matching.

Handles the variation in how player names appear across different sources:
    Polymarket: "Novak Djokovic vs Jannik Sinner"
    Flashscore: "N. Djokovic"
    ESPN:       "Djokovic, N."
    Outcomes:   "Djokovic" or "Novak Djokovic"

Strategy:
    1. Extract surname (last word) — primary match key.
    2. Normalize accents, case, hyphens.
    3. Handle compound surnames (e.g. "De Minaur", "Van de Zandschulp").
    4. Fuzzy fallback for edge cases.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

log = logging.getLogger("tennis.matching")


# ═══════════════════════════════════════════════════════════════════════
#  Known compound surnames (surname → canonical form)
# ═══════════════════════════════════════════════════════════════════════

_COMPOUND_SURNAMES = {
    "de minaur": "de minaur",
    "van de zandschulp": "van de zandschulp",
    "auger-aliassime": "auger-aliassime",
    "auger aliassime": "auger-aliassime",
    "garcia-perez": "garcia-perez",
    "bautista agut": "bautista agut",
    "bautista-agut": "bautista agut",
    "carreno busta": "carreno busta",
    "carreno-busta": "carreno busta",
    "martinez": "martinez",
    "muller": "muller",
    "ramos-vinolas": "ramos-vinolas",
    "di lorenzo": "di lorenzo",
}

# Player name aliases: common variations → canonical
_PLAYER_ALIASES = {
    "nole": "djokovic",
    "rafa": "nadal",
    "fedex": "federer",
    "coco": "gauff",
}


# ═══════════════════════════════════════════════════════════════════════
#  Core Normalization
# ═══════════════════════════════════════════════════════════════════════

def normalize_tennis_name(name: str) -> str:
    """Normalize a tennis player name for comparison.

    Steps:
        1. Lowercase.
        2. Remove accents/diacritics (Ö→O, é→e, etc.).
        3. Strip leading/trailing whitespace.
        4. Remove common prefixes like "mr.", "ms.".
        5. Collapse multiple spaces.

    Returns:
        Normalized name string.
    """
    name = name.lower().strip()

    # Handle characters that don't decompose via NFD
    _CHAR_MAP = {"đ": "d", "ð": "d", "ø": "o", "ł": "l", "ß": "ss", "æ": "ae", "þ": "th"}
    for old, new in _CHAR_MAP.items():
        name = name.replace(old, new)

    # Remove accents
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )

    # Remove periods (for "N. Djokovic" → "N Djokovic")
    name = name.replace(".", " ")

    # Remove parenthetical content (e.g. "(1)" seed numbers)
    name = re.sub(r"\([^)]*\)", "", name)

    # Remove seed numbers like "[1]" or "#1"
    name = re.sub(r"\[?\d+\]?", "", name)
    name = re.sub(r"#\d+", "", name)

    # Collapse spaces
    name = " ".join(name.split())

    return name


def extract_surname(name: str) -> str:
    """Extract the surname (last meaningful word) from a player name.

    Handles:
        "Novak Djokovic"  → "djokovic"
        "N. Djokovic"     → "djokovic"
        "Djokovic, N."    → "djokovic"
        "A. De Minaur"    → "de minaur"
        "Carlos Alcaraz"  → "alcaraz"

    For compound surnames, returns the full compound.
    """
    norm = normalize_tennis_name(name)

    # Handle "Surname, FirstName" format
    if "," in norm:
        parts = [p.strip() for p in norm.split(",")]
        norm = parts[0]  # Surname is before the comma

    # Check for known compound surnames first
    for compound in _COMPOUND_SURNAMES:
        if compound in norm:
            return _COMPOUND_SURNAMES[compound]

    # Split into words, remove single-letter initials
    words = norm.split()
    meaningful = [w for w in words if len(w) > 1]

    if not meaningful:
        return norm

    # Return the last meaningful word as surname
    return meaningful[-1]


def extract_first_name(name: str) -> str:
    """Extract the first name or initial from a player name.

    Returns the first initial letter (useful for disambiguation).
    """
    norm = normalize_tennis_name(name)

    # Handle "Surname, FirstName" format
    if "," in norm:
        parts = [p.strip() for p in norm.split(",")]
        if len(parts) > 1:
            first = parts[1].strip()
            return first[0] if first else ""
        return ""

    words = norm.split()
    if words:
        return words[0][0] if words[0] else ""
    return ""


# ═══════════════════════════════════════════════════════════════════════
#  Matching Functions
# ═══════════════════════════════════════════════════════════════════════

def tennis_name_match_score(name_a: str, name_b: str) -> float:
    """Compute a match score between two player names (0-1).

    Strategy:
        1.0  — Exact normalized match.
        0.95 — Surname match + first initial match.
        0.90 — Surname exact match (different or missing first name).
        0.80 — Surname fuzzy match (≥0.85 ratio).
        0.70 — One name contained in the other.
        <0.70 — SequenceMatcher ratio on full normalized names.

    Args:
        name_a: First player name (any format).
        name_b: Second player name (any format).

    Returns:
        Match confidence (0.0 to 1.0).
    """
    # Normalize both
    na = normalize_tennis_name(name_a)
    nb = normalize_tennis_name(name_b)

    # Exact match after normalization
    if na == nb:
        return 1.0

    # Check aliases
    if na in _PLAYER_ALIASES:
        na = _PLAYER_ALIASES[na]
    if nb in _PLAYER_ALIASES:
        nb = _PLAYER_ALIASES[nb]
    if na == nb:
        return 1.0

    # Extract surnames
    sa = extract_surname(name_a)
    sb = extract_surname(name_b)

    # Surnames match exactly
    if sa == sb:
        # Check first initial for extra confidence
        fa = extract_first_name(name_a)
        fb = extract_first_name(name_b)
        if fa and fb and fa == fb:
            return 0.95  # Surname + initial match
        return 0.90  # Surname match only

    # Surname fuzzy match
    surname_ratio = SequenceMatcher(None, sa, sb).ratio()
    if surname_ratio >= 0.85:
        return 0.80

    # Containment check — one name contains the other
    if sa in nb or sb in na or na in nb or nb in na:
        return 0.70

    # Fallback: full name fuzzy match
    full_ratio = SequenceMatcher(None, na, nb).ratio()
    return full_ratio * 0.65  # Scale down since it's a weak signal


def match_tennis_players(
    poly_title: str,
    source_player_a: str,
    source_player_b: str,
    threshold: float = 0.80,
) -> tuple[bool, float, str, str]:
    """Match a Polymarket title to a pair of player names from a data source.

    Extracts player names from the Polymarket title (expects "Player A vs Player B"
    or "Player A vs. Player B"), then matches each against the source players.

    Args:
        poly_title:      Polymarket market title.
        source_player_a: Player A name from data source (e.g., Flashscore).
        source_player_b: Player B name from data source.
        threshold:       Minimum average match score to accept.

    Returns:
        (matched, avg_score, poly_player_a, poly_player_b):
            matched:       True if players match above threshold.
            avg_score:     Average match score across both players.
            poly_player_a: First player name extracted from Polymarket title.
            poly_player_b: Second player name extracted from Polymarket title.
    """
    # Extract player names from title
    poly_a, poly_b = extract_players_from_title(poly_title)
    if not poly_a or not poly_b:
        return False, 0.0, poly_a, poly_b

    # Try both orderings: Poly(A)↔Source(A) + Poly(B)↔Source(B)
    score_direct = (
        tennis_name_match_score(poly_a, source_player_a) +
        tennis_name_match_score(poly_b, source_player_b)
    ) / 2.0

    # And reversed: Poly(A)↔Source(B) + Poly(B)↔Source(A)
    score_reversed = (
        tennis_name_match_score(poly_a, source_player_b) +
        tennis_name_match_score(poly_b, source_player_a)
    ) / 2.0

    best = max(score_direct, score_reversed)

    if best >= threshold:
        log.info("TENNIS MATCH: '%s' ↔ '%s vs %s' (score=%.3f)",
                 poly_title, source_player_a, source_player_b, best)
    else:
        log.debug("TENNIS NO MATCH: '%s' ↔ '%s vs %s' (score=%.3f < %.2f)",
                  poly_title, source_player_a, source_player_b, best, threshold)

    return best >= threshold, best, poly_a, poly_b


def extract_players_from_title(title: str) -> tuple[str, str]:
    """Extract two player names from a match title.

    Handles:
        "Djokovic vs Sinner"
        "Djokovic vs. Sinner"
        "N. Djokovic v J. Sinner"
        "Novak Djokovic vs Jannik Sinner"
        "Djokovic vs Sinner (ATP Australian Open)"

    Returns:
        (player_a, player_b) — both stripped of whitespace.
        Returns ("", "") if extraction fails.
    """
    t = title.strip()

    # Remove tournament/event info in parentheses
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t)

    # Remove trailing date patterns
    t = re.sub(r"\s*-?\s*\d{4}-\d{2}-\d{2}$", "", t)

    # Remove "More Markets" suffix
    t = re.sub(r"\s*-?\s*More Markets\s*$", "", t, flags=re.IGNORECASE)

    # Split on "vs.", "vs", "v."
    parts = re.split(r"\s+vs\.?\s+|\s+v\s+|\s+v\.\s+", t, maxsplit=1,
                      flags=re.IGNORECASE)

    if len(parts) == 2:
        a = parts[0].strip()
        b = parts[1].strip()
        if a and b:
            return a, b

    return "", ""


# ═══════════════════════════════════════════════════════════════════════
#  Match Polymarket Tennis Markets to Outcome Labels
# ═══════════════════════════════════════════════════════════════════════

def identify_favorite_from_outcomes(
    outcomes: list,  # list of MarketOutcome
    poly_player_a: str,
    poly_player_b: str,
) -> tuple[str, str, float, float]:
    """Identify which outcome corresponds to which player and their prices.

    Returns:
        (player_a_token, player_b_token, price_a, price_b)
    """
    if len(outcomes) < 2:
        return "", "", 0.5, 0.5

    o1 = outcomes[0]
    o2 = outcomes[1]

    # Match outcomes to players
    score_1a = tennis_name_match_score(o1.outcome_label, poly_player_a)
    score_1b = tennis_name_match_score(o1.outcome_label, poly_player_b)

    if score_1a >= score_1b:
        # Outcome 1 → Player A, Outcome 2 → Player B
        return (o1.token_id, o2.token_id,
                o1.last_price, o2.last_price)
    else:
        # Outcome 1 → Player B, Outcome 2 → Player A
        return (o2.token_id, o1.token_id,
                o2.last_price, o1.last_price)
