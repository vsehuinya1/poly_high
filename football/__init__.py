"""
Football in-play probability engine.

Poisson-based model that derives team-specific goal intensities
from pre-match 1X2 probabilities. No external APIs required.
"""
from football.state import FootballState, Probabilities
from football.model import FootballInPlayModel, shock_factor

__all__ = ["FootballState", "Probabilities", "FootballInPlayModel", "shock_factor"]
