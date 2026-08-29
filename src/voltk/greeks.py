"""Risk container.

Quote and base units are carried side by side because on an inverse book the
same delta is a different number in each, and conflating them is the usual way
a crypto options position ends up wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Unit(StrEnum):
    QUOTE = "quote"
    BASE = "base"


@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float
    unit: Unit
    vega_bump: float = 0.01
    theta_period: float = 1.0 / 365.0
