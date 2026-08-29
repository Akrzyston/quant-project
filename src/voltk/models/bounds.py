"""No-arbitrage bounds.

Checked before solving for implied vol, because a price outside these has no
solution and a solver asked for one will either diverge or return a fitted
number that means nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from voltk.models.base import CP, discount


class ArbitrageError(ValueError):
    """A quoted price admits no implied vol because it violates a bound."""


@dataclass(frozen=True, slots=True)
class Bounds:
    lower: float
    upper: float

    def contains(self, price: float, tolerance: float = 0.0) -> bool:
        return self.lower - tolerance <= price <= self.upper + tolerance

    def check(self, price: float, label: str, tolerance: float = 1e-12) -> None:
        if not self.contains(price, tolerance):
            raise ArbitrageError(
                f"{label} price {price:.10g} is outside [{self.lower:.10g}, "
                f"{self.upper:.10g}]. No implied vol exists."
            )


def lognormal_bounds(forward: float, strike: float, tau: float, rate: float, cp: CP) -> Bounds:
    """Discounted intrinsic below; forward (call) or strike (put) above."""
    df = discount(rate, tau)
    lower = df * max(cp.sign * (forward - strike), 0.0)
    upper = df * (forward if cp is CP.CALL else strike)
    return Bounds(lower=lower, upper=upper)


def spot_bounds(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    cp: CP,
    carry_yield: float = 0.0,
) -> Bounds:
    """Spot parameterisation: both legs carry their own discount factor."""
    carry_df = math.exp(-carry_yield * tau)
    rate_df = discount(rate, tau)
    lower = max(cp.sign * (spot * carry_df - strike * rate_df), 0.0)
    upper = spot * carry_df if cp is CP.CALL else strike * rate_df
    return Bounds(lower=lower, upper=upper)


def normal_bounds(forward: float, strike: float, tau: float, rate: float, cp: CP) -> Bounds:
    """Bachelier has no finite upper bound: normal vol admits unbounded prices."""
    df = discount(rate, tau)
    return Bounds(lower=df * max(cp.sign * (forward - strike), 0.0), upper=math.inf)


def inverse_bounds(forward: float, strike: float, tau: float, rate: float, cp: CP) -> Bounds:
    """Settlement-currency bounds.

    A coin-settled call can never be worth more than one coin however far the
    underlying rallies, which is the bound that makes the payoff concave.
    """
    if cp is CP.CALL:
        return Bounds(lower=max(1.0 - strike / forward, 0.0), upper=1.0)
    return Bounds(lower=max(strike / forward - 1.0, 0.0), upper=strike / forward)
