"""Implied vol by safeguarded Newton.

Newton leaves the bracket on low-vega wings; bisection is safe but slow. Each
iteration takes the Newton step when it stays inside the bracket and bisects
otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from voltk.models.base import CP

MAX_ITERATIONS = 100
PRICE_TOLERANCE = 1e-14
VOL_TOLERANCE = 1e-14
MAX_VOL = 50.0

# Below this vega the quote is indistinguishable from intrinsic, so this is
# reported (not identified) rather than raised.
VEGA_FLOOR = 1e-8


class SolverError(RuntimeError):
    """The solver could not bracket or converge."""


def solve_implied_vol(
    price_fn: Callable[[float], float],
    target: float,
    *,
    vega_fn: Callable[[float], float] | None = None,
    low: float = 1e-12,
    high: float = 5.0,
    max_vol: float = MAX_VOL,
    price_tolerance: float = PRICE_TOLERANCE,
) -> float:
    """Invert a monotonically increasing price-in-vol function."""
    f_low = price_fn(low) - target
    if abs(f_low) <= price_tolerance:
        return low

    # Expand upward until the target is bracketed. Price is increasing in vol,
    # so a single-sided expansion is enough.
    f_high = price_fn(high) - target
    while f_high < 0.0:
        high *= 2.0
        if high > max_vol:
            raise SolverError(
                f"Price {target:.10g} is not reachable below a vol of {max_vol:g}."
            )
        f_high = price_fn(high) - target

    if f_low > 0.0:
        raise SolverError(
            f"Price {target:.10g} sits below the zero-vol value; no solution exists."
        )

    vol = 0.5 * (low + high)
    for _ in range(MAX_ITERATIONS):
        residual = price_fn(vol) - target
        if abs(residual) <= price_tolerance:
            return vol

        if residual > 0.0:
            high = vol
        else:
            low = vol

        stepped = False
        if vega_fn is not None:
            slope = vega_fn(vol)
            if slope > 1e-14:
                candidate = vol - residual / slope
                if low < candidate < high and math.isfinite(candidate):
                    vol, stepped = candidate, True
        if not stepped:
            vol = 0.5 * (low + high)

        if high - low < VOL_TOLERANCE:
            return vol

    raise SolverError(
        f"No convergence after {MAX_ITERATIONS} iterations; bracket is [{low:g}, {high:g}]."
    )


def implied_vol_for(
    model,
    price: float,
    forward: float,
    strike: float,
    tau: float,
    rate: float,
    cp: CP,
) -> float:
    """Solve using the model's own price and vega."""

    def price_fn(vol: float) -> float:
        return model.price(forward, strike, tau, vol, rate, cp)

    def vega_fn(vol: float) -> float:
        return model.greeks(forward, strike, tau, vol, rate, cp).vega

    low, high, ceiling = model.vol_bracket(forward, strike)
    return solve_implied_vol(
        price_fn, price, vega_fn=vega_fn, low=low, high=high, max_vol=ceiling
    )


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    vol: float
    price_residual: float
    vega: float

    @property
    def identified(self) -> bool:
        """False when vega is too small for the vol to be recoverable."""
        return abs(self.vega) >= VEGA_FLOOR

    def describe(self) -> str:
        if self.identified:
            return f"vol {self.vol:.4%}, residual {self.price_residual:.2e}"
        return (
            f"vol {self.vol:.4%} not identified: vega {self.vega:.2e} is below the "
            "floor, so the quote is indistinguishable from intrinsic"
        )


def implied_vol_detailed(
    model,
    price: float,
    forward: float,
    strike: float,
    tau: float,
    rate: float,
    cp: CP,
) -> ImpliedVolResult:
    """Solve and report how well determined the answer is."""
    vol = implied_vol_for(model, price, forward, strike, tau, rate, cp)
    return ImpliedVolResult(
        vol=vol,
        price_residual=model.price(forward, strike, tau, vol, rate, cp) - price,
        vega=model.greeks(forward, strike, tau, vol, rate, cp).vega,
    )
