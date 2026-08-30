"""Two-sided quote construction: theoretical value from the surface, width
derived from stated risk inputs rather than hand-set.

Width has three named terms, each a real risk a market maker is actually
compensated for: fit uncertainty (vega times how far the raw market mark
sits from the fitted curve at this strike), rehedge risk (gamma times the
expected spot move over one requoting interval, the standard second-order
cost of not being able to hedge continuously), and illiquidity (a fraction
of the live market's own spread). Coefficients are stated parameters, not
per-instrument tuning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

REHEDGE_INTERVAL_YEARS = 1.0 / (24.0 * 365.0)  # assumed requoting cadence: once an hour


class QuotingError(ValueError):
    """Bad caller input, e.g. a non-positive number of ladder levels."""


@dataclass(frozen=True, slots=True)
class QuoteWidth:
    half_width: float
    fit_term: float
    gamma_term: float
    liquidity_term: float
    floor: float


def derive_width(
    *,
    vega: float,
    gamma: float,
    fit_residual_vol: float,
    underlying: float,
    vol: float,
    market_half_spread: float | None,
    fit_coef: float = 1.0,
    gamma_coef: float = 1.0,
    liquidity_coef: float = 0.5,
    floor: float = 0.0,
) -> QuoteWidth:
    """half_width = max(floor, fit_term + gamma_term + liquidity_term).

    gamma_term is 0.5*gamma*(expected move)^2, the textbook cost of gamma
    exposure over one rehedge interval, with the expected move sized from
    this option's own implied vol rather than a guessed constant.
    """
    hedge_move = underlying * vol * math.sqrt(REHEDGE_INTERVAL_YEARS)
    fit_term = fit_coef * abs(vega) * abs(fit_residual_vol)
    gamma_term = gamma_coef * abs(gamma) * hedge_move * hedge_move / 2.0
    liquidity_term = liquidity_coef * market_half_spread if market_half_spread else 0.0
    half_width = max(floor, fit_term + gamma_term + liquidity_term)
    return QuoteWidth(half_width, fit_term, gamma_term, liquidity_term, floor)


def reservation_price(mid: float, inventory: float, risk_aversion: float, vol: float, tau: float) -> float:
    """Avellaneda & Stoikov (2008): r = s - q*gamma*sigma^2*(T-t). Shades the
    quote away from inventory rather than leaving it centered on theo.
    """
    return mid - inventory * risk_aversion * vol * vol * max(tau, 0.0)


@dataclass(frozen=True, slots=True)
class QuoteLevel:
    level: int
    bid: float
    ask: float
    size: float


def quote_ladder(mid: float, half_width: float, *, levels: int, size: float, level_growth: float = 0.5) -> tuple[QuoteLevel, ...]:
    """Each level out is wider by level_growth, the usual "deeper size gets
    a worse average price" convention.
    """
    if levels < 1:
        raise QuotingError(f"levels must be >=1, got {levels}.")
    return tuple(
        QuoteLevel(
            level=i,
            bid=mid - half_width * (1.0 + level_growth * i),
            ask=mid + half_width * (1.0 + level_growth * i),
            size=size,
        )
        for i in range(levels)
    )


class MarketPosition(StrEnum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    CROSSED = "crossed"
    STRADDLING = "straddling"


def classify_against_market(our_bid: float, our_ask: float, market_bid: float, market_ask: float) -> MarketPosition:
    if our_bid > market_ask or our_ask < market_bid:
        return MarketPosition.CROSSED
    if our_bid >= market_bid and our_ask <= market_ask:
        return MarketPosition.INSIDE
    if our_bid <= market_bid and our_ask >= market_ask:
        return MarketPosition.OUTSIDE
    return MarketPosition.STRADDLING
