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


def cash_greeks_from_coin(coin: Greeks, coin_price: float, forward: float, spot: float) -> Greeks:
    """Cash (quote-currency) Greeks from coin Greeks, given
    V_cash(S) = V_coin(F(S)) * S with F(S) = S*exp(rate*tau). `coin` must be
    Greeks computed at `forward` (every forward-parameterised model here
    reports delta as dV/dF). Delta and gamma pick up a real structural
    correction from the self-quanto product rule, not a plain units
    conversion -- gamma additionally needs the F/S factor, or it's off by a
    material amount versus a direct finite difference of V_cash(S). Vega,
    theta, rho, vanna and volga are simple spot multiples, since those bumps
    hold both F and S fixed.
    """
    ratio = forward / spot
    delta = coin_price + forward * coin.delta
    gamma = ratio * (forward * coin.gamma + 2.0 * coin.delta)
    return Greeks(
        delta=delta,
        gamma=gamma,
        vega=spot * coin.vega,
        theta=spot * coin.theta,
        rho=spot * coin.rho,
        vanna=spot * coin.vanna,
        volga=spot * coin.volga,
        unit=Unit.QUOTE,
        vega_bump=coin.vega_bump,
        theta_period=coin.theta_period,
    )
