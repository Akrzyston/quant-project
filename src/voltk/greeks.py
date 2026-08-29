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
    """Cash (quote-currency) Greeks of a position whose coin-denominated value
    is V_coin(F), given V_cash(S) = V_coin(F(S)) * S with F(S) = S*exp(rate*tau)
    -- the exact futures-basis relationship Universe.implied_rate already
    solves. `coin` must be Greeks computed AT `forward`: every forward-
    parameterised model in this library (Black76, Bachelier, InverseOption)
    reports delta this way, as dV/dF, not dV/dS.

    Delta and gamma pick up a structural correction from the self-quanto
    product rule -- this is not a units conversion, it changes the formula.
    Gamma additionally needs the F/S = exp(rate*tau) factor: naively
    substituting F for S in 2*delta + F*gamma recovers delta exactly (by a
    coincidence of the algebra) but is off by a material amount in gamma,
    verified against a direct finite difference of V_cash(S) itself. Vega,
    theta, rho, vanna and volga are simple spot multiples, since a vol/tau/
    rate bump holds both F and S fixed -- no product-rule term appears.
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
