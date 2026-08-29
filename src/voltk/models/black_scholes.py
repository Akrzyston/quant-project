"""Black-Scholes: spot-based, lognormal.

The first price argument is spot, not forward. With a carry yield of zero this
reprices identically to Black-76 at forward = spot * exp(rate * tau), which is
the sense in which they are the same model; rho differs because here the forward
moves with the rate.
"""

from __future__ import annotations

import math

from voltk.greeks import Greeks, Unit
from voltk.models.bounds import Bounds, spot_bounds
from voltk.models.solver import implied_vol_for
from voltk.models.bounds import Bounds, spot_bounds
from voltk.models.solver import implied_vol_for
from voltk.models.base import (
    MIN_TAU,
    MIN_VOL,
    CP,
    check_inputs,
    discount,
    norm_cdf,
    norm_pdf,
)


class BlackScholes:
    underlying_is_forward = False

    def __init__(self, carry_yield: float = 0.0) -> None:
        # Continuous yield on the underlying: dividends for equities, the
        # convenience or funding yield elsewhere.
        self.carry_yield = carry_yield

    def forward(self, spot: float, tau: float, rate: float) -> float:
        return spot * math.exp((rate - self.carry_yield) * tau)

    def _d1_d2(
        self, spot: float, strike: float, tau: float, vol: float, rate: float
    ) -> tuple[float, float]:
        sigma_root_tau = vol * math.sqrt(tau)
        d1 = (
            math.log(spot / strike) + (rate - self.carry_yield + 0.5 * vol * vol) * tau
        ) / sigma_root_tau
        return d1, d1 - sigma_root_tau

    def price(
        self, spot: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> float:
        check_inputs(spot, strike, tau, rate)
        s = cp.sign
        carry_df = math.exp(-self.carry_yield * tau)
        rate_df = discount(rate, tau)

        if tau <= MIN_TAU or vol <= MIN_VOL:
            return max(s * (spot * carry_df - strike * rate_df), 0.0)

        d1, d2 = self._d1_d2(spot, strike, tau, vol, rate)
        return s * (spot * carry_df * norm_cdf(s * d1) - strike * rate_df * norm_cdf(s * d2))

    def greeks(
        self, spot: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> Greeks:
        check_inputs(spot, strike, tau, rate)
        s = cp.sign
        carry_df = math.exp(-self.carry_yield * tau)
        rate_df = discount(rate, tau)

        if tau <= MIN_TAU or vol <= MIN_VOL:
            in_money = float(s * (spot - strike) > 0)
            return Greeks(
                delta=s * in_money,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=0.0,
                vanna=0.0,
                volga=0.0,
                unit=Unit.QUOTE,
                vega_bump=1.0,
                theta_period=1.0,
            )

        d1, d2 = self._d1_d2(spot, strike, tau, vol, rate)
        root_tau = math.sqrt(tau)
        pdf_d1 = norm_pdf(d1)
        vega = spot * carry_df * pdf_d1 * root_tau

        theta = (
            -spot * carry_df * pdf_d1 * vol / (2.0 * root_tau)
            - s * rate * strike * rate_df * norm_cdf(s * d2)
            + s * self.carry_yield * spot * carry_df * norm_cdf(s * d1)
        )

        return Greeks(
            delta=s * carry_df * norm_cdf(s * d1),
            gamma=carry_df * pdf_d1 / (spot * vol * root_tau),
            vega=vega,
            theta=theta,
            rho=s * strike * tau * rate_df * norm_cdf(s * d2),
            vanna=-vega * d2 / (spot * vol * root_tau),
            volga=vega * d1 * d2 / vol,
            unit=Unit.QUOTE,
            vega_bump=1.0,
            theta_period=1.0,
        )

    def bounds(
        self, forward: float, strike: float, tau: float, rate: float, cp: CP
    ) -> Bounds:
        return spot_bounds(forward, strike, tau, rate, cp, self.carry_yield)

    def implied_vol(
        self, price: float, forward: float, strike: float, tau: float, rate: float, cp: CP
    ) -> float:
        self.bounds(forward, strike, tau, rate, cp).check(price, type(self).__name__)
        return implied_vol_for(self, price, forward, strike, tau, rate, cp)

    def vol_bracket(self, forward: float, strike: float) -> tuple[float, float, float]:
        """Vol is a fraction here, so the bracket is scale-free."""
        return 1e-12, 5.0, 50.0
