"""Black-76: forward-based, lognormal.

The forward is an input, not derived from spot, so rho picks up only the
discount factor. That is the whole difference from the spot parameterisation
and it is a real difference in the risk report, not a cosmetic one.
"""

from __future__ import annotations

import math

from voltk.greeks import Greeks, Unit
from voltk.models.bounds import Bounds, lognormal_bounds
from voltk.models.solver import implied_vol_for
from voltk.models.bounds import Bounds, lognormal_bounds
from voltk.models.solver import implied_vol_for
from voltk.models.base import (
    MIN_TAU,
    MIN_VOL,
    CP,
    check_inputs,
    discount,
    intrinsic,
    norm_cdf,
    norm_pdf,
)


def d1_d2(forward: float, strike: float, tau: float, vol: float) -> tuple[float, float]:
    sigma_root_tau = vol * math.sqrt(tau)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * tau) / sigma_root_tau
    return d1, d1 - sigma_root_tau


class Black76:
    underlying_is_forward = True

    def price(
        self, forward: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> float:
        check_inputs(forward, strike, tau, rate)
        df = discount(rate, tau)
        if tau <= MIN_TAU or vol <= MIN_VOL:
            return df * intrinsic(forward, strike, cp)

        d1, d2 = d1_d2(forward, strike, tau, vol)
        s = cp.sign
        return df * s * (forward * norm_cdf(s * d1) - strike * norm_cdf(s * d2))

    def greeks(
        self, forward: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> Greeks:
        check_inputs(forward, strike, tau, rate)
        df = discount(rate, tau)
        s = cp.sign

        if tau <= MIN_TAU or vol <= MIN_VOL:
            in_money = float(s * (forward - strike) > 0)
            return Greeks(
                delta=df * s * in_money,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=-tau * self.price(forward, strike, tau, vol, rate, cp),
                unit=Unit.QUOTE,
                vega_bump=1.0,
                theta_period=1.0,
            )

        d1, d2 = d1_d2(forward, strike, tau, vol)
        root_tau = math.sqrt(tau)
        pdf_d1 = norm_pdf(d1)
        value = df * s * (forward * norm_cdf(s * d1) - strike * norm_cdf(s * d2))

        # forward * pdf(d1) == strike * pdf(d2), which is what collapses the
        # theta derivative to a single term.
        return Greeks(
            delta=df * s * norm_cdf(s * d1),
            gamma=df * pdf_d1 / (forward * vol * root_tau),
            vega=df * forward * pdf_d1 * root_tau,
            theta=rate * value - df * forward * pdf_d1 * vol / (2.0 * root_tau),
            rho=-tau * value,
            unit=Unit.QUOTE,
            vega_bump=1.0,
            theta_period=1.0,
        )

    def bounds(
        self, forward: float, strike: float, tau: float, rate: float, cp: CP
    ) -> Bounds:
        return lognormal_bounds(forward, strike, tau, rate, cp)

    def implied_vol(
        self, price: float, forward: float, strike: float, tau: float, rate: float, cp: CP
    ) -> float:
        self.bounds(forward, strike, tau, rate, cp).check(price, type(self).__name__)
        return implied_vol_for(self, price, forward, strike, tau, rate, cp)

    def vol_bracket(self, forward: float, strike: float) -> tuple[float, float, float]:
        """Vol is a fraction here, so the bracket is scale-free."""
        return 1e-12, 5.0, 50.0
