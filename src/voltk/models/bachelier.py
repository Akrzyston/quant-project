"""Bachelier: forward-based, normal vol.

Vol is quoted in price units per root year rather than as a fraction, so the
model stays defined when the forward or the strike is negative. That is the
whole reason it exists.
"""

from __future__ import annotations

import math

from voltk.greeks import Greeks, Unit
from voltk.models.bounds import Bounds, normal_bounds
from voltk.models.solver import implied_vol_for
from voltk.models.bounds import Bounds, normal_bounds
from voltk.models.solver import implied_vol_for
from voltk.models.base import MIN_TAU, MIN_VOL, CP, discount, norm_cdf, norm_pdf
from voltk.models.base import PricingError


class Bachelier:
    underlying_is_forward = True
    allows_negative_prices = True

    def _check(self, tau: float, rate: float) -> None:
        if tau < 0:
            raise PricingError(f"Time to expiry must not be negative, got {tau}.")
        if not math.isfinite(rate):
            raise PricingError(f"Rate must be finite, got {rate}.")

    def price(
        self, forward: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> float:
        self._check(tau, rate)
        df = discount(rate, tau)
        s = cp.sign

        if tau <= MIN_TAU or vol <= MIN_VOL:
            return df * max(s * (forward - strike), 0.0)

        sigma_root_tau = vol * math.sqrt(tau)
        d = s * (forward - strike) / sigma_root_tau
        return df * (s * (forward - strike) * norm_cdf(d) + sigma_root_tau * norm_pdf(d))

    def greeks(
        self, forward: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> Greeks:
        self._check(tau, rate)
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

        root_tau = math.sqrt(tau)
        sigma_root_tau = vol * root_tau
        d = s * (forward - strike) / sigma_root_tau
        pdf_d = norm_pdf(d)
        value = df * (s * (forward - strike) * norm_cdf(d) + sigma_root_tau * pdf_d)

        return Greeks(
            delta=df * s * norm_cdf(d),
            gamma=df * pdf_d / sigma_root_tau,
            vega=df * root_tau * pdf_d,
            theta=rate * value - df * vol * pdf_d / (2.0 * root_tau),
            rho=-tau * value,
            unit=Unit.QUOTE,
            vega_bump=1.0,
            theta_period=1.0,
        )

    def bounds(
        self, forward: float, strike: float, tau: float, rate: float, cp: CP
    ) -> Bounds:
        return normal_bounds(forward, strike, tau, rate, cp)

    def implied_vol(
        self, price: float, forward: float, strike: float, tau: float, rate: float, cp: CP
    ) -> float:
        self.bounds(forward, strike, tau, rate, cp).check(price, type(self).__name__)
        return implied_vol_for(self, price, forward, strike, tau, rate, cp)

    def vol_bracket(self, forward: float, strike: float) -> tuple[float, float, float]:
        """Vol is in price units per root year, so the bracket scales with price."""
        scale = max(abs(forward), abs(strike), 1.0)
        return 1e-12, scale, 100.0 * scale
