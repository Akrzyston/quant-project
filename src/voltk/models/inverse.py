"""Inverse (coin-settled) options.

A coin-settled call pays max(S-K,0)/S coins -- bounded, concave, unlike the
unbounded quote-settled payoff. Priced two ways, which must agree: as a
replication of K puts on 1/S under the coin numeraire, and directly as the
ordinary Black-76 premium over spot, rate-independent because the discount
factor cancels against the forward. Full derivation in
docs/inverse_replication.md.
"""

from __future__ import annotations

import math

from voltk.greeks import Greeks, Unit
from voltk.models.bounds import Bounds, inverse_bounds
from voltk.models.solver import implied_vol_for
from voltk.models.bounds import Bounds, inverse_bounds
from voltk.models.solver import implied_vol_for
from voltk.models.base import MIN_TAU, MIN_VOL, CP, check_inputs, norm_cdf, norm_pdf
from voltk.models.black76 import Black76, d1_d2


class InverseOption:
    """Prices in the settlement currency, per one unit of quote-currency notional."""

    underlying_is_forward = True
    settles_in_base = True

    def __init__(self) -> None:
        self._black = Black76()

    def price(
        self, forward: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> float:
        check_inputs(forward, strike, tau, rate)
        if tau <= MIN_TAU or vol <= MIN_VOL:
            return max(cp.sign * (forward - strike), 0.0) / forward
        # Undiscounted Black-76 over the forward: the rate cancels between the
        # quote-currency value and the spot used to convert it.
        return self._black.price(forward, strike, tau, vol, 0.0, cp) / forward

    def price_via_replication(
        self, forward: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> float:
        """K puts on the reciprocal price, struck at 1/K, forward 1/F."""
        check_inputs(forward, strike, tau, rate)
        if tau <= MIN_TAU or vol <= MIN_VOL:
            return max(cp.sign * (forward - strike), 0.0) / forward
        mirrored = CP.PUT if cp is CP.CALL else CP.CALL
        return strike * self._black.price(
            1.0 / forward, 1.0 / strike, tau, vol, 0.0, mirrored
        )

    def greeks(
        self, forward: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> Greeks:
        """Sensitivities of the coin premium to the forward.

        Put greeks come from differentiating the parity relation C - P = 1 - K/F
        rather than being derived separately.
        """
        check_inputs(forward, strike, tau, rate)

        if tau <= MIN_TAU or vol <= MIN_VOL:
            return Greeks(
                delta=0.0,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=0.0,
                vanna=0.0,
                volga=0.0,
                unit=Unit.BASE,
                vega_bump=1.0,
                theta_period=1.0,
            )

        d1, d2 = d1_d2(forward, strike, tau, vol)
        root_tau = math.sqrt(tau)
        pdf_d2 = norm_pdf(d2)

        # V_call = (F N(d1) - K N(d2)) / F, so dV/dF collapses to K N(d2) / F^2
        # once the F N(d1) terms cancel.
        delta = strike * norm_cdf(d2) / (forward * forward)
        gamma = (
            strike
            * (pdf_d2 / (vol * root_tau) - 2.0 * norm_cdf(d2))
            / (forward * forward * forward)
        )
        # F pdf(d1) == K pdf(d2) leaves a single term in both vega and theta.
        vega = strike * pdf_d2 * root_tau / forward
        theta = -strike * pdf_d2 * vol / (2.0 * root_tau * forward)
        # Same cancellation carries vanna and volga: cp-independent, like
        # vega and theta above, not split by the parity-derived put branch.
        vanna = -(vega / forward) * (1.0 + d2 / (vol * root_tau))
        volga = vega * d1 * d2 / vol

        if cp is CP.PUT:
            delta -= strike / (forward * forward)
            gamma += 2.0 * strike / (forward * forward * forward)

        return Greeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=0.0,
            vanna=vanna,
            volga=volga,
            unit=Unit.BASE,
            vega_bump=1.0,
            theta_period=1.0,
        )

    def quote_currency_greeks(
        self, forward: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> Greeks:
        """The ordinary Black-76 risk, for comparison against the coin view."""
        return self._black.greeks(forward, strike, tau, vol, rate, cp)

    @staticmethod
    def parity_gap(
        call_price: float, put_price: float, forward: float, strike: float
    ) -> float:
        """C - P - (1 - K/F). Zero for an arbitrage-free pair."""
        return call_price - put_price - (1.0 - strike / forward)

    def bounds(
        self, forward: float, strike: float, tau: float, rate: float, cp: CP
    ) -> Bounds:
        return inverse_bounds(forward, strike, tau, rate, cp)

    def implied_vol(
        self, price: float, forward: float, strike: float, tau: float, rate: float, cp: CP
    ) -> float:
        self.bounds(forward, strike, tau, rate, cp).check(price, type(self).__name__)
        return implied_vol_for(self, price, forward, strike, tau, rate, cp)

    def vol_bracket(self, forward: float, strike: float) -> tuple[float, float, float]:
        """Vol is a fraction here, so the bracket is scale-free."""
        return 1e-12, 5.0, 50.0
