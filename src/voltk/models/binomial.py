"""Cox-Ross-Rubinstein binomial tree with early exercise.

Spot-parameterised. Discrete cash dividends use the escrowed model: present
value of dividends still to be paid is stripped off spot, the tree is built
on the remainder, and the escrow is added back at each node. Delta and
gamma read off the tree's own early nodes; vega, rho, vanna and volga have
no tree analogue and are central differences over rebuilt trees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from voltk.greeks import Greeks, Unit
from voltk.models.base import MIN_TAU, MIN_VOL, CP, check_inputs, discount
from voltk.models.bounds import Bounds, spot_bounds
from voltk.models.solver import implied_vol_for

DEFAULT_STEPS = 512


@dataclass(frozen=True, slots=True)
class Dividend:
    """A cash dividend of `amount` paid at `time` years from now."""

    time: float
    amount: float


class Binomial:
    underlying_is_forward = False

    def __init__(
        self,
        steps: int = DEFAULT_STEPS,
        american: bool = True,
        dividends: tuple[Dividend, ...] = (),
    ) -> None:
        if steps < 2:
            raise ValueError(f"A tree needs at least two steps, got {steps}.")
        self.steps = steps
        self.american = american
        self.dividends = tuple(sorted(dividends, key=lambda d: d.time))

    def escrow(self, tau: float, rate: float, after: float = 0.0) -> float:
        """Present value at time `after` of dividends paid between then and expiry."""
        return sum(
            d.amount * math.exp(-rate * (d.time - after))
            for d in self.dividends
            if after <= d.time <= tau
        )

    def _lattice(self, spot: float, strike: float, tau: float, vol: float, rate: float, cp: CP):
        """Return option values at steps 0, 1 and 2 plus the spot grid and dt."""
        steps = self.steps
        dt = tau / steps
        up = math.exp(vol * math.sqrt(dt))
        down = 1.0 / up
        growth = math.exp(rate * dt)
        p = (growth - down) / (up - down)
        p = min(max(p, 0.0), 1.0)
        df = 1.0 / growth

        seed = spot - self.escrow(tau, rate)
        if seed <= 0:
            raise ValueError("Dividends exceed spot; the escrowed tree is undefined.")

        def spot_at(step: int, ups: int) -> float:
            t = step * dt
            return seed * up**ups * down ** (step - ups) + self.escrow(tau, rate, after=t)

        values = [
            max(cp.sign * (spot_at(steps, j) - strike), 0.0) for j in range(steps + 1)
        ]

        kept: dict[int, list[float]] = {}
        for step in range(steps - 1, -1, -1):
            values = [df * (p * values[j + 1] + (1.0 - p) * values[j]) for j in range(step + 1)]
            if self.american:
                values = [
                    max(values[j], cp.sign * (spot_at(step, j) - strike)) for j in range(step + 1)
                ]
            if step <= 2:
                kept[step] = list(values)

        grid = {step: [spot_at(step, j) for j in range(step + 1)] for step in kept}
        return kept, grid, dt

    def price(
        self, spot: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> float:
        check_inputs(spot, strike, tau, rate)
        if tau <= MIN_TAU or vol <= MIN_VOL:
            return max(cp.sign * (spot - strike), 0.0)
        values, _, _ = self._lattice(spot, strike, tau, vol, rate, cp)
        return values[0][0]

    def greeks(
        self, spot: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> Greeks:
        check_inputs(spot, strike, tau, rate)
        if tau <= MIN_TAU or vol <= MIN_VOL:
            in_money = float(cp.sign * (spot - strike) > 0)
            return Greeks(
                delta=cp.sign * in_money,
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

        values, grid, dt = self._lattice(spot, strike, tau, vol, rate, cp)

        delta = (values[1][1] - values[1][0]) / (grid[1][1] - grid[1][0])
        upper = (values[2][2] - values[2][1]) / (grid[2][2] - grid[2][1])
        lower = (values[2][1] - values[2][0]) / (grid[2][1] - grid[2][0])
        gamma = (upper - lower) / (0.5 * (grid[2][2] - grid[2][0]))
        theta = (values[2][1] - values[0][0]) / (2.0 * dt)

        # No tree analogue for these two, so reprice.
        bump_vol = max(vol * 1e-3, 1e-6)
        vega = (
            self.price(spot, strike, tau, vol + bump_vol, rate, cp)
            - self.price(spot, strike, tau, vol - bump_vol, rate, cp)
        ) / (2.0 * bump_vol)

        bump_rate = 1e-4
        rho = (
            self.price(spot, strike, tau, vol, rate + bump_rate, cp)
            - self.price(spot, strike, tau, vol, rate - bump_rate, cp)
        ) / (2.0 * bump_rate)

        # No tree analogue for these either: vanna is the cross-partial and
        # volga the second vol-partial, both by repricing on rebuilt trees.
        bump_spot = max(spot * 1e-3, 1e-6)
        vanna = (
            self.price(spot + bump_spot, strike, tau, vol + bump_vol, rate, cp)
            - self.price(spot + bump_spot, strike, tau, vol - bump_vol, rate, cp)
            - self.price(spot - bump_spot, strike, tau, vol + bump_vol, rate, cp)
            + self.price(spot - bump_spot, strike, tau, vol - bump_vol, rate, cp)
        ) / (4.0 * bump_spot * bump_vol)

        base_price = self.price(spot, strike, tau, vol, rate, cp)
        volga = (
            self.price(spot, strike, tau, vol + bump_vol, rate, cp)
            - 2.0 * base_price
            + self.price(spot, strike, tau, vol - bump_vol, rate, cp)
        ) / (bump_vol * bump_vol)

        return Greeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            vanna=vanna,
            volga=volga,
            unit=Unit.QUOTE,
            vega_bump=1.0,
            theta_period=1.0,
        )

    def early_exercise_premium(
        self, spot: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> float:
        """American value less the European value on the same tree.

        Differencing two trees rather than against a closed form keeps the
        discretisation error out of the answer.
        """
        european = Binomial(self.steps, american=False, dividends=self.dividends)
        return self.price(spot, strike, tau, vol, rate, cp) - european.price(
            spot, strike, tau, vol, rate, cp
        )

    def bounds(
        self, spot: float, strike: float, tau: float, rate: float, cp: CP
    ) -> Bounds:
        european = spot_bounds(spot, strike, tau, rate, cp)
        if not self.american:
            return european
        # An American option is worth at least its immediate exercise value.
        return Bounds(
            lower=max(european.lower, max(cp.sign * (spot - strike), 0.0)),
            upper=spot if cp is CP.CALL else strike,
        )

    def implied_vol(
        self, price: float, spot: float, strike: float, tau: float, rate: float, cp: CP
    ) -> float:
        self.bounds(spot, strike, tau, rate, cp).check(price, type(self).__name__)
        return implied_vol_for(self, price, spot, strike, tau, rate, cp)

    def vol_bracket(self, spot: float, strike: float) -> tuple[float, float, float]:
        return 1e-9, 5.0, 50.0
