"""Cash Greeks derived from coin Greeks: not a units conversion.

InverseOption's Greeks are FORWARD Greeks (delta = dV_coin/dF), and cash
value V_cash(S) = V_coin(F(S))*S is a genuinely spot quantity, with
F(S) = S*exp(rate*tau). Delta and gamma pick up a structural correction from
that self-quanto product rule; a candidate who just multiplies by spot gets
delta right by a coincidence of the algebra and gamma wrong by a real amount.
"""

from __future__ import annotations

import math

import pytest

from tests.test_greeks_finite_difference import first_derivative, second_derivative
from voltk.greeks import cash_greeks_from_coin
from voltk.models.base import CP
from voltk.models.inverse import InverseOption

RATE = 0.08  # deliberately large: makes the gamma gap visible, not lost in noise
STRIKES = (40_000.0, 60_000.0, 90_000.0)
TAUS = (0.05, 1.5)
VOL = 0.6
SPOT = 60_000.0


def _forward(spot: float, tau: float) -> float:
    return spot * math.exp(RATE * tau)


def _cash_value(spot: float, strike: float, tau: float, cp: CP) -> float:
    """V_cash(S), built independently of cash_greeks_from_coin."""
    model = InverseOption()
    return model.price(_forward(spot, tau), strike, tau, VOL, RATE, cp) * spot


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("cp", list(CP))
def test_cash_delta_is_not_naive_unit_conversion(strike, tau, cp) -> None:
    model = InverseOption()
    forward = _forward(SPOT, tau)
    coin = model.greeks(forward, strike, tau, VOL, RATE, cp)
    coin_price = model.price(forward, strike, tau, VOL, RATE, cp)

    cash = cash_greeks_from_coin(coin, coin_price, forward, SPOT)

    assert cash.delta != pytest.approx(coin.delta * SPOT)


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("cp", list(CP))
def test_cash_gamma_is_not_naive_forward_substitution(strike, tau, cp) -> None:
    """Substituting F for S in 2*delta + F*gamma (skipping the exp(rate*tau)
    factor) recovers delta exactly but is off by a material amount in gamma.
    """
    model = InverseOption()
    forward = _forward(SPOT, tau)
    coin = model.greeks(forward, strike, tau, VOL, RATE, cp)
    coin_price = model.price(forward, strike, tau, VOL, RATE, cp)

    cash = cash_greeks_from_coin(coin, coin_price, forward, SPOT)
    naive_gamma = 2.0 * coin.delta + forward * coin.gamma

    relative_gap = abs(cash.gamma - naive_gamma) / abs(cash.gamma)
    assert relative_gap > 1e-3


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("cp", list(CP))
def test_cash_delta_and_gamma_match_finite_difference_of_cash_value(strike, tau, cp) -> None:
    model = InverseOption()
    forward = _forward(SPOT, tau)
    coin = model.greeks(forward, strike, tau, VOL, RATE, cp)
    coin_price = model.price(forward, strike, tau, VOL, RATE, cp)
    cash = cash_greeks_from_coin(coin, coin_price, forward, SPOT)

    step = SPOT * 1e-4
    fd_delta = first_derivative(lambda s: _cash_value(s, strike, tau, cp), SPOT, step)
    fd_gamma = second_derivative(lambda s: _cash_value(s, strike, tau, cp), SPOT, step * 10)

    assert cash.delta == pytest.approx(fd_delta, rel=1e-5)
    assert cash.gamma == pytest.approx(fd_gamma, rel=1e-3)


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("cp", list(CP))
def test_cash_vega_theta_are_simple_spot_multiples(strike, tau, cp) -> None:
    """The explicit contrast case: unlike delta/gamma, these ARE Greek*spot,
    since a vol/tau bump holds both F and S fixed.
    """
    model = InverseOption()
    forward = _forward(SPOT, tau)
    coin = model.greeks(forward, strike, tau, VOL, RATE, cp)
    coin_price = model.price(forward, strike, tau, VOL, RATE, cp)

    cash = cash_greeks_from_coin(coin, coin_price, forward, SPOT)

    assert cash.vega == pytest.approx(coin.vega * SPOT)
    assert cash.theta == pytest.approx(coin.theta * SPOT)
    assert cash.vanna == pytest.approx(coin.vanna * SPOT)
    assert cash.volga == pytest.approx(coin.volga * SPOT)


def test_cash_rho_is_exactly_zero() -> None:
    model = InverseOption()
    forward = _forward(SPOT, 0.5)
    for cp in CP:
        coin = model.greeks(forward, 60_000.0, 0.5, VOL, RATE, cp)
        coin_price = model.price(forward, 60_000.0, 0.5, VOL, RATE, cp)
        cash = cash_greeks_from_coin(coin, coin_price, forward, SPOT)
        assert cash.rho == 0.0
