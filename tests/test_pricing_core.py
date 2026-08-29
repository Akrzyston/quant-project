"""Round trip, parity and bounds for every model.

Grids are deliberately small: wings plus at-the-money, a short and a long expiry.
A wider grid re-tests the same branches and turns one bug into fifty red lines.
"""

from __future__ import annotations

import math

import pytest

from voltk.models.bachelier import Bachelier
from voltk.models.base import CP, PricingError
from voltk.models.black76 import Black76
from voltk.models.black_scholes import BlackScholes
from voltk.models.bounds import ArbitrageError
from voltk.models.inverse import InverseOption

RATE = 0.03
FORWARD = 100.0
STRIKES = [70.0, 100.0, 140.0]
TAUS = [0.02, 1.5]

LOGNORMAL = [Black76(), BlackScholes(), InverseOption()]


def name(model) -> str:
    return type(model).__name__


def vols_for(model) -> list[float]:
    return [10.0, 45.0] if isinstance(model, Bachelier) else [0.15, 1.2]


@pytest.mark.parametrize("model", LOGNORMAL + [Bachelier()], ids=name)
@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("cp", list(CP))
def test_round_trip_price_to_vol_to_price(model, strike, tau, cp) -> None:
    for vol in vols_for(model):
        price = model.price(FORWARD, strike, tau, vol, RATE, cp)
        recovered = model.implied_vol(price, FORWARD, strike, tau, RATE, cp)
        assert abs(model.price(FORWARD, strike, tau, recovered, RATE, cp) - price) < 1e-10


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("tau", TAUS)
def test_forward_model_parity(strike, tau) -> None:
    """C - P = D(F - K) for both forward-based models."""
    expected = math.exp(-RATE * tau) * (FORWARD - strike)
    for model, vol in ((Black76(), 0.4), (Bachelier(), 25.0)):
        call = model.price(FORWARD, strike, tau, vol, RATE, CP.CALL)
        put = model.price(FORWARD, strike, tau, vol, RATE, CP.PUT)
        assert call - put == pytest.approx(expected, abs=1e-12), name(model)


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("tau", TAUS)
def test_spot_model_parity(strike, tau) -> None:
    """C - P = S - K*D when the underlying carries no yield."""
    model = BlackScholes()
    call = model.price(FORWARD, strike, tau, 0.4, RATE, CP.CALL)
    put = model.price(FORWARD, strike, tau, 0.4, RATE, CP.PUT)
    assert call - put == pytest.approx(FORWARD - strike * math.exp(-RATE * tau), abs=1e-12)


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("tau", TAUS)
def test_inverse_parity(strike, tau) -> None:
    """Coin settlement gives C - P = 1 - K/F, not D(F - K)."""
    model = InverseOption()
    call = model.price(FORWARD, strike, tau, 0.6, RATE, CP.CALL)
    put = model.price(FORWARD, strike, tau, 0.6, RATE, CP.PUT)
    assert model.parity_gap(call, put, FORWARD, strike) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("model", LOGNORMAL + [Bachelier()], ids=name)
@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("cp", list(CP))
def test_price_stays_inside_no_arbitrage_bounds(model, strike, cp) -> None:
    for tau in TAUS:
        bounds = model.bounds(FORWARD, strike, tau, RATE, cp)
        for vol in vols_for(model):
            price = model.price(FORWARD, strike, tau, vol, RATE, cp)
            assert bounds.contains(price, tolerance=1e-12), (
                f"{name(model)} {cp.value} K={strike} tau={tau} vol={vol}: {price} "
                f"outside {bounds}"
            )


@pytest.mark.parametrize("model", LOGNORMAL, ids=name)
def test_price_above_the_upper_bound_is_refused(model) -> None:
    bounds = model.bounds(FORWARD, 100.0, 0.5, RATE, CP.CALL)
    with pytest.raises(ArbitrageError):
        model.implied_vol(bounds.upper * 2.0, FORWARD, 100.0, 0.5, RATE, CP.CALL)


@pytest.mark.parametrize("model", LOGNORMAL + [Bachelier()], ids=name)
@pytest.mark.parametrize("cp", list(CP))
def test_zero_vol_and_zero_tau_give_intrinsic(model, cp) -> None:
    strike = 140.0 if cp is CP.PUT else 70.0
    lower = model.bounds(FORWARD, strike, 0.5, RATE, cp).lower
    assert model.price(FORWARD, strike, 0.5, 0.0, RATE, cp) == pytest.approx(lower, abs=1e-12)

    undiscounted = max(cp.sign * (FORWARD - strike), 0.0)
    if isinstance(model, InverseOption):
        undiscounted /= FORWARD
    assert model.price(FORWARD, strike, 0.0, vols_for(model)[0], RATE, cp) == pytest.approx(
        undiscounted, abs=1e-12
    )


def test_spot_and_forward_parameterisations_agree_on_price() -> None:
    """Same model, two parameterisations. Prices match at the implied forward."""
    spot, strike, tau, vol = 100.0, 105.0, 0.75, 0.35
    forward = spot * math.exp(RATE * tau)
    for cp in CP:
        assert BlackScholes().price(spot, strike, tau, vol, RATE, cp) == pytest.approx(
            Black76().price(forward, strike, tau, vol, RATE, cp), abs=1e-12
        )


def test_spot_and_forward_parameterisations_disagree_on_rho() -> None:
    """Black-76 holds the forward fixed, so only the discount factor moves."""
    spot, strike, tau, vol = 100.0, 105.0, 0.75, 0.35
    forward = spot * math.exp(RATE * tau)
    assert BlackScholes().greeks(spot, strike, tau, vol, RATE, CP.CALL).rho > 0
    assert Black76().greeks(forward, strike, tau, vol, RATE, CP.CALL).rho < 0


@pytest.mark.parametrize("model", LOGNORMAL, ids=name)
def test_lognormal_models_reject_non_positive_inputs(model) -> None:
    with pytest.raises(PricingError):
        model.price(-1.0, 100.0, 0.5, 0.3, RATE, CP.CALL)
    with pytest.raises(PricingError):
        model.price(FORWARD, -100.0, 0.5, 0.3, RATE, CP.CALL)
    with pytest.raises(PricingError):
        model.price(FORWARD, 100.0, -0.5, 0.3, RATE, CP.CALL)


def test_bachelier_prices_negative_underlyings() -> None:
    """The reason the model is in the suite at all."""
    price = Bachelier().price(-5.0, -10.0, 0.5, 20.0, RATE, CP.CALL)
    assert price > 0 and math.isfinite(price)
    with pytest.raises(PricingError):
        Black76().price(-5.0, -10.0, 0.5, 0.3, RATE, CP.CALL)
