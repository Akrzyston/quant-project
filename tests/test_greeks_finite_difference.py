"""Analytic greeks against finite difference.

Two numerical points matter here.

Step size: a central difference carries truncation error of order h^2 and
cancellation error of order eps*|f|/h, so too small a step is as wrong as too
large. Both differences below are Richardson-extrapolated to kill the leading
truncation term; without that, gamma sits around 3e-4 from the analytic value on
correct code.

Tolerance: a fixed tolerance per greek is not enough. A deep in-the-money inverse
call has a value near a third of a coin and a vega near 2.5e-6, so the difference
being resolved sits eleven orders below the price and no step recovers more than
about six digits. Each check takes the looser of its per-greek tolerance and a
floor derived from that conditioning.
"""

from __future__ import annotations

import sys

import pytest

from voltk.models.bachelier import Bachelier
from voltk.models.base import CP
from voltk.models.black76 import Black76
from voltk.models.black_scholes import BlackScholes
from voltk.models.inverse import InverseOption

EPS = sys.float_info.epsilon
RATE = 0.03

TOLERANCE = {"delta": 1e-7, "gamma": 1e-5, "vega": 1e-7, "theta": 1e-6, "rho": 1e-7}

FIRST_STEP = 1e-4
SECOND_STEP = 1e-3

# Wings and at-the-money, one short and one long expiry. The inverse model keeps
# a second vol because it is the one that is not textbook.
CASES = (
    [
        pytest.param(Black76(), 100.0, k, t, 0.45, id=f"b76-{k:g}-{t:g}")
        for k in (70.0, 100.0, 140.0)
        for t in (0.05, 1.5)
    ]
    + [
        pytest.param(BlackScholes(), 100.0, k, t, 0.45, id=f"bs-{k:g}-{t:g}")
        for k in (70.0, 100.0, 140.0)
        for t in (0.05, 1.5)
    ]
    + [
        pytest.param(Bachelier(), 100.0, k, t, 25.0, id=f"bach-{k:g}-{t:g}")
        for k in (70.0, 100.0, 140.0)
        for t in (0.05, 1.5)
    ]
    + [
        pytest.param(InverseOption(), 60000.0, k, t, v, id=f"inv-{k:g}-{t:g}-{v:g}")
        for k in (40000.0, 60000.0, 90000.0)
        for t in (0.05, 1.5)
        for v in (0.4, 1.1)
    ]
)


def _central(fn, x: float, h: float) -> float:
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


def _second(fn, x: float, h: float) -> float:
    return (fn(x + h) - 2.0 * fn(x) + fn(x - h)) / (h * h)


def first_derivative(fn, x: float, h: float) -> float:
    return (4.0 * _central(fn, x, h / 2.0) - _central(fn, x, h)) / 3.0


def second_derivative(fn, x: float, h: float) -> float:
    return (4.0 * _second(fn, x, h / 2.0) - _second(fn, x, h)) / 3.0


def conditioning_floor(value: float, derivative: float, step: float, order: int) -> float:
    if derivative == 0.0 or step == 0.0:
        return 0.0
    return EPS * abs(value) / (step**order * abs(derivative))


def assert_matches(analytic, numeric, greek, *, value, step, order=1) -> None:
    floor = conditioning_floor(value, analytic, step, order)
    tolerance = max(TOLERANCE[greek], 20.0 * floor)
    scale = max(abs(numeric), abs(analytic), 1e-14)
    error = abs(analytic - numeric) / scale
    assert error <= tolerance, (
        f"{greek}: analytic {analytic:.12g} vs numeric {numeric:.12g}, relative error "
        f"{error:.2e} exceeds {tolerance:.2e} (conditioning floor {floor:.2e})"
    )


@pytest.mark.parametrize("model,underlying,strike,tau,vol", CASES)
@pytest.mark.parametrize("cp", list(CP))
def test_delta(model, underlying, strike, tau, vol, cp) -> None:
    value = model.price(underlying, strike, tau, vol, RATE, cp)
    step = underlying * FIRST_STEP
    numeric = first_derivative(
        lambda x: model.price(x, strike, tau, vol, RATE, cp), underlying, step
    )
    analytic = model.greeks(underlying, strike, tau, vol, RATE, cp).delta
    assert_matches(analytic, numeric, "delta", value=value, step=step)


@pytest.mark.parametrize("model,underlying,strike,tau,vol", CASES)
@pytest.mark.parametrize("cp", list(CP))
def test_gamma(model, underlying, strike, tau, vol, cp) -> None:
    value = model.price(underlying, strike, tau, vol, RATE, cp)
    step = underlying * SECOND_STEP
    numeric = second_derivative(
        lambda x: model.price(x, strike, tau, vol, RATE, cp), underlying, step
    )
    analytic = model.greeks(underlying, strike, tau, vol, RATE, cp).gamma
    assert_matches(analytic, numeric, "gamma", value=value, step=step, order=2)


@pytest.mark.parametrize("model,underlying,strike,tau,vol", CASES)
@pytest.mark.parametrize("cp", list(CP))
def test_vega(model, underlying, strike, tau, vol, cp) -> None:
    value = model.price(underlying, strike, tau, vol, RATE, cp)
    step = vol * FIRST_STEP
    numeric = first_derivative(
        lambda v: model.price(underlying, strike, tau, v, RATE, cp), vol, step
    )
    analytic = model.greeks(underlying, strike, tau, vol, RATE, cp).vega
    assert_matches(analytic, numeric, "vega", value=value, step=step)


@pytest.mark.parametrize("model,underlying,strike,tau,vol", CASES)
@pytest.mark.parametrize("cp", list(CP))
def test_theta(model, underlying, strike, tau, vol, cp) -> None:
    """Theta is dV/dt, so minus the derivative in time to expiry."""
    value = model.price(underlying, strike, tau, vol, RATE, cp)
    step = tau * FIRST_STEP
    numeric = -first_derivative(
        lambda t: model.price(underlying, strike, t, vol, RATE, cp), tau, step
    )
    analytic = model.greeks(underlying, strike, tau, vol, RATE, cp).theta
    assert_matches(analytic, numeric, "theta", value=value, step=step)


@pytest.mark.parametrize("model,underlying,strike,tau,vol", CASES)
@pytest.mark.parametrize("cp", list(CP))
def test_rho(model, underlying, strike, tau, vol, cp) -> None:
    value = model.price(underlying, strike, tau, vol, RATE, cp)
    step = 1e-5
    numeric = first_derivative(
        lambda r: model.price(underlying, strike, tau, vol, r, cp), RATE, step
    )
    analytic = model.greeks(underlying, strike, tau, vol, RATE, cp).rho
    assert_matches(analytic, numeric, "rho", value=value, step=step)


def test_inverse_rho_is_exactly_zero() -> None:
    """The coin premium does not depend on the rate once the forward is given."""
    model = InverseOption()
    for cp in CP:
        assert model.greeks(60000.0, 65000.0, 0.5, 0.6, 0.03, cp).rho == 0.0
        assert model.price(60000.0, 65000.0, 0.5, 0.6, 0.0, cp) == model.price(
            60000.0, 65000.0, 0.5, 0.6, 0.9, cp
        )
