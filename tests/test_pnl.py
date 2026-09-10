"""P&L attribution: the Taylor identity, its residual, and unit conventions.

Residual shrinks as O(h^2) when spot/vol/time all move by a small common
factor together (a real 2nd-order Taylor property, checked before picking
the tolerances below) -- but NOT when only some of the three
move a little while another moves by a fixed, disproportionate amount (e.g.
one full day of theta while spot/vol barely move), so the "small move" case
here scales all three together.
"""

from __future__ import annotations

import pytest

from voltk.greeks import Greeks, Unit
from voltk.models.base import CP
from voltk.models.black76 import Black76
from voltk.pnl import attribute_pnl

MODEL = Black76()
STRIKE = 65_000.0
RATE = 0.05
FORWARD = 60_000.0
VOL = 0.6
TAU = 0.5


def _move(scale: float):
    forward2 = FORWARD * (1.0 + 0.01 * scale)
    vol2 = VOL + 0.01 * scale
    d_t = (1.0 / 365.0) * scale
    tau2 = TAU - d_t

    price_a = MODEL.price(FORWARD, STRIKE, TAU, VOL, RATE, CP.CALL)
    price_b = MODEL.price(forward2, STRIKE, tau2, vol2, RATE, CP.CALL)
    greeks_a = MODEL.greeks(FORWARD, STRIKE, TAU, VOL, RATE, CP.CALL)

    return attribute_pnl(
        greeks_a, price_a, price_b, forward2 - FORWARD, vol2 - VOL, d_t
    )


def test_small_proportional_move_closes_the_identity_tightly() -> None:
    result = _move(scale=0.1)
    assert abs(result.residual / result.actual_pnl) < 0.005


def test_larger_combined_move_has_a_bounded_not_zero_residual() -> None:
    """~1% is the expected size of the vanna/volga cross-terms this 5-term
    formula deliberately omits, not a sign anything is wrong.
    """
    result = _move(scale=1.0)
    assert 0.0 < abs(result.residual / result.actual_pnl) < 0.02


def test_theta_sign_convention_pure_time_roll() -> None:
    """theta = -dV/dtau in this codebase, matching the FD test harness's own
    sign flip, so Theta*elapsed_years with elapsed_years positive should
    closely match a pure time roll with nothing else moving.
    """
    d_t = 10.0 / 365.0
    price_a = MODEL.price(FORWARD, STRIKE, TAU, VOL, RATE, CP.CALL)
    price_b = MODEL.price(FORWARD, STRIKE, TAU - d_t, VOL, RATE, CP.CALL)
    greeks_a = MODEL.greeks(FORWARD, STRIKE, TAU, VOL, RATE, CP.CALL)

    result = attribute_pnl(greeks_a, price_a, price_b, 0.0, 0.0, d_t)

    assert result.theta_term == pytest.approx(result.actual_pnl, rel=0.02)
    assert result.delta_term == 0.0
    assert result.gamma_term == 0.0
    assert result.vega_term == 0.0


def test_residual_is_reported_not_absorbed_on_a_curved_case() -> None:
    result = _move(scale=1.0)
    assert result.residual != 0.0
    assert result.explained == pytest.approx(result.actual_pnl - result.residual)
    assert (
        result.delta_term + result.gamma_term + result.vega_term + result.theta_term
        == pytest.approx(result.explained)
    )


def test_vega_and_theta_divide_by_their_own_bump_period_metadata() -> None:
    """Defensive scaling: the formula must not hardcode vega_bump=1.0,
    theta_period=1.0 even though every model in this library happens to set
    both that way today.
    """
    greeks_a = Greeks(
        delta=0.0, gamma=0.0, vega=10.0, theta=20.0, rho=0.0, vanna=0.0, volga=0.0,
        unit=Unit.QUOTE, vega_bump=0.5, theta_period=2.0,
    )
    result = attribute_pnl(greeks_a, price_a=100.0, price_b=100.0, d_underlying=0.0, d_vol=1.0, d_t_years=4.0)

    assert result.vega_term == pytest.approx(10.0 * (1.0 / 0.5))
    assert result.theta_term == pytest.approx(20.0 * (4.0 / 2.0))
