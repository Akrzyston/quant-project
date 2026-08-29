"""Binomial tree: convergence, early exercise, and greeks.

The tree is checked against the closed forms it should reproduce, and against
itself for the early-exercise premium. Step counts are kept low enough that the
suite stays fast; the convergence test is the one that uses a large tree.
"""

from __future__ import annotations

import math

import pytest

from voltk.models.base import CP
from voltk.models.binomial import Binomial, Dividend
from voltk.models.black_scholes import BlackScholes

RATE = 0.05
SPOT = 100.0
VOL = 0.25
TAU = 1.0


@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
@pytest.mark.parametrize("cp", list(CP))
def test_european_tree_converges_to_black_scholes(strike, cp) -> None:
    reference = BlackScholes().price(SPOT, strike, TAU, VOL, RATE, cp)
    errors = [
        abs(Binomial(n, american=False).price(SPOT, strike, TAU, VOL, RATE, cp) - reference)
        for n in (100, 400, 1600)
    ]
    assert errors == sorted(errors, reverse=True), "error must fall as steps rise"
    # Convergence is first order, so a sixteen-fold refinement buys about 16x.
    assert errors[0] / errors[2] > 8.0
    assert errors[2] < 2e-3


def test_american_call_without_dividends_is_never_exercised_early() -> None:
    """The classic check: no dividend means no reason to exercise a call early."""
    tree = Binomial(400)
    for strike in (80.0, 100.0, 125.0):
        assert tree.early_exercise_premium(SPOT, strike, TAU, VOL, RATE, CP.CALL) == 0.0


def test_american_put_carries_a_positive_early_exercise_premium() -> None:
    tree = Binomial(400)
    premium = tree.early_exercise_premium(SPOT, 100.0, TAU, VOL, RATE, CP.PUT)
    assert premium > 0.1
    assert tree.price(SPOT, 100.0, TAU, VOL, RATE, CP.PUT) > BlackScholes().price(
        SPOT, 100.0, TAU, VOL, RATE, CP.PUT
    )


def test_deep_in_the_money_american_put_is_worth_intrinsic() -> None:
    """Far enough in the money, immediate exercise dominates."""
    tree = Binomial(300)
    price = tree.price(SPOT, 400.0, TAU, VOL, RATE, CP.PUT)
    assert price == pytest.approx(400.0 - SPOT, rel=1e-3)


def test_dividends_make_early_call_exercise_worthwhile() -> None:
    dividends = (Dividend(0.25, 3.0), Dividend(0.75, 3.0))
    tree = Binomial(400, dividends=dividends)
    assert tree.early_exercise_premium(SPOT, 100.0, TAU, VOL, RATE, CP.CALL) > 0.0
    # A dividend-paying underlying is worth less to a call holder.
    assert tree.price(SPOT, 100.0, TAU, VOL, RATE, CP.CALL) < Binomial(400).price(
        SPOT, 100.0, TAU, VOL, RATE, CP.CALL
    )


def test_escrow_discounts_only_dividends_still_to_come() -> None:
    tree = Binomial(50, dividends=(Dividend(0.5, 4.0),))
    assert tree.escrow(TAU, RATE) == pytest.approx(4.0 * math.exp(-RATE * 0.5), rel=1e-12)
    assert tree.escrow(TAU, RATE, after=0.75) == 0.0


def test_dividends_exceeding_spot_are_refused() -> None:
    tree = Binomial(20, dividends=(Dividend(0.5, 200.0),))
    with pytest.raises(ValueError, match="exceed spot"):
        tree.price(SPOT, 100.0, TAU, VOL, RATE, CP.CALL)


@pytest.mark.parametrize("cp", list(CP))
def test_european_tree_greeks_track_black_scholes(cp) -> None:
    """Tree greeks carry discretisation error, so the tolerance is loose."""
    tree = Binomial(800, american=False)
    analytic = BlackScholes().greeks(SPOT, 100.0, TAU, VOL, RATE, cp)
    numeric = tree.greeks(SPOT, 100.0, TAU, VOL, RATE, cp)
    assert numeric.delta == pytest.approx(analytic.delta, abs=2e-3)
    assert numeric.gamma == pytest.approx(analytic.gamma, abs=2e-3)
    assert numeric.vega == pytest.approx(analytic.vega, rel=1e-2)
    assert numeric.theta == pytest.approx(analytic.theta, rel=5e-2)
    assert numeric.rho == pytest.approx(analytic.rho, rel=1e-2)


def test_american_bounds_include_immediate_exercise() -> None:
    tree = Binomial(50)
    band = tree.bounds(SPOT, 150.0, TAU, RATE, CP.PUT)
    assert band.lower >= 50.0


@pytest.mark.parametrize("cp", list(CP))
def test_round_trip_through_implied_vol(cp) -> None:
    tree = Binomial(120)
    price = tree.price(SPOT, 110.0, TAU, VOL, RATE, cp)
    recovered = tree.implied_vol(price, SPOT, 110.0, TAU, RATE, cp)
    assert abs(tree.price(SPOT, 110.0, TAU, recovered, RATE, cp) - price) < 1e-10


def test_a_tree_needs_at_least_two_steps() -> None:
    with pytest.raises(ValueError, match="at least two steps"):
        Binomial(1)
