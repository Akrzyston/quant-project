"""Model-free variance index: discretization accuracy, forward sensitivity,
constant-maturity interpolation, and DVOL comparison."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from tests.synthetic_chain import CURRENCY, build_synthetic_chain
from voltk.universe import Universe
from voltk.variance import (
    ConstantMaturityVariance,
    VarianceError,
    VarianceSlice,
    compare_to_dvol,
    constant_maturity_variance,
    model_free_variance,
    variance_term_structure,
)

FLAT_SVI = dict(a=0.09, b=0.0, rho=0.0, m=0.0, sigma=0.2)
TAU_30D = 30.0 / 365.0


def test_model_free_variance_recovers_flat_vol_within_tolerance() -> None:
    universe, marks, implied_forward = build_synthetic_chain(
        svi=FLAT_SVI, tau=TAU_30D, n_strikes=41
    )
    known_variance = FLAT_SVI["a"] / TAU_30D

    result = model_free_variance(universe, marks, implied_forward)

    assert result is not None
    assert result.variance == pytest.approx(known_variance, rel=0.02)


def test_variance_uses_the_implied_forward_not_forward_for() -> None:
    universe, marks, implied_forward = build_synthetic_chain(
        svi=FLAT_SVI, tau=TAU_30D, n_strikes=41
    )
    known_variance = FLAT_SVI["a"] / TAU_30D

    correct = model_free_variance(universe, marks, implied_forward)
    wrong_forward = dataclasses.replace(implied_forward, forward=implied_forward.forward * 0.85)
    wrong = model_free_variance(universe, marks, wrong_forward)

    assert correct is not None and wrong is not None
    assert abs(correct.variance - known_variance) < abs(wrong.variance - known_variance)


def test_variance_term_structure_skips_expiries_with_too_few_strikes() -> None:
    good_universe, good_marks, good_forward = build_synthetic_chain(tau=TAU_30D)
    thin_universe, thin_marks, thin_forward = build_synthetic_chain(tau=60.0 / 365.0)
    # Strip all but two strikes' marks so the thin expiry can't clear
    # MIN_STRIKES_FOR_VARIANCE.
    keep = {n for i, n in enumerate(thin_marks) if i < 2}
    thin_marks = {n: p for n, p in thin_marks.items() if n in keep}

    combined = Universe(
        spec=good_universe.spec,
        as_of=good_universe.as_of,
        instruments=good_universe.instruments + thin_universe.instruments,
        index_prices=good_universe.index_prices,
        forward_curve=(),
    )
    combined_marks = good_marks | thin_marks

    slices = variance_term_structure(combined, combined_marks, [good_forward, thin_forward])

    assert len(slices) == 1
    assert slices[0].expiry == good_forward.expiry


def _slice(tau: float, variance: float) -> VarianceSlice:
    return VarianceSlice(
        currency=CURRENCY,
        expiry=datetime(2026, 9, 19, 8, 0, tzinfo=UTC),
        tau=tau,
        forward=60_000.0,
        rate=0.0,
        k0=60_000.0,
        k0_extrapolated=False,
        variance=variance,
        strikes_used=10,
        contributions=(),
    )


def test_constant_maturity_variance_interpolates_between_two_slices() -> None:
    near, far = _slice(20 / 365, 0.80), _slice(50 / 365, 1.00)
    target_tau = 30 / 365

    result = constant_maturity_variance([near, far], target_tau, currency=CURRENCY)

    expected = (
        near.tau * near.variance * (far.tau - target_tau)
        + far.tau * far.variance * (target_tau - near.tau)
    ) / (far.tau - near.tau) / target_tau

    assert result is not None
    assert result.variance == pytest.approx(expected, rel=1e-9)
    assert result.bracketed and not result.extrapolated and not result.single_slice


def test_constant_maturity_variance_extrapolates_past_the_far_end_with_a_flag() -> None:
    near, far = _slice(10 / 365, 0.80), _slice(20 / 365, 0.90)

    result = constant_maturity_variance([near, far], 60 / 365, currency=CURRENCY)

    assert result is not None
    assert result.extrapolated and not result.bracketed


def test_constant_maturity_variance_returns_the_single_slice_unflagged_when_only_one_exists() -> None:
    only = _slice(30 / 365, 0.90)

    result = constant_maturity_variance([only], 45 / 365, currency=CURRENCY)

    assert result is not None
    assert result.single_slice
    assert result.variance == only.variance


def test_constant_maturity_variance_returns_none_with_zero_usable_slices() -> None:
    assert constant_maturity_variance([], 30 / 365, currency=CURRENCY) is None

    invalid = _slice(30 / 365, -0.5)
    assert constant_maturity_variance([invalid], 30 / 365, currency=CURRENCY) is None


def test_constant_maturity_variance_rejects_non_positive_target_tau() -> None:
    with pytest.raises(VarianceError):
        constant_maturity_variance([_slice(30 / 365, 0.9)], 0.0, currency=CURRENCY)


def test_compare_to_dvol_leaves_gaps_none_when_inputs_are_missing() -> None:
    cmv = ConstantMaturityVariance(
        currency=CURRENCY, target_tau=30 / 365, variance=0.09, vol=0.3,
        bracketed=True, extrapolated=False, single_slice=False, left=None, right=None,
    )

    both_missing = compare_to_dvol(None, None, None, currency=CURRENCY)
    assert both_missing.gap_vs_dvol is None and both_missing.gap_vs_surface is None

    dvol_only = compare_to_dvol(cmv, 0.28, None, currency=CURRENCY)
    assert dvol_only.gap_vs_dvol == pytest.approx(0.02)
    assert dvol_only.gap_vs_surface is None


def test_compare_to_dvol_computes_both_gaps_when_available() -> None:
    cmv = ConstantMaturityVariance(
        currency=CURRENCY, target_tau=30 / 365, variance=0.09, vol=0.3,
        bracketed=True, extrapolated=False, single_slice=False, left=None, right=None,
    )

    result = compare_to_dvol(cmv, 0.30, 0.33, currency=CURRENCY)

    assert result.gap_vs_dvol == pytest.approx(0.0, abs=1e-12)
    assert result.gap_vs_surface == pytest.approx(-0.03)
    assert result.gap_vs_surface_bps == pytest.approx(-0.03 / 0.33 * 10_000)
