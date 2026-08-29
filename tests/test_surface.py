"""SVI calibration, the strike/time surface, and its no-arbitrage checks."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from tests.synthetic_chain import build_synthetic_chain
from voltk.surface import (
    SVISlice,
    Surface,
    SurfaceError,
    butterfly_check,
    calendar_check,
    calibrate_surface,
    calibrate_svi_slice,
    log_moneyness,
    smile_points,
)

EXPIRY = datetime(2026, 9, 19, 8, 0, tzinfo=UTC)
TAU_30D = 30.0 / 365.0


def _slice(**overrides) -> SVISlice:
    fields = dict(
        currency="XBT", expiry=EXPIRY, tau=TAU_30D,
        a=0.05, b=0.05, rho=0.0, m=0.0, sigma=0.2,
        sse=0.0, points_used=10, points_available=10, k_range=(-1.0, 1.0),
    )
    fields.update(overrides)
    return SVISlice(**fields)


def test_log_moneyness_zero_at_the_forward() -> None:
    assert log_moneyness(60_000.0, 60_000.0) == pytest.approx(0.0)


def test_calibration_round_trips_a_known_svi_curve() -> None:
    universe, marks, implied_forward = build_synthetic_chain()
    tau = (implied_forward.expiry - universe.as_of).total_seconds() / (365 * 24 * 3600)

    points = smile_points(universe, marks, implied_forward)
    fit = calibrate_svi_slice(points, currency="XBT", expiry=implied_forward.expiry, tau=tau)

    for p in points:
        if not p.identified:
            continue
        assert fit.total_variance(p.k) == pytest.approx(p.total_variance, abs=1e-6)


def test_calibration_raises_with_too_few_points() -> None:
    universe, marks, implied_forward = build_synthetic_chain(n_strikes=15)
    points = smile_points(universe, marks, implied_forward)[:2]

    with pytest.raises(SurfaceError):
        calibrate_svi_slice(points, currency="XBT", expiry=implied_forward.expiry, tau=TAU_30D)


def test_smile_points_skip_unpriceable_strikes_without_crashing() -> None:
    universe, marks, implied_forward = build_synthetic_chain()
    # Push one mark below intrinsic -- unpriceable at any vol.
    victim = next(iter(marks))
    marks[victim] = -1.0

    points = smile_points(universe, marks, implied_forward)

    assert 0 < len(points) < len(marks)


def test_butterfly_check_flags_a_deliberately_bad_slice() -> None:
    # Gatheral's own canonical shape: rho near -1 with sigma tiny relative to
    # b passes all four raw-SVI parameter constraints yet still
    # butterfly-arbitrages near the kink.
    bad = _slice(a=0.001, b=1.0, rho=-0.98, m=0.0, sigma=0.005, k_range=(-0.05, 0.05))

    report = butterfly_check(bad)

    assert not report.clean
    assert report.violations


def test_butterfly_check_passes_a_well_behaved_slice() -> None:
    good = _slice(a=0.02, b=0.10, rho=-0.3, m=0.0, sigma=0.2, k_range=(-1.3, 1.3))

    assert butterfly_check(good).clean


def test_calendar_check_flags_a_deliberately_decreasing_total_variance_pair() -> None:
    near = _slice(tau=10 / 365, a=0.05, b=0.05)
    far = _slice(tau=20 / 365, a=0.01, b=0.05)

    report = calendar_check(near, far)

    assert not report.clean
    zero_k_node = next(n for n in report.nodes if abs(n.k) < 1e-9)
    assert zero_k_node.violated


def test_calendar_check_passes_a_genuinely_increasing_pair() -> None:
    near = _slice(tau=10 / 365, a=0.01, b=0.05)
    far = _slice(tau=20 / 365, a=0.05, b=0.05)

    assert calendar_check(near, far).clean


def test_calendar_check_raises_when_expiries_are_out_of_order() -> None:
    near = _slice(tau=10 / 365)
    far = _slice(tau=20 / 365)

    with pytest.raises(SurfaceError):
        calendar_check(far, near)


def test_surface_total_variance_matches_each_slice_at_its_own_tau() -> None:
    near, far = _slice(tau=10 / 365, a=0.05), _slice(tau=20 / 365, a=0.08)
    surface = Surface(slices=(near, far))

    assert surface.total_variance(0.3, near.tau) == pytest.approx(near.total_variance(0.3))
    assert surface.total_variance(0.3, far.tau) == pytest.approx(far.total_variance(0.3))


def test_surface_extrapolates_by_holding_the_variance_rate_constant() -> None:
    near, far = _slice(tau=10 / 365, a=0.05), _slice(tau=20 / 365, a=0.08)
    surface = Surface(slices=(near, far))

    before = surface.total_variance(0.0, 5 / 365)
    assert before == pytest.approx(near.total_variance(0.0) * (5 / 365) / near.tau)

    after = surface.total_variance(0.0, 40 / 365)
    assert after == pytest.approx(far.total_variance(0.0) * (40 / 365) / far.tau)


def test_dvol_dforward_sticky_delta_is_nonzero_on_a_sloped_smile_and_zero_on_a_flat_one() -> None:
    sloped = Surface(slices=(_slice(b=0.10, rho=-0.5),))
    flat = Surface(slices=(_slice(b=0.0, rho=0.0),))

    assert sloped.dvol_dforward_sticky_delta(60_000.0, 60_000.0, TAU_30D) != pytest.approx(0.0, abs=1e-12)
    assert flat.dvol_dforward_sticky_delta(60_000.0, 60_000.0, TAU_30D) == pytest.approx(0.0, abs=1e-12)


def test_dvol_dspot_sticky_delta_scales_by_exp_rate_tau() -> None:
    sloped = Surface(slices=(_slice(b=0.10, rho=-0.5),))
    rate = 0.08

    d_forward = sloped.dvol_dforward_sticky_delta(60_000.0, 60_000.0, TAU_30D)
    d_spot = sloped.dvol_dspot_sticky_delta(60_000.0, 60_000.0, TAU_30D, rate)

    assert d_spot == pytest.approx(d_forward * math.exp(rate * TAU_30D))


def test_dvol_sticky_strike_is_always_exactly_zero() -> None:
    sloped = Surface(slices=(_slice(b=0.10, rho=-0.5),))

    assert sloped.dvol_sticky_strike(60_000.0, 60_000.0, TAU_30D, 0.08) == 0.0


def test_calibrate_surface_is_bit_for_bit_deterministic() -> None:
    """calibrate_svi_slice's seed schedule is a pure function of the input
    data (no RNG, no threading), so re-running the fitter against identical
    inputs must reproduce identical output exactly -- this is the surface-
    fitting half of M5's "reload a snapshot, re-run the fitter, and assert
    the surface reproduces bit-for-bit" accept criterion. The other half
    (reload reproduces an identical universe/marks) is already covered by
    tests/test_snapshots.py and tests/test_replay_determinism.py; composing
    the two proves the whole chain.
    """
    universe, marks, implied_forward = build_synthetic_chain()

    slices_1, failures_1 = calibrate_surface(universe, marks, [implied_forward])
    slices_2, failures_2 = calibrate_surface(universe, marks, [implied_forward])

    assert slices_1 == slices_2
    assert failures_1 == failures_2
