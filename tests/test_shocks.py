"""Structured shocks: level, skew, curvature -- three distinctly-shaped
transformations of the fitted SVI parameter space, not one parallel shift.

Each shock's defining property is exact, not approximate, at the slice's own
vertex k=m: w(m)=a+b*sigma, dw/dk(m)=b*rho, d2w/dk2(m)=b/sigma. A level shock
(bumps a) leaves both derivatives untouched. A skew shock (bumps rho) leaves
w(m) and the curvature untouched, changing only the slope. A curvature shock
(bumps sigma) leaves the slope untouched, changing the curvature.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from voltk.marketdata.parse import DvolPoint, DvolSeries
from voltk.shocks import ShockDefinition, ShockFactor, level_shock_magnitudes_from_dvol, shock_slice
from voltk.surface import SVISlice

EXPIRY = datetime(2026, 9, 19, 8, 0, tzinfo=UTC)
BASE = SVISlice(
    currency="XBT", expiry=EXPIRY, tau=30 / 365,
    a=0.05, b=0.10, rho=-0.3, m=0.1, sigma=0.2,
    sse=0.0, points_used=10, points_available=10, k_range=(-1.0, 1.0),
)


def test_level_shock_shifts_total_variance_uniformly() -> None:
    shocked = shock_slice(BASE, ShockDefinition(ShockFactor.LEVEL, 0.01, "level"))

    for k in (-1.0, -0.5, BASE.m, 0.5, 1.0):
        assert shocked.total_variance(k) - BASE.total_variance(k) == pytest.approx(0.01)
    assert shocked.dw_dk(BASE.m) == pytest.approx(BASE.dw_dk(BASE.m))
    assert shocked.d2w_dk2(BASE.m) == pytest.approx(BASE.d2w_dk2(BASE.m))


def test_skew_shock_changes_atm_slope_not_level_or_curvature() -> None:
    shocked = shock_slice(BASE, ShockDefinition(ShockFactor.SKEW, 0.1, "skew"))
    m = BASE.m

    assert shocked.total_variance(m) == pytest.approx(BASE.total_variance(m))
    assert shocked.d2w_dk2(m) == pytest.approx(BASE.d2w_dk2(m))
    assert shocked.dw_dk(m) != pytest.approx(BASE.dw_dk(m))


def test_curvature_shock_changes_atm_curvature_not_slope() -> None:
    shocked = shock_slice(BASE, ShockDefinition(ShockFactor.CURVATURE, 0.05, "curvature"))
    m = BASE.m

    assert shocked.dw_dk(m) == pytest.approx(BASE.dw_dk(m))
    assert shocked.d2w_dk2(m) != pytest.approx(BASE.d2w_dk2(m))


def test_shocks_are_not_a_parallel_vol_shift() -> None:
    """A parallel vol shift changes vol by the same amount at every k. None of
    the three shocks do -- and their shapes differ from each other too.
    """
    ks = (-1.0, -0.5, BASE.m, 0.5, 1.0)
    base_vols = [BASE.vol(k) for k in ks]

    shifts = {}
    for factor, magnitude in (
        (ShockFactor.LEVEL, 0.01), (ShockFactor.SKEW, 0.1), (ShockFactor.CURVATURE, 0.05),
    ):
        shocked = shock_slice(BASE, ShockDefinition(factor, magnitude, str(factor)))
        shifts[factor] = [shocked.vol(k) - v0 for k, v0 in zip(ks, base_vols)]

    for shift in shifts.values():
        assert len(set(round(s, 8) for s in shift)) > 1, "not a flat parallel shift"

    factors = list(shifts)
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            a, b = shifts[factors[i]], shifts[factors[j]]
            ratios = [x / y for x, y in zip(a, b) if abs(y) > 1e-9]
            assert len(set(round(r, 4) for r in ratios)) > 1, (
                f"{factors[i]} and {factors[j]} shift vol proportionally -- same shape"
            )


def test_shock_slice_keeps_variance_non_negative_at_extreme_magnitude() -> None:
    extreme = shock_slice(BASE, ShockDefinition(ShockFactor.LEVEL, -10.0, "crash"))
    for k in (-1.0, extreme.m, 1.0):
        assert extreme.total_variance(k) >= 0.0


def test_level_shock_magnitude_from_dvol_uses_realized_vol_of_vol() -> None:
    points = tuple(
        DvolPoint(timestamp=EXPIRY, open=c, high=c, low=c, close=c)
        for c in (0.20, 0.22, 0.19, 0.21, 0.23, 0.20)
    )
    series = DvolSeries(currency="XBT", points=points)

    magnitudes = level_shock_magnitudes_from_dvol(series, base_level=0.05)

    assert len(magnitudes) == 4
    assert magnitudes[0] < 0.0 < magnitudes[-1]
    # Multiples are (-2,-1,1,2), so each pair scales by exactly 2x.
    assert magnitudes[0] == pytest.approx(2.0 * magnitudes[1])
    assert magnitudes[-1] == pytest.approx(2.0 * magnitudes[-2])


def test_level_shock_magnitude_handles_missing_dvol() -> None:
    assert level_shock_magnitudes_from_dvol(None, base_level=0.05) == ()
    assert level_shock_magnitudes_from_dvol(
        DvolSeries(currency="XBT", points=()), base_level=0.05
    ) == ()
