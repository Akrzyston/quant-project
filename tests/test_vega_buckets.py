"""Vega bucketed by surface control point.

Reconciliation tolerance (1e-3 relative) is documented, not just loosened:
empirically ~8e-5 on the synthetic fixture, so 1e-3 leaves comfortable
margin for real chains with less regular strike spacing.
"""

from __future__ import annotations

import dataclasses

import pytest

from tests.synthetic_chain import build_synthetic_chain
from voltk.models.black76 import Black76
from voltk.models.base import CP
from voltk.surface import calibrate_svi_slice, smile_points
from voltk.vega_buckets import VegaBucketError, bucket_vega

RECONCILIATION_TOLERANCE = 1e-3


def _fixture():
    universe, marks, implied_forward = build_synthetic_chain()
    points = smile_points(universe, marks, implied_forward)
    tau = (implied_forward.expiry - universe.as_of).total_seconds() / (365 * 24 * 3600)
    base = calibrate_svi_slice(points, currency="XBT", expiry=implied_forward.expiry, tau=tau)
    return universe, points, implied_forward, tau, base


def test_bucket_definitions_partition_all_identified_points() -> None:
    _, points, implied_forward, tau, base = _fixture()
    forward = implied_forward.forward

    result = bucket_vega(
        points, base, Black76(), forward, forward, tau, 0.0, CP.CALL,
        currency="XBT", expiry=implied_forward.expiry,
    )

    identified = [p for p in points if p.identified]
    assert sum(b.points_used for b in result.buckets) == len(identified)


def test_level_bump_is_absorbed_by_a_alone() -> None:
    _, points, implied_forward, tau, base = _fixture()
    identified = [p for p in points if p.identified]
    dw = 1e-5

    bumped = tuple(dataclasses.replace(p, total_variance=p.total_variance + dw) for p in identified)
    refit = calibrate_svi_slice(bumped, currency="XBT", expiry=implied_forward.expiry, tau=tau)

    assert refit.a - base.a == pytest.approx(dw, rel=1e-6)
    assert refit.b == pytest.approx(base.b, abs=1e-9)
    assert refit.rho == pytest.approx(base.rho, abs=1e-9)
    assert refit.m == pytest.approx(base.m, abs=1e-9)
    assert refit.sigma == pytest.approx(base.sigma, abs=1e-9)


def test_bucketed_vegas_sum_to_parallel_vega_within_tolerance() -> None:
    _, points, implied_forward, tau, base = _fixture()
    forward = implied_forward.forward

    result = bucket_vega(
        points, base, Black76(), forward, forward, tau, 0.0, CP.CALL,
        currency="XBT", expiry=implied_forward.expiry,
    )

    assert result.reconciliation_error < RECONCILIATION_TOLERANCE


def test_bucket_vega_raises_with_too_few_identified_points() -> None:
    _, points, implied_forward, tau, base = _fixture()
    forward = implied_forward.forward
    thin_points = tuple(p for p in points if p.identified)[:2]

    with pytest.raises(VegaBucketError):
        bucket_vega(
            thin_points, base, Black76(), forward, forward, tau, 0.0, CP.CALL,
            currency="XBT", expiry=implied_forward.expiry,
        )
