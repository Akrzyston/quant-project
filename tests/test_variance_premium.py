from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from voltk.marketdata.parse import DvolPoint, RealizedVolPoint
from voltk.variance_premium import summarise_premium, variance_risk_premium

START = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)


def _dvol(hours: int, close: float) -> DvolPoint:
    t = START + timedelta(hours=hours)
    return DvolPoint(timestamp=t, open=close, high=close, low=close, close=close)


def _realized(hours: int, value: float) -> RealizedVolPoint:
    return RealizedVolPoint(timestamp=START + timedelta(hours=hours), value=value)


def test_variance_risk_premium_pairs_matching_timestamps() -> None:
    dvol = [_dvol(0, 0.60), _dvol(1, 0.62)]
    realized = [_realized(0, 0.50), _realized(1, 0.55)]

    points = variance_risk_premium(dvol, realized)

    assert len(points) == 2
    assert points[0].implied == 0.60
    assert points[0].realized == 0.50
    assert points[0].premium == pytest.approx(0.10)


def test_variance_risk_premium_matches_the_nearest_realized_point() -> None:
    dvol = [_dvol(0, 0.60)]
    realized = [_realized(-2, 0.40), _realized(0, 0.48)]

    points = variance_risk_premium(dvol, realized)

    assert points[0].realized == 0.48


def test_variance_risk_premium_drops_points_beyond_the_max_gap() -> None:
    dvol = [_dvol(0, 0.60)]
    realized = [_realized(5, 0.50)]  # 5 hours away

    points = variance_risk_premium(dvol, realized, max_gap_seconds=3600.0)

    assert points == ()


def test_variance_risk_premium_empty_inputs() -> None:
    assert variance_risk_premium([], []) == ()
    assert variance_risk_premium([_dvol(0, 0.6)], []) == ()


def test_summarise_premium_reports_mean_and_fraction_positive() -> None:
    points = variance_risk_premium(
        [_dvol(0, 0.60), _dvol(1, 0.50)],
        [_realized(0, 0.50), _realized(1, 0.55)],
    )

    summary = summarise_premium(points)

    assert summary.n == 2
    assert summary.mean_premium == pytest.approx((0.10 + (-0.05)) / 2)
    assert summary.fraction_positive == pytest.approx(0.5)
    assert summary.inversions == (points[1].timestamp,)


def test_summarise_premium_none_on_empty() -> None:
    assert summarise_premium(()) is None
