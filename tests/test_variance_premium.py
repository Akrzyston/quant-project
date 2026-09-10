from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from voltk.marketdata.parse import DvolPoint, RealizedVolPoint
from voltk.variance_premium import (
    PremiumPoint,
    cumulative_captured_premium,
    favorable_windows,
    max_drawdown,
    summarise_premium,
    variance_risk_premium,
)

START = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)


def _dvol(hours: int, close: float) -> DvolPoint:
    t = START + timedelta(hours=hours)
    return DvolPoint(timestamp=t, open=close, high=close, low=close, close=close)


def _realized(hours: int, value: float) -> RealizedVolPoint:
    return RealizedVolPoint(timestamp=START + timedelta(hours=hours), value=value)


def test_variance_risk_premium_pairs_the_nearest_realized_point() -> None:
    dvol = [_dvol(0, 0.60), _dvol(1, 0.62)]
    realized = [_realized(-2, 0.40), _realized(0, 0.50), _realized(1, 0.55)]

    points = variance_risk_premium(dvol, realized)

    assert len(points) == 2
    assert points[0].implied == 0.60
    assert points[0].realized == 0.50
    assert points[0].premium == pytest.approx(0.10)


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


def _premium(hours: int, value: float) -> PremiumPoint:
    return PremiumPoint(timestamp=START + timedelta(hours=hours), implied=0.0, realized=0.0, premium=value)


def test_favorable_windows_finds_one_run_surrounded_by_negative_points() -> None:
    points = [_premium(0, -0.01), _premium(1, 0.02), _premium(2, 0.03), _premium(3, -0.01)]

    windows = favorable_windows(points)

    assert len(windows) == 1
    assert windows[0].start == START + timedelta(hours=1)
    assert windows[0].end == START + timedelta(hours=2)
    assert windows[0].n == 2
    assert windows[0].mean_premium == pytest.approx(0.025)


def test_favorable_windows_all_negative_returns_nothing() -> None:
    points = [_premium(0, -0.01), _premium(1, -0.02)]
    assert favorable_windows(points) == ()


def test_favorable_windows_all_positive_spans_the_whole_series() -> None:
    points = [_premium(0, 0.01), _premium(1, 0.02), _premium(2, 0.03)]

    windows = favorable_windows(points)

    assert len(windows) == 1
    assert windows[0].start == START
    assert windows[0].end == START + timedelta(hours=2)
    assert windows[0].n == 3


def test_favorable_windows_separates_non_adjacent_positive_runs() -> None:
    # a negative point between two positive ones must not merge them into one run
    points = [_premium(0, 0.01), _premium(1, -0.01), _premium(2, 0.02)]

    windows = favorable_windows(points)

    assert len(windows) == 2
    assert windows[0].n == 1
    assert windows[1].n == 1


def test_cumulative_captured_premium_sums_only_favorable_days() -> None:
    points = [_premium(0, 0.01), _premium(1, -0.02), _premium(2, 0.03)]

    cumulative = cumulative_captured_premium(points)

    assert cumulative == pytest.approx((0.01, 0.01, 0.04))


def test_cumulative_captured_premium_matches_points_length() -> None:
    points = [_premium(h, 0.01) for h in range(5)]
    assert len(cumulative_captured_premium(points)) == len(points)


def test_cumulative_captured_premium_all_negative_stays_at_zero() -> None:
    points = [_premium(0, -0.01), _premium(1, -0.02)]
    assert cumulative_captured_premium(points) == (0.0, 0.0)


def test_cumulative_captured_premium_empty_input() -> None:
    assert cumulative_captured_premium(()) == ()


def test_max_drawdown_finds_the_largest_peak_to_trough_decline() -> None:
    # rises to 10, falls to 4 (drawdown -6), rises to 12, falls to 9 (drawdown -3)
    cumulative = [0.0, 5.0, 10.0, 7.0, 4.0, 8.0, 12.0, 9.0]
    assert max_drawdown(cumulative) == pytest.approx(-6.0)


def test_max_drawdown_is_zero_for_a_monotonically_rising_series() -> None:
    assert max_drawdown([0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_max_drawdown_empty_input() -> None:
    assert max_drawdown(()) == 0.0
