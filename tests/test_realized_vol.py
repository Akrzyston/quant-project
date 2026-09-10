"""Realized volatility estimators.

Grids stay small: a handful of hand-built candles is enough to check each
estimator's formula and its edge cases, not a long synthetic price path.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from voltk.marketdata.parse import Candle
from voltk.marketdata.parse import RealizedVolPoint
from voltk.realized_vol import (
    RealizedVolError,
    close_to_close,
    infer_periods_per_year,
    parkinson,
    reconcile,
    rolling_realized_vol,
)

START = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)


def _candles(closes: list[float], *, step: timedelta = timedelta(hours=1), highs=None, lows=None) -> list[Candle]:
    highs = highs or [c * 1.001 for c in closes]
    lows = lows or [c * 0.999 for c in closes]
    return [
        Candle(timestamp=START + i * step, open=c, high=h, low=l, close=c, volume=1.0)
        for i, (c, h, l) in enumerate(zip(closes, highs, lows))
    ]


def test_infer_periods_per_year_reads_actual_candle_spacing_not_a_requested_resolution() -> None:
    assert infer_periods_per_year(_candles([100.0, 101.0, 102.0])) == pytest.approx(365.0 * 24)
    quarter_hour = _candles([100.0, 101.0, 102.0], step=timedelta(minutes=15))
    assert infer_periods_per_year(quarter_hour) == pytest.approx(365.0 * 24 * 4)


def test_realized_vol_functions_reject_too_few_candles() -> None:
    one_candle = _candles([100.0])
    with pytest.raises(RealizedVolError):
        infer_periods_per_year(one_candle)
    with pytest.raises(RealizedVolError):
        close_to_close(one_candle, periods_per_year=365.0 * 24)
    with pytest.raises(RealizedVolError):
        parkinson([], periods_per_year=365.0 * 24)


def test_close_to_close_recovers_a_known_constant_step_vol() -> None:
    # A deterministic alternating log-return of +-r reproduces sample stdev = r exactly.
    r = 0.01
    closes = [100.0]
    for i in range(1, 21):
        step = r if i % 2 else -r
        closes.append(closes[-1] * math.exp(step))
    candles = _candles(closes)
    periods = infer_periods_per_year(candles)

    result = close_to_close(candles, periods_per_year=periods)

    assert result.n_observations == 20
    # mean return is exactly 0 (10 up-steps, 10 down-steps of equal size), so
    # the sample variance is sum(r^2)/(n-1) = n*r^2/(n-1), not r^2 itself.
    expected = r * math.sqrt(20.0 / 19.0) * math.sqrt(periods)
    assert result.value == pytest.approx(expected, rel=1e-9)


def test_parkinson_is_zero_for_a_flat_high_equals_low_series() -> None:
    candles = _candles([100.0, 101.0, 99.0], highs=[100.0, 101.0, 99.0], lows=[100.0, 101.0, 99.0])
    result = parkinson(candles, periods_per_year=365.0 * 24)
    assert result.value == pytest.approx(0.0, abs=1e-12)


def test_parkinson_is_positive_when_ranges_are_nonzero() -> None:
    candles = _candles([100.0, 101.0, 99.0])
    result = parkinson(candles, periods_per_year=365.0 * 24)
    assert result.value > 0.0
    assert result.n_observations == 3


def test_rolling_realized_vol_produces_one_point_per_step_in_range() -> None:
    candles = _candles([100.0] * 11)  # hours 0..10
    points = rolling_realized_vol(candles, window=timedelta(hours=3), step=timedelta(hours=1))
    assert len(points) == 8
    assert points[0].timestamp == START + timedelta(hours=3)
    assert points[-1].timestamp == START + timedelta(hours=10)


def test_rolling_realized_vol_single_window_matches_a_direct_close_to_close_call() -> None:
    candles = _candles([100.0, 101.0, 102.0, 101.0, 103.0])  # hours 0..4
    window = timedelta(hours=4)
    points = rolling_realized_vol(candles, window=window, step=window)

    assert len(points) == 1
    # window is (start, start+4h], excluding the candle at hour 0
    in_window = candles[1:]
    periods = infer_periods_per_year(in_window)
    expected = close_to_close(in_window, periods_per_year=periods).value
    assert points[0].value == pytest.approx(expected)
    assert points[0].timestamp == START + window


def test_rolling_realized_vol_skips_windows_with_too_few_candles() -> None:
    # A dense run (0-4h), an isolated candle (10h), then another dense run (20-23h).
    # close_to_close needs >=3 candles (2 returns) per window, so a 3h window
    # only succeeds where the underlying data is actually dense enough.
    hours = [0, 1, 2, 3, 4, 10, 20, 21, 22, 23]
    candles = [
        Candle(
            timestamp=START + timedelta(hours=h),
            open=100.0, high=100.1, low=99.9, close=100.0 + h * 0.1, volume=1.0,
        )
        for h in hours
    ]
    points = rolling_realized_vol(candles, window=timedelta(hours=3), step=timedelta(hours=1))
    timestamps = [p.timestamp for p in points]
    assert timestamps == [
        START + timedelta(hours=3), START + timedelta(hours=4),
        START + timedelta(hours=22), START + timedelta(hours=23),
    ]


def test_rolling_realized_vol_empty_input() -> None:
    assert rolling_realized_vol([], window=timedelta(hours=1), step=timedelta(hours=1)) == ()


def _rvp(hours: int, value: float) -> RealizedVolPoint:
    return RealizedVolPoint(timestamp=START + timedelta(hours=hours), value=value)


def test_reconcile_identical_series_has_zero_mean_diff() -> None:
    a = [_rvp(0, 0.60), _rvp(1, 0.62)]
    b = [_rvp(0, 0.60), _rvp(1, 0.62)]

    result = reconcile(a, b)

    assert result.n == 2
    assert result.mean_abs_diff == pytest.approx(0.0)


def test_reconcile_reports_a_constant_offset() -> None:
    a = [_rvp(0, 0.60), _rvp(1, 0.62)]
    b = [_rvp(0, 0.55), _rvp(1, 0.57)]

    result = reconcile(a, b)

    assert result.n == 2
    assert result.mean_abs_diff == pytest.approx(0.05)


def test_reconcile_drops_points_beyond_the_max_gap() -> None:
    a = [_rvp(0, 0.60)]
    b = [_rvp(5, 0.60)]  # 5 hours away

    assert reconcile(a, b, max_gap_seconds=3600.0) is None


def test_reconcile_empty_inputs_return_none() -> None:
    assert reconcile([], []) is None
    assert reconcile([_rvp(0, 0.6)], []) is None
