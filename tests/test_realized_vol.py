"""Realized volatility estimators.

Grids stay small: a handful of hand-built candles is enough to check each
estimator's formula and its edge cases, not a long synthetic price path.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from voltk.marketdata.parse import Candle
from voltk.realized_vol import RealizedVolError, close_to_close, infer_periods_per_year, parkinson

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
