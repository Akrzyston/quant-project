from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from voltk.marketdata.parse import Candle, CandleSeries
from voltk.models.base import CP
from voltk.models.black76 import Black76
from voltk.sticky_regime import (
    StickyRegimeError,
    VolObservation,
    observations_from_candles,
    regress_vol_on_spot,
)

EXPIRY = datetime(2026, 9, 20, 8, 0, tzinfo=UTC)
STRIKE = 70000.0
MODEL = Black76()


def _coin_price(cp: CP, forward: float, tau: float, vol: float) -> float:
    return MODEL.price(forward, STRIKE, tau, vol, 0.0, cp) / forward


def _candle(t: datetime, close: float) -> Candle:
    return Candle(timestamp=t, open=close, high=close, low=close, close=close, volume=1.0)


def _tau(t: datetime) -> float:
    return (EXPIRY - t).total_seconds() / (365.0 * 24 * 3600)


def test_observations_from_candles_recovers_the_forward_and_vol_used_to_construct_prices() -> None:
    true_vol = 0.55
    start = EXPIRY - timedelta(days=3)
    forwards = [68000.0, 68500.0, 69000.0]
    times = [start + timedelta(hours=i) for i in range(3)]

    calls, puts, perps = [], [], []
    for t, forward in zip(times, forwards):
        tau = _tau(t)
        calls.append(_candle(t, _coin_price(CP.CALL, forward, tau, true_vol)))
        puts.append(_candle(t, _coin_price(CP.PUT, forward, tau, true_vol)))
        perps.append(_candle(t, forward))

    observations = observations_from_candles(
        CandleSeries(instrument_name="X-C", candles=tuple(calls)),
        CandleSeries(instrument_name="X-P", candles=tuple(puts)),
        CandleSeries(instrument_name="X-PERPETUAL", candles=tuple(perps)),
        strike=STRIKE,
        expiry=EXPIRY,
    )

    assert len(observations) == 3
    for obs, forward in zip(observations, forwards):
        assert obs.forward == pytest.approx(forward, rel=1e-6)
        assert obs.implied_vol == pytest.approx(true_vol, abs=1e-6)


def test_observations_from_candles_drops_timestamps_missing_or_past_expiry() -> None:
    t0, t1 = EXPIRY - timedelta(days=1), EXPIRY - timedelta(days=1) + timedelta(hours=1)
    after_expiry = EXPIRY + timedelta(hours=1)
    forward, vol = 69000.0, 0.5
    tau0, tau1 = _tau(t0), _tau(t1)

    calls = [
        _candle(t0, _coin_price(CP.CALL, forward, tau0, vol)),
        _candle(t1, _coin_price(CP.CALL, forward, tau1, vol)),
        _candle(after_expiry, 0.01),
    ]
    puts = [_candle(t0, _coin_price(CP.PUT, forward, tau0, vol)), _candle(after_expiry, 0.01)]  # t1 missing
    perps = [_candle(t0, forward), _candle(t1, forward), _candle(after_expiry, forward)]

    observations = observations_from_candles(
        CandleSeries(instrument_name="X-C", candles=tuple(calls)),
        CandleSeries(instrument_name="X-P", candles=tuple(puts)),
        CandleSeries(instrument_name="X-PERPETUAL", candles=tuple(perps)),
        strike=STRIKE,
        expiry=EXPIRY,
    )

    assert len(observations) == 1
    assert observations[0].timestamp == t0


def test_regress_vol_on_spot_recovers_an_exact_known_slope() -> None:
    true_slope = -2e-6
    start = EXPIRY - timedelta(days=1)
    spots = [69000.0, 69100.0, 69300.0, 69200.0, 69500.0]
    observations = [
        VolObservation(timestamp=start + timedelta(hours=i), spot=s, forward=s, implied_vol=0.5 + true_slope * s)
        for i, s in enumerate(spots)
    ]

    result = regress_vol_on_spot(observations, strike=STRIKE, sticky_delta_prediction=-1.5e-6)

    assert result.slope == pytest.approx(true_slope, rel=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)
    assert result.sticky_strike_prediction == 0.0
    assert result.sticky_delta_prediction == -1.5e-6
    assert result.n_observations == 4


def test_regress_vol_on_spot_rejects_too_few_observations_or_no_spot_movement() -> None:
    too_few = [
        VolObservation(timestamp=EXPIRY - timedelta(hours=1), spot=69000.0, forward=69000.0, implied_vol=0.5),
        VolObservation(timestamp=EXPIRY, spot=69100.0, forward=69100.0, implied_vol=0.51),
    ]
    flat_spot = [
        VolObservation(timestamp=EXPIRY - timedelta(hours=i), spot=69000.0, forward=69000.0, implied_vol=0.5)
        for i in range(3, 0, -1)
    ]
    for obs in (too_few, flat_spot):
        with pytest.raises(StickyRegimeError):
            regress_vol_on_spot(obs, strike=STRIKE, sticky_delta_prediction=0.0)
