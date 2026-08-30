"""Historical realized-vol and candle parsing (get_historical_volatility,
get_tradingview_chart_data)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from voltk.marketdata import parse
from voltk.provenance import Provenance, RawResponse

AS_OF = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def _response(result: object, endpoint: str) -> RawResponse:
    body = json.dumps({"result": result}).encode()
    return RawResponse(
        provenance=Provenance(source="test", endpoint=endpoint, retrieved_at=AS_OF), body=body
    )


def test_historical_volatility_series_converts_percentage_to_decimal() -> None:
    ts = int(AS_OF.timestamp() * 1000)
    response = _response([[ts, 22.3]], "public/get_historical_volatility")

    series = parse.historical_volatility_series(response, currency="XBT")

    assert len(series.points) == 1
    assert series.points[0].value == 22.3 / 100.0
    assert series.points[0].timestamp == datetime.fromtimestamp(ts / 1000, tz=UTC)


def test_historical_volatility_series_skips_malformed_rows() -> None:
    ts = int(AS_OF.timestamp() * 1000)
    response = _response([[ts], [ts, 20.0]], "public/get_historical_volatility")

    series = parse.historical_volatility_series(response, currency="XBT")

    assert len(series.points) == 1


def test_historical_volatility_series_sorts_by_timestamp() -> None:
    early, late = int(AS_OF.timestamp() * 1000), int(AS_OF.timestamp() * 1000) + 3600_000
    response = _response([[late, 21.0], [early, 20.0]], "public/get_historical_volatility")

    series = parse.historical_volatility_series(response, currency="XBT")

    assert [p.value for p in series.points] == [0.20, 0.21]


def test_candle_series_parses_parallel_arrays() -> None:
    ts = int(AS_OF.timestamp() * 1000)
    result = {
        "status": "ok",
        "ticks": [ts],
        "open": [100.0],
        "high": [105.0],
        "low": [95.0],
        "close": [102.0],
        "volume": [10.0],
    }
    response = _response(result, "public/get_tradingview_chart_data")

    series = parse.candle_series(response, instrument_name="XBT-PERPETUAL")

    assert len(series.candles) == 1
    candle = series.candles[0]
    assert candle.timestamp == datetime.fromtimestamp(ts / 1000, tz=UTC)
    assert (candle.open, candle.high, candle.low, candle.close, candle.volume) == (100.0, 105.0, 95.0, 102.0, 10.0)


def test_candle_series_no_data_status_is_an_empty_series_not_an_error() -> None:
    result = {"status": "no_data", "ticks": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    response = _response(result, "public/get_tradingview_chart_data")

    series = parse.candle_series(response, instrument_name="XBT-1JAN27-1000-C")

    assert series.candles == ()
