"""DVOL candle parsing."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from voltk.marketdata import parse
from voltk.provenance import Provenance, RawResponse

AS_OF = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def _response(data: list) -> RawResponse:
    body = json.dumps({"result": {"data": data, "continuation": None}}).encode()
    return RawResponse(
        provenance=Provenance(source="test", endpoint="public/get_volatility_index_data",
                               retrieved_at=AS_OF),
        body=body,
    )


def test_dvol_series_parses_ohlc_rows_converting_percentage_to_decimal() -> None:
    ts = int(AS_OF.timestamp() * 1000)
    response = _response([[ts, 20.0, 22.0, 19.0, 21.0]])

    series = parse.dvol_series(response, currency="BTC")

    assert len(series.points) == 1
    point = series.points[0]
    assert point.timestamp == datetime.fromtimestamp(ts / 1000, tz=UTC)
    assert (point.open, point.high, point.low, point.close) == (0.20, 0.22, 0.19, 0.21)
    assert series.latest is point


def test_dvol_series_wire_values_are_a_percentage_not_a_decimal_fraction() -> None:
    # The live endpoint returns values in the 20s-50s for both BTC and ETH,
    # not 0.2-0.5 -- a raw wire value this large would be an absurd
    # annualized vol if it were already a decimal fraction.
    ts = int(AS_OF.timestamp() * 1000)
    response = _response([[ts, 37.95, 38.14, 37.84, 37.95]])

    series = parse.dvol_series(response, currency="BTC")

    assert series.latest.close == pytest.approx(0.3795)
    assert series.latest.close < 1.0


def test_dvol_series_skips_malformed_rows() -> None:
    ts = int(AS_OF.timestamp() * 1000)
    response = _response([[ts, 20.0, 20.0, 20.0], [ts, 20.0, 22.0, 19.0, 21.0]])

    series = parse.dvol_series(response, currency="BTC")

    assert len(series.points) == 1


def test_dvol_series_sorts_by_timestamp() -> None:
    early = int(AS_OF.timestamp() * 1000)
    late = early + 3600_000
    response = _response(
        [[late, 20.0, 20.0, 20.0, 22.0], [early, 20.0, 20.0, 20.0, 20.0]]
    )

    series = parse.dvol_series(response, currency="BTC")

    assert [p.close for p in series.points] == [0.20, 0.22]


def test_dvol_series_empty_data_list() -> None:
    response = _response([])

    series = parse.dvol_series(response, currency="BTC")

    assert series.points == ()
    assert series.latest is None
