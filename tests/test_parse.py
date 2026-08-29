"""Book-summary quote parsing."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from voltk.marketdata import parse
from voltk.provenance import Provenance, RawResponse

AS_OF = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def _response(rows: list[dict]) -> RawResponse:
    body = json.dumps({"result": rows}).encode()
    return RawResponse(
        provenance=Provenance(source="test", endpoint="public/get_book_summary_by_currency",
                               retrieved_at=AS_OF),
        body=body,
    )


def test_quotes_extracts_bid_ask_mark() -> None:
    response = _response(
        [{"instrument_name": "XBT-1JAN27-60000-C", "bid_price": 0.04,
          "ask_price": 0.05, "mark_price": 0.045}]
    )
    found = parse.quotes(response)
    quote = found["XBT-1JAN27-60000-C"]
    assert quote.bid == 0.04
    assert quote.ask == 0.05
    assert quote.mark == 0.045


def test_quotes_skips_rows_missing_instrument_name() -> None:
    response = _response([{"bid_price": 0.04, "ask_price": 0.05}])
    assert parse.quotes(response) == {}


def test_quotes_tolerates_missing_bid_or_ask() -> None:
    response = _response(
        [{"instrument_name": "XBT-1JAN27-60000-C", "bid_price": 0.04, "mark_price": 0.045}]
    )
    quote = parse.quotes(response)["XBT-1JAN27-60000-C"]
    assert quote.bid == 0.04
    assert quote.ask is None
    assert quote.mark == 0.045
