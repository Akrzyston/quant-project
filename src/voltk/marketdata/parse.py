"""Payload parsing, shared by live capture and snapshot replay."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from voltk.instruments import Instrument
from voltk.provenance import RawResponse


def instruments(response: RawResponse) -> list[Instrument]:
    return [Instrument.from_deribit(item, response.provenance) for item in response.result()]


def index_price(response: RawResponse) -> float:
    return float(response.result()["index_price"])


def book_summary(response: RawResponse) -> list[Mapping[str, Any]]:
    return list(response.result())


def mark_prices(response: RawResponse) -> dict[str, float]:
    """Instrument name to mark price, skipping rows the venue has not marked."""
    marks: dict[str, float] = {}
    for row in book_summary(response):
        name, mark = row.get("instrument_name"), row.get("mark_price")
        if name and mark is not None:
            marks[name] = float(mark)
    return marks


def index_names(instrument_list: Sequence[Instrument]) -> dict[str, str]:
    """Currency to price index name, taken from instrument metadata."""
    found: dict[str, str] = {}
    for inst in instrument_list:
        if inst.price_index and inst.base_currency not in found:
            found[inst.base_currency] = inst.price_index
    return found
