"""Payload parsing, shared by live capture and snapshot replay."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class Quote:
    bid: float | None
    ask: float | None
    mark: float | None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def quotes(response: RawResponse) -> dict[str, Quote]:
    """Instrument name to bid/ask/mark, skipping rows the venue has not named."""
    found: dict[str, Quote] = {}
    for row in book_summary(response):
        name = row.get("instrument_name")
        if not name:
            continue
        found[name] = Quote(
            bid=_optional_float(row.get("bid_price")),
            ask=_optional_float(row.get("ask_price")),
            mark=_optional_float(row.get("mark_price")),
        )
    return found


def index_names(instrument_list: Sequence[Instrument]) -> dict[str, str]:
    """Currency to price index name, taken from instrument metadata."""
    found: dict[str, str] = {}
    for inst in instrument_list:
        if inst.price_index and inst.base_currency not in found:
            found[inst.base_currency] = inst.price_index
    return found
