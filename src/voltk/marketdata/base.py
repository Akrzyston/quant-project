"""Market data source contract.

Sources return raw response bodies. Parsing happens above this layer so a live
fetch and a snapshot replay run through identical parsing code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from voltk.instruments import Kind
from voltk.provenance import RawResponse


class MarketDataError(RuntimeError):
    """A market data call failed. Message is fit to show a user."""


@runtime_checkable
class MarketDataSource(Protocol):
    @property
    def label(self) -> str: ...

    @property
    def is_live(self) -> bool: ...

    def now(self) -> datetime:
        """Observation time: wall clock when live, capture time when replaying."""

    def fetch_instruments(self, currency: str, kind: Kind) -> RawResponse: ...

    def fetch_index(self, index_name: str) -> RawResponse: ...

    def fetch_book_summary(self, currency: str, kind: Kind) -> RawResponse: ...
