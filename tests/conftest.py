"""A scripted venue for tests.

Payloads are built rather than recorded so a test can move the index between
calls and prove the drift gate fires.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from voltk.instruments import Kind
from voltk.provenance import Provenance, RawResponse

BASE = "XBT"
ALT = "XET"
AS_OF = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def _expiries(count: int = 3) -> list[datetime]:
    return [
        (AS_OF + timedelta(days=30 * (n + 1))).replace(hour=8, minute=0, second=0, microsecond=0)
        for n in range(count)
    ]


def option_payload(currency: str, expiry: datetime, strike: float, call: bool) -> dict[str, Any]:
    return {
        "instrument_name": f"{currency}-{expiry:%d%b%y}-{int(strike)}-{'C' if call else 'P'}".upper(),
        "kind": str(Kind.OPTION),
        "base_currency": currency,
        "quote_currency": "USD",
        "settlement_currency": currency,
        "instrument_type": "reversed",
        "contract_size": 1.0,
        "tick_size": 0.0005,
        "min_trade_amount": 0.1,
        "maker_commission": 0.0003,
        "taker_commission": 0.0003,
        "expiration_timestamp": int(expiry.timestamp() * 1000),
        "strike": strike,
        "option_type": "call" if call else "put",
        "price_index": f"{currency.lower()}_usd",
        "is_active": True,
    }


def future_payload(currency: str, expiry: datetime) -> dict[str, Any]:
    return {
        "instrument_name": f"{currency}-{expiry:%d%b%y}".upper(),
        "kind": str(Kind.FUTURE),
        "base_currency": currency,
        "quote_currency": "USD",
        "settlement_currency": currency,
        "instrument_type": "reversed",
        "contract_size": 10.0,
        "tick_size": 0.5,
        "min_trade_amount": 10.0,
        "expiration_timestamp": int(expiry.timestamp() * 1000),
        "price_index": f"{currency.lower()}_usd",
        "is_active": True,
    }


class FakeVenue:
    """Implements the MarketDataSource protocol with scripted responses."""

    def __init__(
        self,
        currencies: tuple[str, ...] = (BASE,),
        index: float = 60000.0,
        index_path: list[float] | None = None,
    ) -> None:
        self.currencies = currencies
        self.index = index
        self.index_path = list(index_path) if index_path else None
        self.calls: list[str] = []

    @property
    def label(self) -> str:
        return "FakeVenue"

    @property
    def is_live(self) -> bool:
        return True

    def now(self) -> datetime:
        return AS_OF

    def _wrap(self, endpoint: str, result: Any, params: dict[str, Any]) -> RawResponse:
        self.calls.append(endpoint)
        body = json.dumps({"result": result}, sort_keys=True).encode()
        return RawResponse(
            provenance=Provenance(
                source=self.label, endpoint=endpoint, retrieved_at=AS_OF, params=params
            ),
            body=body,
        )

    def fetch_instruments(self, currency: str, kind: Kind) -> RawResponse:
        expiries = _expiries()
        if kind is Kind.OPTION:
            result = [
                option_payload(currency, expiry, strike, call)
                for expiry in expiries
                for strike in (50000.0, 60000.0, 70000.0)
                for call in (True, False)
            ]
        else:
            result = [future_payload(currency, expiry) for expiry in expiries]
        return self._wrap(
            "public/get_instruments", result, {"currency": currency, "kind": str(kind)}
        )

    def fetch_index(self, index_name: str) -> RawResponse:
        value = self.index
        if self.index_path:
            value = self.index_path.pop(0)
        return self._wrap(
            "public/get_index_price", {"index_price": value}, {"index_name": index_name}
        )

    def fetch_book_summary(self, currency: str, kind: Kind) -> RawResponse:
        expiries = _expiries()
        if kind is Kind.FUTURE:
            result = [
                {
                    "instrument_name": future_payload(currency, expiry)["instrument_name"],
                    "mark_price": self.index * (1 + 0.01 * (n + 1)),
                }
                for n, expiry in enumerate(expiries)
            ]
        else:
            result = [
                {
                    "instrument_name": option_payload(currency, expiry, 60000.0, True)[
                        "instrument_name"
                    ],
                    "mark_price": 0.05,
                }
                for expiry in expiries
            ]
        return self._wrap(
            "public/get_book_summary_by_currency",
            result,
            {"currency": currency, "kind": str(kind)},
        )


@pytest.fixture
def venue() -> FakeVenue:
    return FakeVenue()


@pytest.fixture
def cross_venue() -> FakeVenue:
    return FakeVenue(currencies=(BASE, ALT))


@pytest.fixture
def store(tmp_path):
    from voltk.snapshots import SnapshotStore

    return SnapshotStore(tmp_path / "snapshots.db")
