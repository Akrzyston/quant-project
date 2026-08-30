"""Deribit public API client.

Public endpoints only. Calls return the raw body with provenance so captures
store exactly what the venue sent.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import requests

from voltk.instruments import Kind
from voltk.marketdata.base import MarketDataError
from voltk.provenance import Provenance, RawResponse

PRODUCTION = "https://www.deribit.com/api/v2"
TESTNET = "https://test.deribit.com/api/v2"

ALL_CURRENCIES = "any"


class DeribitClient:
    def __init__(
        self,
        base_url: str = PRODUCTION,
        timeout: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "voltk/0.2"})

    @property
    def label(self) -> str:
        return f"Deribit {'testnet' if 'test.' in self._base_url else 'live'}"

    @property
    def is_live(self) -> bool:
        return True

    def now(self) -> datetime:
        return datetime.now(UTC)

    def fetch(self, method: str, **params: Any) -> RawResponse:
        url = f"{self._base_url}/public/{method}"
        requested_at = datetime.now(UTC)
        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
        except requests.RequestException as exc:
            raise MarketDataError(f"Could not reach {url}. Check the network, then retry.") from exc

        if response.status_code != 200:
            raise MarketDataError(f"HTTP {response.status_code} from {method} with {params}.")

        body = response.content
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise MarketDataError(f"{method} returned a body that is not JSON.") from exc

        if "error" in parsed:
            error = parsed["error"]
            raise MarketDataError(
                f"{method} rejected: {error.get('message', 'unknown')} (code {error.get('code')})."
            )

        return RawResponse(
            provenance=Provenance(
                source=self.label,
                endpoint=f"public/{method}",
                retrieved_at=requested_at,
                params=dict(params),
            ),
            body=body,
        )

    def fetch_instruments(self, currency: str, kind: Kind) -> RawResponse:
        return self.fetch("get_instruments", currency=currency, kind=str(kind), expired="false")

    def fetch_index(self, index_name: str) -> RawResponse:
        return self.fetch("get_index_price", index_name=index_name)

    def fetch_book_summary(self, currency: str, kind: Kind) -> RawResponse:
        return self.fetch("get_book_summary_by_currency", currency=currency, kind=str(kind))

    def fetch_currencies(self) -> RawResponse:
        return self.fetch("get_currencies")

    def fetch_dvol(
        self, currency: str, *, start: datetime, end: datetime, resolution: str = "3600"
    ) -> RawResponse:
        """DVOL candle history. resolution is seconds per candle ("1D" also
        accepted) -- "3600" is hourly.
        """
        return self.fetch(
            "get_volatility_index_data",
            currency=currency,
            start_timestamp=int(start.timestamp() * 1000),
            end_timestamp=int(end.timestamp() * 1000),
            resolution=resolution,
        )

    def fetch_historical_volatility(self, currency: str) -> RawResponse:
        """Deribit's own published realized volatility, whole history, no
        window control -- the venue does not accept a start/end here.
        """
        return self.fetch("get_historical_volatility", currency=currency)

    def fetch_candles(
        self, instrument_name: str, *, start: datetime, end: datetime, resolution: str = "60"
    ) -> RawResponse:
        """OHLCV candles for any instrument, spot or option. resolution is in
        minutes here ("60" = hourly, "1D" also accepted) -- fetch_dvol's own
        resolution is seconds, easy to conflate since both look like plain
        numeric strings.
        """
        return self.fetch(
            "get_tradingview_chart_data",
            instrument_name=instrument_name,
            start_timestamp=int(start.timestamp() * 1000),
            end_timestamp=int(end.timestamp() * 1000),
            resolution=resolution,
        )

    def server_time(self) -> datetime:
        return datetime.fromtimestamp(self.fetch("get_time").result() / 1000, tz=UTC)
