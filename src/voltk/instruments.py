"""Instrument metadata.

Contract conventions are read from the venue payload. A missing field raises
rather than defaulting, because a guessed multiplier is worse than a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Mapping

from voltk.provenance import Provenance


class Kind(StrEnum):
    OPTION = "option"
    FUTURE = "future"
    SPOT = "spot"
    FUTURE_COMBO = "future_combo"
    OPTION_COMBO = "option_combo"


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


class Settlement(StrEnum):
    INVERSE = "inverse"
    LINEAR = "linear"


class MetadataError(ValueError):
    """A venue payload lacked a field the toolkit refuses to guess."""


# Sentinel used by the venue for contracts with no expiry.
_PERPETUAL_TIMESTAMP_FLOOR = 4_000_000_000_000
_SECONDS_PER_YEAR = 365.0 * 24 * 3600


def _require(payload: Mapping[str, Any], key: str, instrument: str) -> Any:
    if payload.get(key) is None:
        raise MetadataError(
            f"{instrument}: missing {key!r}. Contract conventions are never defaulted."
        )
    return payload[key]


@dataclass(frozen=True, slots=True)
class Instrument:
    name: str
    kind: Kind
    base_currency: str
    quote_currency: str
    settlement_currency: str
    settlement: Settlement
    contract_size: float
    tick_size: float
    min_trade_amount: float
    maker_commission: float | None = None
    taker_commission: float | None = None
    price_index: str | None = None
    expiry: datetime | None = None
    strike: float | None = None
    option_type: OptionType | None = None
    is_active: bool = True
    provenance: Provenance | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_inverse(self) -> bool:
        return self.settlement is Settlement.INVERSE

    @property
    def expiry_label(self) -> str:
        return self.expiry.strftime("%d%b%y").upper() if self.expiry else "PERPETUAL"

    @property
    def retrieved_at(self) -> datetime | None:
        return self.provenance.retrieved_at if self.provenance else None

    def tau(self, as_of: datetime) -> float | None:
        """ACT/365 year fraction to expiry. as_of is required so replay is deterministic."""
        if self.expiry is None:
            return None
        return (self.expiry - as_of).total_seconds() / _SECONDS_PER_YEAR

    def is_expired(self, as_of: datetime) -> bool:
        return self.expiry is not None and self.expiry <= as_of

    @classmethod
    def from_deribit(
        cls, payload: Mapping[str, Any], provenance: Provenance | None = None
    ) -> Instrument:
        name = payload.get("instrument_name") or "<unnamed>"
        base = _require(payload, "base_currency", name)
        settlement_currency = payload.get("settlement_currency") or base

        expiry_ms = payload.get("expiration_timestamp")
        expiry = (
            datetime.fromtimestamp(expiry_ms / 1000, tz=UTC)
            if expiry_ms and expiry_ms < _PERPETUAL_TIMESTAMP_FLOOR
            else None
        )

        instrument_type = payload.get("instrument_type")
        if instrument_type is not None:
            inverse = instrument_type == "reversed"
        else:
            inverse = settlement_currency == base
        option_type = payload.get("option_type")

        return cls(
            name=name,
            kind=Kind(_require(payload, "kind", name)),
            base_currency=base,
            quote_currency=payload.get("quote_currency") or payload.get("counter_currency", ""),
            settlement_currency=settlement_currency,
            settlement=Settlement.INVERSE if inverse else Settlement.LINEAR,
            contract_size=float(_require(payload, "contract_size", name)),
            tick_size=float(_require(payload, "tick_size", name)),
            min_trade_amount=float(_require(payload, "min_trade_amount", name)),
            maker_commission=_optional_float(payload.get("maker_commission")),
            taker_commission=_optional_float(payload.get("taker_commission")),
            price_index=payload.get("price_index"),
            expiry=expiry,
            strike=_optional_float(payload.get("strike")),
            option_type=OptionType(option_type) if option_type else None,
            is_active=bool(payload.get("is_active", True)),
            provenance=provenance,
            raw=dict(payload),
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
