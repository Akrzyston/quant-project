"""Venue conventions loaded from a sourced spec file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import time
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable

SPEC_PACKAGE = "voltk.specs"


class ConventionError(ValueError):
    """Live metadata contradicts the recorded convention."""


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    maker_rate: float
    taker_rate: float
    delivery_rate: float
    cap_fraction_of_premium: float | None

    def fee(self, underlying_price: float, premium: float, *, maker: bool) -> float:
        """Charged on the underlying, optionally capped against the premium."""
        rate = self.maker_rate if maker else self.taker_rate
        raw = rate * underlying_price
        if self.cap_fraction_of_premium is None:
            return raw
        return min(raw, self.cap_fraction_of_premium * premium)


@dataclass(frozen=True, slots=True)
class VenueConventions:
    venue: str
    source_url: str
    retrieved_at: str
    schema_version: int
    expiry_time_utc: time
    settlement_window_minutes: int
    fees: dict[str, FeeSchedule]

    def fee_schedule(self, kind: str) -> FeeSchedule:
        try:
            return self.fees[kind]
        except KeyError:
            raise ConventionError(
                f"No fee schedule for kind {kind!r} in the {self.venue} spec."
            ) from None


def _fee_schedule(block: dict[str, Any], notional_key: str) -> FeeSchedule:
    return FeeSchedule(
        maker_rate=float(block[f"maker_rate_of_{notional_key}"]),
        taker_rate=float(block[f"taker_rate_of_{notional_key}"]),
        delivery_rate=float(block[f"delivery_rate_of_{notional_key}"]),
        cap_fraction_of_premium=(
            float(block["cap_fraction_of_premium"])
            if block.get("cap_fraction_of_premium") is not None
            else None
        ),
    )


@lru_cache(maxsize=8)
def load(venue: str) -> VenueConventions:
    path = resources.files(SPEC_PACKAGE).joinpath(f"{venue}.json")
    if not path.is_file():
        raise ConventionError(f"No spec file for venue {venue!r}.")
    spec = json.loads(path.read_text())
    hours, minutes = (int(part) for part in spec["expiry"]["time_utc"].split(":"))
    return VenueConventions(
        venue=spec["venue"],
        source_url=spec["source_url"],
        retrieved_at=spec["retrieved_at"],
        schema_version=int(spec["schema_version"]),
        expiry_time_utc=time(hours, minutes),
        settlement_window_minutes=int(spec["settlement"]["index_window_minutes"]),
        fees={
            "option": _fee_schedule(spec["fees"]["option"], "underlying"),
            "future": _fee_schedule(spec["fees"]["future"], "notional"),
        },
    )


def validate_expiry_convention(instruments: Iterable[Any], conventions: VenueConventions) -> None:
    """Check live expiries land on the documented time."""
    offenders = [
        inst.name
        for inst in instruments
        if inst.expiry is not None
        and (inst.expiry.hour, inst.expiry.minute) != (
            conventions.expiry_time_utc.hour,
            conventions.expiry_time_utc.minute,
        )
    ]
    if offenders:
        raise ConventionError(
            f"{len(offenders)} instruments expire off the documented "
            f"{conventions.expiry_time_utc:%H:%M} UTC convention, first: {offenders[0]}. "
            "Update the spec file rather than the check."
        )
