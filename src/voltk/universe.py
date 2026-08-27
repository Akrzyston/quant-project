"""Tradeable universe.

A universe is a set of currencies, not one. Capturing several in a single window
is what makes cross-currency work possible later: a BTC/ETH relative value trade
needs both chains observed at the same instant, and that cannot be reconstructed
by stitching two separate captures together.

Currencies are always passed in, discovered from the venue at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Iterator, Sequence

from voltk.instruments import Instrument, Kind


class UniverseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class UniverseSpec:
    """What to capture. Serialised into every snapshot."""

    currencies: tuple[str, ...]
    kinds: tuple[Kind, ...] = (Kind.OPTION, Kind.FUTURE)
    include_inactive: bool = False

    def __post_init__(self) -> None:
        if not self.currencies:
            raise UniverseError("A universe needs at least one currency.")
        if len(set(self.currencies)) != len(self.currencies):
            raise UniverseError(f"Duplicate currencies in {self.currencies}.")
        if not self.kinds:
            raise UniverseError("A universe needs at least one instrument kind.")

    @classmethod
    def of(cls, currencies: Iterable[str], kinds: Iterable[Kind] | None = None) -> UniverseSpec:
        ordered = tuple(dict.fromkeys(currencies))
        return cls(
            currencies=ordered,
            kinds=tuple(kinds) if kinds else (Kind.OPTION, Kind.FUTURE),
        )

    @property
    def is_cross_currency(self) -> bool:
        return len(self.currencies) > 1

    def key(self) -> str:
        kinds = "+".join(sorted(str(k) for k in self.kinds))
        return f"{'-'.join(self.currencies)}:{kinds}"


@dataclass(frozen=True, slots=True)
class ForwardPoint:
    currency: str
    expiry: datetime
    forward: float
    instrument_name: str


@dataclass(frozen=True, slots=True)
class Universe:
    """Instruments and prices resolved at one instant."""

    spec: UniverseSpec
    as_of: datetime
    instruments: tuple[Instrument, ...]
    index_prices: dict[str, float] = field(default_factory=dict)
    forward_curve: tuple[ForwardPoint, ...] = ()

    def __iter__(self) -> Iterator[Instrument]:
        return iter(self.instruments)

    def for_currency(self, currency: str) -> tuple[Instrument, ...]:
        return tuple(i for i in self.instruments if i.base_currency == currency)

    def options(self, currency: str | None = None) -> tuple[Instrument, ...]:
        return tuple(
            i
            for i in self.instruments
            if i.kind is Kind.OPTION and (currency is None or i.base_currency == currency)
        )

    def futures(self, currency: str | None = None) -> tuple[Instrument, ...]:
        return tuple(
            i
            for i in self.instruments
            if i.kind is Kind.FUTURE and (currency is None or i.base_currency == currency)
        )

    def expiries(self, currency: str) -> tuple[datetime, ...]:
        return tuple(sorted({i.expiry for i in self.options(currency) if i.expiry}))

    def strikes(self, currency: str, expiry: datetime) -> tuple[float, ...]:
        return tuple(
            sorted(
                {
                    i.strike
                    for i in self.options(currency)
                    if i.expiry == expiry and i.strike is not None
                }
            )
        )

    def index(self, currency: str) -> float:
        try:
            return self.index_prices[currency]
        except KeyError:
            raise UniverseError(
                f"No index price captured for {currency}. Available: "
                f"{sorted(self.index_prices) or 'none'}."
            ) from None

    def forwards(self, currency: str) -> tuple[ForwardPoint, ...]:
        return tuple(p for p in self.forward_curve if p.currency == currency)

    def forward_for(self, currency: str, expiry: datetime) -> float:
        """Interpolate the futures curve at an option expiry.

        Linear in time between bracketing futures; flat extrapolation outside.
        Option expiries rarely coincide with future expiries, so some rule is
        needed and this one is the least surprising.
        """
        points = sorted(self.forwards(currency), key=lambda p: p.expiry)
        if not points:
            return self.index(currency)
        if expiry <= points[0].expiry:
            return points[0].forward
        if expiry >= points[-1].expiry:
            return points[-1].forward
        for left, right in zip(points, points[1:]):
            if left.expiry <= expiry <= right.expiry:
                span = (right.expiry - left.expiry).total_seconds()
                if span == 0:
                    return left.forward
                weight = (expiry - left.expiry).total_seconds() / span
                return left.forward + weight * (right.forward - left.forward)
        return points[-1].forward


def currencies_with(instruments: Sequence[Instrument], kind: Kind) -> list[str]:
    """Base currencies that have live instruments of a kind, busiest first."""
    counts: dict[str, int] = {}
    for inst in instruments:
        if inst.kind is kind:
            counts[inst.base_currency] = counts.get(inst.base_currency, 0) + 1
    return sorted(counts, key=lambda c: (-counts[c], c))
