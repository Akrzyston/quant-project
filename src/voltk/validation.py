"""Validation against observed quotes.

Parity holds exactly in the models but not on a real chain (stale, wide,
asynchronous quotes), so a breach wider than the combined spread is a data
problem, not an arbitrage. Reports; does not clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from voltk.instruments import Instrument, OptionType
from voltk.models.base import CP
from voltk.universe import Universe


@dataclass(frozen=True, slots=True)
class ParityRow:
    currency: str
    expiry: datetime
    strike: float
    call_price: float
    put_price: float
    forward: float
    gap: float
    tolerance: float

    @property
    def breached(self) -> bool:
        return abs(self.gap) > self.tolerance

    @property
    def gap_bps_of_forward(self) -> float:
        return abs(self.gap) / self.forward * 10_000 if self.forward else 0.0


@dataclass(frozen=True, slots=True)
class ParityReport:
    rows: tuple[ParityRow, ...]
    settles_in_base: bool

    @property
    def checked(self) -> int:
        return len(self.rows)

    @property
    def breaches(self) -> tuple[ParityRow, ...]:
        return tuple(row for row in self.rows if row.breached)

    @property
    def worst(self) -> ParityRow | None:
        return max(self.rows, key=lambda r: abs(r.gap), default=None)

    def summary(self) -> str:
        if not self.rows:
            return "No call-put pairs with two-sided marks."
        return (
            f"{len(self.breaches)} of {self.checked} pairs breach parity; "
            f"worst {self.worst.gap_bps_of_forward:.1f} bps of forward"
        )


def call_put_pairs(
    universe: Universe, currency: str, expiry: datetime
) -> list[tuple[float, Instrument, Instrument]]:
    """Strike, call, put for every strike quoted on both sides at one expiry."""
    calls: dict[float, Instrument] = {}
    puts: dict[float, Instrument] = {}
    for inst in universe.options(currency):
        if inst.expiry != expiry or inst.strike is None:
            continue
        target = calls if inst.option_type is OptionType.CALL else puts
        target[inst.strike] = inst
    return [(k, calls[k], puts[k]) for k in sorted(set(calls) & set(puts))]


def parity_report(
    universe: Universe,
    marks: Mapping[str, float],
    *,
    currency: str,
    expiries: Sequence[datetime] | None = None,
    settles_in_base: bool = True,
    spreads: Mapping[str, float] | None = None,
    floor_bps: float = 5.0,
) -> ParityReport:
    """Check C - P against its no-arbitrage value for every two-sided pair.

    settles_in_base selects the relation: coin-settled contracts satisfy
    C - P = 1 - K/F, quote-settled ones satisfy C - P = D(F - K).
    """
    spreads = spreads or {}
    rows: list[ParityRow] = []
    targets = expiries if expiries is not None else universe.expiries(currency)

    for expiry in targets:
        forward = universe.forward_for(currency, expiry)
        if forward <= 0:
            continue
        rate = universe.implied_rate(currency, expiry)
        tau = (expiry - universe.as_of).total_seconds() / (365.0 * 24 * 3600)

        for strike, call, put in call_put_pairs(universe, currency, expiry):
            call_price, put_price = marks.get(call.name), marks.get(put.name)
            if call_price is None or put_price is None:
                continue

            if settles_in_base:
                expected = 1.0 - strike / forward
                scale = 1.0
            else:
                expected = math.exp(-rate * max(tau, 0.0)) * (forward - strike)
                scale = forward

            observed = call_price - put_price
            # A pair cannot be judged tighter than the two spreads allow.
            spread = spreads.get(call.name, 0.0) + spreads.get(put.name, 0.0)
            tolerance = max(spread, floor_bps / 10_000 * scale)

            rows.append(
                ParityRow(
                    currency=currency,
                    expiry=expiry,
                    strike=strike,
                    call_price=call_price,
                    put_price=put_price,
                    forward=forward,
                    gap=observed - expected,
                    tolerance=tolerance,
                )
            )

    return ParityReport(rows=tuple(rows), settles_in_base=settles_in_base)


def bounds_violations(
    universe: Universe,
    marks: Mapping[str, float],
    model,
    *,
    currency: str,
) -> list[tuple[Instrument, float, str]]:
    """Quotes sitting outside the model's no-arbitrage band."""
    found: list[tuple[Instrument, float, str]] = []
    for inst in universe.options(currency):
        price = marks.get(inst.name)
        if price is None or inst.expiry is None or inst.strike is None:
            continue
        tau = max(inst.tau(universe.as_of) or 0.0, 0.0)
        forward = universe.forward_for(currency, inst.expiry)
        cp = CP.CALL if inst.option_type is OptionType.CALL else CP.PUT
        band = model.bounds(forward, inst.strike, tau, 0.0, cp)
        if not band.contains(price, tolerance=1e-12):
            side = "below intrinsic" if price < band.lower else "above the cap"
            found.append((inst, price, side))
    return found
