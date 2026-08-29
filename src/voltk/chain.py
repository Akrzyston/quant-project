"""Chain hygiene: what gets excluded before pricing, and why.

Raw-quote hygiene only -- existence, positivity, crossed markets. Model-
dependent economic checks (parity, no-arbitrage bounds) live in validation.py
and have their own display surface; folding them into this funnel would blur
"the data is broken" with "the data disagrees with my model."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from voltk.instruments import Instrument
from voltk.marketdata.parse import Quote


@dataclass(frozen=True, slots=True)
class FilterOutcome:
    name: str
    description: str
    total: int
    dropped: tuple[str, ...]

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    @property
    def dropped_fraction(self) -> float:
        return self.dropped_count / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class ChainReport:
    currency: str
    total: int
    kept: tuple[Instrument, ...]
    outcomes: tuple[FilterOutcome, ...]

    def summary(self) -> str:
        return f"{len(self.kept)} of {self.total} instruments kept."


def filter_chain(
    instruments: Sequence[Instrument],
    marks: Mapping[str, float],
    quotes: Mapping[str, Quote] | None = None,
) -> ChainReport:
    """Sequential hygiene funnel.

    Each rule runs only on the survivors of the ones before it, so an
    instrument is attributed to the first rule that drops it, never more than
    one. Every dropped_fraction is against the original chain size, not the
    shrinking survivor pool, so the rules can be read independently.
    """
    total = len(instruments)
    currency = instruments[0].base_currency if instruments else ""
    survivors = list(instruments)
    outcomes: list[FilterOutcome] = []

    missing = [i for i in survivors if i.name not in marks]
    survivors = [i for i in survivors if i not in missing]
    outcomes.append(FilterOutcome(
        "missing quote", "No mark price from the venue.", total,
        tuple(i.name for i in missing),
    ))

    non_positive = [i for i in survivors if marks.get(i.name, 0.0) <= 0]
    survivors = [i for i in survivors if i not in non_positive]
    outcomes.append(FilterOutcome(
        "non-positive mark", "Mark price at or below zero.", total,
        tuple(i.name for i in non_positive),
    ))

    crossed: list[Instrument] = []
    if quotes is not None:
        for inst in survivors:
            q = quotes.get(inst.name)
            if q is not None and q.bid is not None and q.ask is not None and q.bid > q.ask:
                crossed.append(inst)
        survivors = [i for i in survivors if i not in crossed]
    outcomes.append(FilterOutcome(
        "crossed market", "Bid above ask.", total,
        tuple(i.name for i in crossed),
    ))

    return ChainReport(
        currency=currency, total=total, kept=tuple(survivors), outcomes=tuple(outcomes),
    )
