"""Implied forward from put-call parity.

Deribit's own chain display converts USD bid/ask off the index price, not the
forward. Reading that off the screen gives a chain that is quietly skewed at
every expiry by a nonzero basis. This module inverts the parity relation per
expiry, off marked quotes, for an independent estimate of the forward that can
be checked against the traded future directly.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from voltk.universe import Universe
from voltk.validation import call_put_pairs


@dataclass(frozen=True, slots=True)
class ImpliedForward:
    currency: str
    expiry: datetime
    forward: float
    traded_future: float
    basis: float
    basis_bps: float
    pairs_used: int
    per_strike: tuple[tuple[float, float], ...]


def forward_from_parity(
    strike: float, call_price: float, put_price: float, *, settles_in_base: bool, discount: float = 1.0
) -> float | None:
    """One strike's implied forward from put-call parity. None if the parity
    relation degenerates at this strike (a bad quote), not raised, so a
    caller can skip one strike out of a ladder without losing the rest.

    Coin-settled: C - P = 1 - K/F  =>  F_K = K / (1 - (C-P))
    Quote-settled: C - P = D(F-K)  =>  F_K = K + (C-P)/D, D = exp(-rate*tau)
    """
    diff = call_price - put_price
    if settles_in_base:
        denom = 1.0 - diff
        if denom <= 0:
            return None
        forward = strike / denom
    else:
        if discount <= 0:
            return None
        forward = strike + diff / discount
    return forward if forward > 0 else None


def implied_forward_curve(
    universe: Universe,
    marks: Mapping[str, float],
    *,
    currency: str,
    expiries: Sequence[datetime] | None = None,
    settles_in_base: bool = True,
) -> tuple[ImpliedForward, ...]:
    """Median per-strike implied forward, per expiry, from two-sided marks.

    Coin-settled: C - P = 1 - K/F  =>  F_K = K / (1 - (C-P))
    Quote-settled: C - P = D(F-K)  =>  F_K = K + (C-P)/D, D = exp(-rate*tau)

    Median over strikes rather than a regression: robust to one bad strike,
    with no extra machinery to validate.
    """
    results: list[ImpliedForward] = []
    targets = expiries if expiries is not None else universe.expiries(currency)

    for expiry in targets:
        traded_future = universe.forward_for(currency, expiry)
        rate = universe.implied_rate(currency, expiry)
        tau = (expiry - universe.as_of).total_seconds() / (365.0 * 24 * 3600)
        discount = math.exp(-rate * max(tau, 0.0))

        per_strike: list[tuple[float, float]] = []
        for strike, call, put in call_put_pairs(universe, currency, expiry):
            c, p = marks.get(call.name), marks.get(put.name)
            if c is None or p is None:
                continue
            forward = forward_from_parity(strike, c, p, settles_in_base=settles_in_base, discount=discount)
            if forward is not None:
                per_strike.append((strike, forward))

        if not per_strike:
            continue
        per_strike.sort()
        forward = statistics.median(f for _, f in per_strike)
        basis = forward - traded_future
        basis_bps = basis / traded_future * 10_000 if traded_future > 0 else 0.0

        results.append(
            ImpliedForward(
                currency=currency,
                expiry=expiry,
                forward=forward,
                traded_future=traded_future,
                basis=basis,
                basis_bps=basis_bps,
                pairs_used=len(per_strike),
                per_strike=tuple(per_strike),
            )
        )
    return tuple(results)
