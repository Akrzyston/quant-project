"""Minimal, session-local portfolio aggregation.

Real position tracking (persisted, with P&L) is M8's job per the milestone
table. This is deliberately smaller: a list of signed-size positions held
only in the dashboard's own session state, summed for a risk view.

Cash aggregation is always valid: cash value is additive across positions
once each position's own cash Greek has been correctly derived (see
voltk.greeks.cash_greeks_from_coin), regardless of what each position
settles in. Native-unit aggregation is only meaningful within one settlement
currency and one settlement convention -- summing coin Greeks across
different currencies, or mixing coin- and quote-settled positions natively,
is exactly the "reporting a Greek in coin terms as though it were the cash
exposure" failure mode, just one step removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from voltk.greeks import Greeks
from voltk.models.base import CP


class PortfolioError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Position:
    instrument_name: str
    currency: str
    expiry: datetime
    strike: float
    cp: CP
    size: float
    settles_in_base: bool
    coin_greeks: Greeks
    cash_greeks: Greeks


@dataclass(frozen=True, slots=True)
class PortfolioGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float


def _check_native_aggregation_is_valid(positions: Sequence[Position]) -> None:
    currencies = {p.currency for p in positions}
    settle_flags = {p.settles_in_base for p in positions}
    if len(currencies) > 1 or len(settle_flags) > 1:
        raise PortfolioError(
            "Native-unit aggregation needs one settlement currency and "
            f"convention; got currencies={currencies}, "
            f"settles_in_base={settle_flags}. Aggregate in cash terms instead."
        )


def aggregate_greeks(positions: Sequence[Position], *, use_cash: bool = True) -> PortfolioGreeks:
    if positions and not use_cash:
        _check_native_aggregation_is_valid(positions)

    totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0, "vanna": 0.0, "volga": 0.0}
    for position in positions:
        g = position.cash_greeks if use_cash else position.coin_greeks
        totals["delta"] += position.size * g.delta
        totals["gamma"] += position.size * g.gamma
        totals["vega"] += position.size * g.vega
        totals["theta"] += position.size * g.theta
        totals["rho"] += position.size * g.rho
        totals["vanna"] += position.size * g.vanna
        totals["volga"] += position.size * g.volga
    return PortfolioGreeks(**totals)


def vega_by_expiry(positions: Sequence[Position], *, use_cash: bool = True) -> dict[datetime, float]:
    result: dict[datetime, float] = {}
    for position in positions:
        g = position.cash_greeks if use_cash else position.coin_greeks
        result[position.expiry] = result.get(position.expiry, 0.0) + position.size * g.vega
    return result
