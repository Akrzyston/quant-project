"""Simulated market-making session over a real underlying (and optionally
vol) path, with fills attributed rather than assumed profitable.

A fill happens whenever the next step's theoretical value would have moved
through our currently quoted side -- the maximally-informed counterparty,
since real adverse selection sits somewhere between this and never getting
run over. Each fill's markout P&L reuses voltk.pnl.attribute_pnl: the
directional move (delta+gamma) is what "adverse selection" means here,
vega is reported on its own, and the residual is never folded in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Sequence

from voltk.models.base import CP
from voltk.pnl import attribute_pnl

_SECONDS_PER_YEAR = 365.0 * 24 * 3600


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class SessionStep:
    timestamp: datetime
    underlying: float
    tau: float
    vol: float


@dataclass(frozen=True, slots=True)
class Fill:
    timestamp: datetime
    side: Side
    price: float
    theo_at_fill: float


@dataclass(frozen=True, slots=True)
class FillAttribution:
    fill: Fill
    edge_captured: float
    delta_term: float
    gamma_term: float
    vega_term: float
    theta_term: float
    residual: float

    @property
    def total_pnl(self) -> float:
        return self.edge_captured + self.delta_term + self.gamma_term + self.vega_term + self.theta_term + self.residual


@dataclass(frozen=True, slots=True)
class SessionResult:
    fills: tuple[FillAttribution, ...]
    ending_inventory: float

    @property
    def total_edge(self) -> float:
        return sum(f.edge_captured for f in self.fills)

    @property
    def total_adverse_selection(self) -> float:
        return sum(f.delta_term + f.gamma_term for f in self.fills)

    @property
    def total_vega_pnl(self) -> float:
        return sum(f.vega_term for f in self.fills)

    @property
    def total_theta_pnl(self) -> float:
        return sum(f.theta_term for f in self.fills)

    @property
    def total_residual(self) -> float:
        return sum(f.residual for f in self.fills)

    @property
    def total_pnl(self) -> float:
        return sum(f.total_pnl for f in self.fills)


def simulate_session(
    model,
    steps: Sequence[SessionStep],
    *,
    strike: float,
    rate: float,
    cp: CP,
    half_width: float,
    risk_aversion: float = 0.0,
    markout_steps: int = 1,
) -> SessionResult:
    """Quotes reservation_price +/- half_width at every step but the last
    markout_steps, skewed by accumulated inventory. half_width is fixed for
    the session -- it's a property of the strike's fit quality and the live
    market at the moment the session starts, not something that should
    change fill to fill.
    """
    inventory = 0.0
    fills: list[FillAttribution] = []

    for i in range(len(steps) - markout_steps):
        step = steps[i]
        theo = model.price(step.underlying, strike, step.tau, step.vol, rate, cp)
        greeks = model.greeks(step.underlying, strike, step.tau, step.vol, rate, cp)
        mid = theo - inventory * risk_aversion * step.vol * step.vol * step.tau
        bid, ask = mid - half_width, mid + half_width

        nxt = steps[i + 1]
        theo_next = model.price(nxt.underlying, strike, nxt.tau, nxt.vol, rate, cp)

        if theo_next >= ask:
            side, price, sign = Side.SELL, ask, -1.0
        elif theo_next <= bid:
            side, price, sign = Side.BUY, bid, 1.0
        else:
            continue

        inventory += sign
        mark = steps[min(i + markout_steps, len(steps) - 1)]
        theo_mark = model.price(mark.underlying, strike, mark.tau, mark.vol, rate, cp)
        d_t = (mark.timestamp - step.timestamp).total_seconds() / _SECONDS_PER_YEAR
        markout = attribute_pnl(greeks, theo, theo_mark, mark.underlying - step.underlying, mark.vol - step.vol, d_t)

        fills.append(
            FillAttribution(
                fill=Fill(timestamp=step.timestamp, side=side, price=price, theo_at_fill=theo),
                edge_captured=sign * (theo - price),
                delta_term=sign * markout.delta_term,
                gamma_term=sign * markout.gamma_term,
                vega_term=sign * markout.vega_term,
                theta_term=sign * markout.theta_term,
                residual=sign * markout.residual,
            )
        )

    return SessionResult(fills=tuple(fills), ending_inventory=inventory)
