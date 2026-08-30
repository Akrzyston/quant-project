"""Turning a volatility view into a sized, hedged, risk-bounded position.

The standard framework (Natenberg): if implied vol looks rich against a
realized-vol forecast, sell options and delta-hedge to isolate the vol bet
from direction. A delta-hedged position's expected P&L over an interval is
gamma P&L from the actual move plus theta collected -- the same
representative-move construction voltk.quoting already uses for width, not
a separate assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from voltk.portfolio import PortfolioGreeks


class StrategyError(ValueError):
    """Bad caller input, e.g. a non-positive vega budget."""


def hedge_delta(option_delta_total: float) -> float:
    """Underlying size that zeroes portfolio delta against the option leg."""
    return -option_delta_total


def with_hedge(option_greeks: PortfolioGreeks, hedge_size: float) -> PortfolioGreeks:
    """Combined Greeks once the hedge is in place -- a linear underlying
    position only ever changes delta.
    """
    return replace(option_greeks, delta=option_greeks.delta + hedge_size)


@dataclass(frozen=True, slots=True)
class ExpectedEdge:
    gamma_pnl: float
    theta_pnl: float

    @property
    def total(self) -> float:
        return self.gamma_pnl + self.theta_pnl


def expected_daily_edge(gamma: float, theta: float, underlying: float, realized_vol: float) -> ExpectedEdge:
    """Delta-hedged expected P&L over one day: 0.5*gamma*(representative
    move)^2 + theta, where the representative move is sized from
    realized_vol the same way voltk.quoting.derive_width sizes its gamma
    term from implied vol. Positive gamma profits from the move regardless
    of sign; theta is whatever the position already carries. Pass the
    position's own signed Greeks (negative gamma for a net-short position).
    """
    dt = 1.0 / 365.0
    move = underlying * realized_vol * math.sqrt(dt)
    return ExpectedEdge(gamma_pnl=0.5 * gamma * move * move, theta_pnl=theta * dt)


def size_for_vega_budget(vega_per_contract: float, vega_budget: float) -> float:
    """Largest whole number of contracts keeping total vega within budget."""
    if vega_budget <= 0:
        raise StrategyError(f"vega_budget must be positive, got {vega_budget!r}.")
    if vega_per_contract == 0:
        raise StrategyError("vega_per_contract is zero; can't size off it.")
    return float(math.floor(vega_budget / abs(vega_per_contract)))


@dataclass(frozen=True, slots=True)
class Capacity:
    contracts: float
    open_interest: float
    participation: float


def capacity_from_open_interest(open_interest: float, *, participation: float = 0.1) -> Capacity:
    """How much size to show without dominating the visible market --
    a stated participation rate of open interest, not a hand-picked number
    per instrument.
    """
    if participation <= 0 or participation > 1:
        raise StrategyError(f"participation must be in (0, 1], got {participation!r}.")
    return Capacity(contracts=open_interest * participation, open_interest=open_interest, participation=participation)


@dataclass(frozen=True, slots=True)
class KillCheck:
    name: str
    breached: bool
    detail: str


def check_kill_conditions(
    *,
    realized_vol: float,
    entry_vol: float,
    vol_stop_multiple: float,
    pnl: float,
    risk_budget: float,
    loss_stop_fraction: float,
    tau: float,
    min_tau: float,
) -> tuple[KillCheck, ...]:
    """Three independent, always-evaluated checks -- never short-circuited,
    so a caller sees every breach at once rather than the first one found.
    """
    vol_stop = entry_vol * vol_stop_multiple
    return (
        KillCheck(
            name="realized vol thesis",
            breached=realized_vol >= vol_stop,
            detail=f"realized {realized_vol:.2%} vs stop {vol_stop:.2%} ({vol_stop_multiple:g}x entry {entry_vol:.2%})",
        ),
        KillCheck(
            name="loss budget",
            breached=pnl <= -loss_stop_fraction * risk_budget,
            detail=f"pnl {pnl:+.6f} vs stop {-loss_stop_fraction * risk_budget:+.6f}",
        ),
        KillCheck(
            name="time to expiry",
            breached=tau <= min_tau,
            detail=f"tau {tau:.4f}y vs floor {min_tau:.4f}y",
        ),
    )


@dataclass(frozen=True, slots=True)
class BacktestedEdge:
    n: int
    mean_daily_premium: float
    std_daily_premium: float
    annualized_sharpe: float | None


def backtest_premium_sharpe(daily_premiums: Sequence[float]) -> BacktestedEdge | None:
    """Sharpe of the daily implied-minus-realized premium itself, as a
    directional proxy for the delta-hedged edge -- both scale with
    implied-minus-realized variance. Not a dollar P&L backtest: there is no
    historical option chain to reprice a real position against day by day,
    only the premium series M6 already measures. A backtested number like
    this is exactly the kind that overstates a real Sharpe -- it's computed
    over whatever window happened not to contain a vol spike.
    """
    n = len(daily_premiums)
    if n < 2:
        return None
    mean = sum(daily_premiums) / n
    variance = sum((p - mean) ** 2 for p in daily_premiums) / (n - 1)
    std = math.sqrt(variance)
    sharpe = (mean / std) * math.sqrt(365.0) if std > 0 else None
    return BacktestedEdge(n=n, mean_daily_premium=mean, std_daily_premium=std, annualized_sharpe=sharpe)
