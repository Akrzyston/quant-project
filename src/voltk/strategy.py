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
from datetime import datetime
from typing import Sequence

from voltk.market_making import SessionStep
from voltk.models.base import CP
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


def autocorrelation(values: Sequence[float], *, lag: int = 1) -> float | None:
    """Sample autocorrelation at the given lag: covariance of the series
    with itself shifted by `lag`, normalized by the full-series variance --
    the standard ACF estimator. None if there are too few points or the
    series has zero variance (correlation is undefined, not zero).
    """
    n = len(values)
    if n < lag + 2:
        return None
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values)
    if variance == 0:
        return None
    covariance = sum((values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag))
    return covariance / variance


def skewness(values: Sequence[float]) -> float | None:
    """Adjusted Fisher-Pearson sample skewness (the convention behind
    Excel's SKEW and scipy's skew(bias=False)). None below 3 points or with
    zero variance. A short-gamma position's P&L is textbook negatively
    skewed -- small frequent gains, rare large losses -- so a premium series
    with skew near zero or positive is not yet showing that signature.
    """
    n = len(values)
    if n < 3:
        return None
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    m3 = sum((v - mean) ** 3 for v in values) / n
    if m2 <= 0:
        return None
    g1 = m3 / m2**1.5
    return math.sqrt(n * (n - 1)) / (n - 2) * g1


@dataclass(frozen=True, slots=True)
class AutocorrelationAdjustedSharpe:
    raw_sharpe: float
    lag1_autocorrelation: float
    effective_annual_observations: float
    adjusted_sharpe: float


def autocorrelation_adjusted_sharpe(daily_premiums: Sequence[float]) -> AutocorrelationAdjustedSharpe | None:
    """`backtest_premium_sharpe`'s sqrt(365) annualization assumes 365
    independent daily draws. For an AR(1)-like series with lag-1
    autocorrelation rho, the variance of a sum over N days scales as
    N*(1+rho)/(1-rho) rather than N, so the correctly annualized Sharpe is
    the naive one scaled by sqrt((1-rho)/(1+rho)) -- equivalently, treating
    the year as having only `365*(1-rho)/(1+rho)` independent observations
    instead of 365. rho is clamped away from +/-1 so a near-degenerate
    series doesn't divide by zero.
    """
    raw = backtest_premium_sharpe(daily_premiums)
    if raw is None or raw.annualized_sharpe is None:
        return None
    rho = autocorrelation(daily_premiums, lag=1)
    if rho is None:
        return None
    rho = max(min(rho, 0.999), -0.999)
    factor = (1.0 - rho) / (1.0 + rho)
    return AutocorrelationAdjustedSharpe(
        raw_sharpe=raw.annualized_sharpe,
        lag1_autocorrelation=rho,
        effective_annual_observations=365.0 * factor,
        adjusted_sharpe=raw.annualized_sharpe * math.sqrt(factor),
    )


@dataclass(frozen=True, slots=True)
class PositionPathStep:
    timestamp: datetime
    underlying: float
    option_pnl: float
    hedge_pnl: float
    total_pnl: float
    cumulative_pnl: float


def simulate_short_straddle_path(
    model, steps: Sequence[SessionStep], *, strike: float, rate: float, size: float
) -> tuple[PositionPathStep, ...]:
    """Walks a short `size`-straddle position through a historical path,
    rehedging to flat delta at the end of every step -- the same
    delta-hedged construction this whole module assumes, applied day over
    day instead of once. P&L is exact repricing (price_now - price_prev),
    not a Taylor approximation, since the model is available at every step.

    `vol` on each step is a single number (DVOL, typically), not this
    strike's own historical smile -- nothing here stores historical smiles,
    so the approximation degrades as the strike drifts away from the money
    over the walk. Needs >=2 steps.
    """
    if len(steps) < 2:
        return ()
    first = steps[0]
    call_price = model.price(first.underlying, strike, first.tau, first.vol, rate, CP.CALL)
    put_price = model.price(first.underlying, strike, first.tau, first.vol, rate, CP.PUT)
    call_greeks = model.greeks(first.underlying, strike, first.tau, first.vol, rate, CP.CALL)
    put_greeks = model.greeks(first.underlying, strike, first.tau, first.vol, rate, CP.PUT)
    hedge = hedge_delta(size * (call_greeks.delta + put_greeks.delta))
    underlying_prev = first.underlying

    cumulative = 0.0
    out: list[PositionPathStep] = []
    for step in steps[1:]:
        call_now = model.price(step.underlying, strike, step.tau, step.vol, rate, CP.CALL)
        put_now = model.price(step.underlying, strike, step.tau, step.vol, rate, CP.PUT)
        option_pnl = -size * ((call_now - call_price) + (put_now - put_price))
        hedge_pnl = hedge * (step.underlying - underlying_prev)
        total = option_pnl + hedge_pnl
        cumulative += total
        out.append(PositionPathStep(
            timestamp=step.timestamp, underlying=step.underlying,
            option_pnl=option_pnl, hedge_pnl=hedge_pnl, total_pnl=total, cumulative_pnl=cumulative,
        ))
        call_greeks = model.greeks(step.underlying, strike, step.tau, step.vol, rate, CP.CALL)
        put_greeks = model.greeks(step.underlying, strike, step.tau, step.vol, rate, CP.PUT)
        hedge = hedge_delta(size * (call_greeks.delta + put_greeks.delta))
        call_price, put_price, underlying_prev = call_now, put_now, step.underlying
    return tuple(out)
