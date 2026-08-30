from __future__ import annotations

import math

import pytest

from voltk.portfolio import PortfolioGreeks
from voltk.strategy import (
    StrategyError,
    capacity_from_open_interest,
    check_kill_conditions,
    expected_daily_edge,
    hedge_delta,
    size_for_vega_budget,
    with_hedge,
)


def test_hedge_delta_zeroes_out_the_combined_position() -> None:
    hedge = hedge_delta(12.5)
    greeks = PortfolioGreeks(delta=12.5, gamma=-0.01, vega=-50.0, theta=20.0, rho=0.0, vanna=0.0, volga=0.0)
    assert with_hedge(greeks, hedge).delta == pytest.approx(0.0)


def test_with_hedge_only_changes_delta() -> None:
    greeks = PortfolioGreeks(delta=0.0, gamma=-0.01, vega=-50.0, theta=20.0, rho=1.0, vanna=2.0, volga=3.0)
    combined = with_hedge(greeks, 5.0)
    assert combined.gamma == greeks.gamma
    assert combined.vega == greeks.vega
    assert combined.theta == greeks.theta


def _consistent_short_gamma_theta(gamma: float, implied_vol: float, underlying: float) -> float:
    # theta ~ -0.5*gamma*implied_vol^2*S^2 is the standard delta-hedged
    # relationship (Natenberg): a short-gamma position's theta collection is
    # sized by the same implied vol it was struck at.
    return -0.5 * gamma * implied_vol * implied_vol * underlying * underlying


def test_expected_daily_edge_short_position_profits_when_realized_stays_below_entry_vol() -> None:
    underlying, gamma = 60_000.0, -0.00001
    theta = _consistent_short_gamma_theta(gamma, implied_vol=0.4, underlying=underlying)

    edge = expected_daily_edge(gamma=gamma, theta=theta, underlying=underlying, realized_vol=0.2)

    assert edge.total > 0.0
    assert edge.theta_pnl == pytest.approx(theta / 365.0)


def test_expected_daily_edge_short_position_loses_when_realized_exceeds_entry_vol() -> None:
    underlying, gamma = 60_000.0, -0.00001
    theta = _consistent_short_gamma_theta(gamma, implied_vol=0.4, underlying=underlying)

    edge = expected_daily_edge(gamma=gamma, theta=theta, underlying=underlying, realized_vol=0.8)

    assert edge.total < 0.0


def test_size_for_vega_budget_floors_to_whole_contracts() -> None:
    assert size_for_vega_budget(vega_per_contract=30.0, vega_budget=100.0) == 3


def test_size_for_vega_budget_rejects_non_positive_budget() -> None:
    with pytest.raises(StrategyError):
        size_for_vega_budget(vega_per_contract=30.0, vega_budget=0.0)


def test_capacity_scales_with_participation() -> None:
    cap = capacity_from_open_interest(1000.0, participation=0.1)
    assert cap.contracts == pytest.approx(100.0)


def test_capacity_rejects_bad_participation() -> None:
    with pytest.raises(StrategyError):
        capacity_from_open_interest(1000.0, participation=1.5)


def test_kill_conditions_report_every_breach_not_just_the_first() -> None:
    checks = check_kill_conditions(
        realized_vol=0.9, entry_vol=0.4, vol_stop_multiple=1.5,
        pnl=-500.0, risk_budget=100.0, loss_stop_fraction=1.0,
        tau=0.001, min_tau=0.01,
    )
    assert {c.name for c in checks if c.breached} == {"realized vol thesis", "loss budget", "time to expiry"}


def test_kill_conditions_none_breached_when_everything_is_fine() -> None:
    checks = check_kill_conditions(
        realized_vol=0.3, entry_vol=0.4, vol_stop_multiple=1.5,
        pnl=10.0, risk_budget=100.0, loss_stop_fraction=1.0,
        tau=0.5, min_tau=0.01,
    )
    assert all(not c.breached for c in checks)
