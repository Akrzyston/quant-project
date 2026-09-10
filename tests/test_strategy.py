from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from voltk.market_making import SessionStep
from voltk.models.inverse import InverseOption
from voltk.portfolio import PortfolioGreeks
from voltk.strategy import (
    StrategyError,
    autocorrelation,
    autocorrelation_adjusted_sharpe,
    backtest_premium_sharpe,
    capacity_from_open_interest,
    check_kill_conditions,
    expected_daily_edge,
    hedge_delta,
    simulate_short_straddle_path,
    size_for_vega_budget,
    skewness,
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


def test_backtest_premium_sharpe_matches_a_hand_computed_case() -> None:
    result = backtest_premium_sharpe([0.01, 0.03, 0.02, -0.01, 0.02])
    mean = (0.01 + 0.03 + 0.02 - 0.01 + 0.02) / 5
    variance = sum((p - mean) ** 2 for p in [0.01, 0.03, 0.02, -0.01, 0.02]) / 4
    assert result.mean_daily_premium == pytest.approx(mean)
    assert result.annualized_sharpe == pytest.approx((mean / math.sqrt(variance)) * math.sqrt(365.0))


def test_backtest_premium_sharpe_none_with_too_few_observations() -> None:
    assert backtest_premium_sharpe([0.01]) is None


def test_autocorrelation_matches_a_hand_computed_case() -> None:
    # period-2 series [0, 1, 0, 1]: mean 0.5, deviations [-.5, .5, -.5, .5]
    values = [0.0, 1.0, 0.0, 1.0]
    variance = sum((v - 0.5) ** 2 for v in values)  # 1.0
    covariance = (-0.5 * 0.5) + (0.5 * -0.5) + (-0.5 * 0.5)  # -0.75
    assert autocorrelation(values, lag=1) == pytest.approx(covariance / variance)


def test_autocorrelation_none_with_too_few_points() -> None:
    assert autocorrelation([1.0, 2.0], lag=1) is None


def test_autocorrelation_none_for_a_constant_series() -> None:
    assert autocorrelation([1.0, 1.0, 1.0, 1.0], lag=1) is None


def test_skewness_is_zero_for_a_symmetric_series() -> None:
    assert skewness([-1.0, 0.0, 1.0]) == pytest.approx(0.0)


def test_skewness_is_positive_for_a_right_skewed_series() -> None:
    assert skewness([1.0, 1.0, 1.0, 10.0]) > 0.0


def test_skewness_none_below_three_points() -> None:
    assert skewness([1.0, 2.0]) is None


def test_autocorrelation_adjusted_sharpe_matches_the_stated_formula() -> None:
    premiums = [0.01, 0.03, 0.02, -0.01, 0.02, 0.015]
    raw = backtest_premium_sharpe(premiums)
    rho = autocorrelation(premiums, lag=1)

    result = autocorrelation_adjusted_sharpe(premiums)

    factor = (1.0 - rho) / (1.0 + rho)
    assert result.lag1_autocorrelation == pytest.approx(rho)
    assert result.effective_annual_observations == pytest.approx(365.0 * factor)
    assert result.adjusted_sharpe == pytest.approx(raw.annualized_sharpe * factor**0.5)


def test_autocorrelation_adjusted_sharpe_none_with_too_few_observations() -> None:
    assert autocorrelation_adjusted_sharpe([0.01]) is None


def test_autocorrelation_adjusted_sharpe_shrinks_a_highly_autocorrelated_series() -> None:
    # a smooth monotonic ramp (like a slow-moving vol level) has high rho,
    # so the adjustment should pull the annualized Sharpe well below the naive one
    premiums = [0.01 * i for i in range(1, 9)]
    raw = backtest_premium_sharpe(premiums)
    result = autocorrelation_adjusted_sharpe(premiums)
    assert result.lag1_autocorrelation > 0.5
    assert abs(result.adjusted_sharpe) < abs(raw.annualized_sharpe)


INVERSE = InverseOption()
STRIKE = 60_000.0
RATE = 0.0
START = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)


def _path_steps(underlyings: list[float], vol: float = 0.6, tau0: float = 0.1) -> list[SessionStep]:
    return [
        SessionStep(timestamp=START + timedelta(days=i), underlying=u, tau=max(tau0 - i * 0.01, 1e-4), vol=vol)
        for i, u in enumerate(underlyings)
    ]


def test_simulate_short_straddle_path_too_few_steps_returns_empty() -> None:
    steps = _path_steps([60_000.0])
    assert simulate_short_straddle_path(INVERSE, steps, strike=STRIKE, rate=RATE, size=1.0) == ()


def test_simulate_short_straddle_path_flat_underlying_has_zero_hedge_pnl() -> None:
    steps = _path_steps([60_000.0] * 5)

    path = simulate_short_straddle_path(INVERSE, steps, strike=STRIKE, rate=RATE, size=1.0)

    assert len(path) == 4
    assert all(p.hedge_pnl == pytest.approx(0.0) for p in path)
    # ATM straddle, short, tau shrinking toward expiry with nothing else
    # moving: theta decay helps a short position, so each step should profit
    assert all(p.option_pnl > 0.0 for p in path)


def test_simulate_short_straddle_path_cumulative_pnl_is_the_running_sum() -> None:
    steps = _path_steps([60_000.0, 61_000.0, 59_500.0, 60_200.0])

    path = simulate_short_straddle_path(INVERSE, steps, strike=STRIKE, rate=RATE, size=1.0)

    running = 0.0
    for p in path:
        running += p.total_pnl
        assert p.cumulative_pnl == pytest.approx(running)


def test_simulate_short_straddle_path_hedge_offsets_most_of_a_directional_move() -> None:
    steps = _path_steps([60_000.0, 66_000.0])  # a large single move

    path = simulate_short_straddle_path(INVERSE, steps, strike=STRIKE, rate=RATE, size=1.0)

    assert len(path) == 1
    assert abs(path[0].total_pnl) < abs(path[0].option_pnl)
