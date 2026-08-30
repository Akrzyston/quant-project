from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from voltk.market_making import Side, SessionStep, simulate_session
from voltk.models.base import CP
from voltk.models.black76 import Black76

MODEL = Black76()
STRIKE = 60_000.0
RATE = 0.0
START = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)


def _steps(forwards: list[float], vol: float = 0.6, tau0: float = 0.1) -> list[SessionStep]:
    return [
        SessionStep(timestamp=START + timedelta(hours=i), underlying=f, tau=tau0 - i * 1e-6, vol=vol)
        for i, f in enumerate(forwards)
    ]


def test_no_fill_when_the_path_stays_inside_the_quote() -> None:
    steps = _steps([60_000.0] * 5)
    half_width = 500.0  # much wider than the flat path ever moves

    result = simulate_session(MODEL, steps, strike=STRIKE, rate=RATE, cp=CP.CALL, half_width=half_width)

    assert result.fills == ()
    assert result.ending_inventory == 0.0


def test_a_rising_path_lifts_the_offer() -> None:
    steps = _steps([60_000.0, 62_000.0, 62_010.0])
    half_width = 50.0

    result = simulate_session(MODEL, steps, strike=STRIKE, rate=RATE, cp=CP.CALL, half_width=half_width)

    assert len(result.fills) == 1
    assert result.fills[0].fill.side == Side.SELL
    assert result.ending_inventory == -1.0


def test_edge_captured_is_the_quoted_half_width_with_no_inventory_skew() -> None:
    steps = _steps([60_000.0, 62_000.0, 62_010.0])
    half_width = 50.0

    result = simulate_session(MODEL, steps, strike=STRIKE, rate=RATE, cp=CP.CALL, half_width=half_width)

    assert result.fills[0].edge_captured == pytest.approx(half_width, rel=1e-6)


def test_fill_pnl_terms_sum_to_total_pnl() -> None:
    steps = _steps([60_000.0, 61_500.0, 62_000.0, 61_000.0])
    result = simulate_session(MODEL, steps, strike=STRIKE, rate=RATE, cp=CP.CALL, half_width=100.0)

    assert result.fills
    for f in result.fills:
        assert f.total_pnl == pytest.approx(
            f.edge_captured + f.delta_term + f.gamma_term + f.vega_term + f.theta_term + f.residual
        )
    assert result.total_pnl == pytest.approx(sum(f.total_pnl for f in result.fills))


def test_inventory_skew_can_trigger_a_second_fill_that_covers_the_position() -> None:
    # Step 0->1 sells us short; step 1->2 barely moves. Unshaded, that small
    # move doesn't reach the next bid. Skewed by the short inventory, the
    # reservation price (and so the bid) shifts up enough to get bought back.
    steps = _steps([60_000.0, 61_000.0, 60_990.0], tau0=0.5)

    unshaded = simulate_session(MODEL, steps, strike=STRIKE, rate=RATE, cp=CP.CALL, half_width=100.0)
    assert len(unshaded.fills) == 1
    assert unshaded.ending_inventory == -1.0

    shaded = simulate_session(
        MODEL, steps, strike=STRIKE, rate=RATE, cp=CP.CALL, half_width=100.0, risk_aversion=1000.0
    )
    assert [f.fill.side for f in shaded.fills] == [Side.SELL, Side.BUY]
    assert shaded.ending_inventory == 0.0
