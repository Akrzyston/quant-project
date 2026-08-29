"""Minimal portfolio aggregation: cash always sums, native only within one
settlement currency and convention.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from voltk.greeks import Greeks, Unit
from voltk.models.base import CP
from voltk.portfolio import Position, PortfolioError, aggregate_greeks, vega_by_expiry

EXPIRY_1 = datetime(2026, 9, 19, 8, 0, tzinfo=UTC)
EXPIRY_2 = datetime(2026, 10, 17, 8, 0, tzinfo=UTC)


def _greeks(delta: float, vega: float) -> Greeks:
    return Greeks(
        delta=delta, gamma=0.001, vega=vega, theta=-0.01, rho=0.0,
        vanna=0.0, volga=0.0, unit=Unit.QUOTE, vega_bump=1.0, theta_period=1.0,
    )


def _position(size: float, currency: str = "XBT", expiry=EXPIRY_1, settles_in_base=True) -> Position:
    return Position(
        instrument_name="XBT-TEST", currency=currency, expiry=expiry, strike=60_000.0,
        cp=CP.CALL, size=size, settles_in_base=settles_in_base,
        coin_greeks=_greeks(0.5, 100.0), cash_greeks=_greeks(30_000.0, 6_000_000.0),
    )


def test_aggregate_greeks_sums_cash_greeks_across_positions() -> None:
    positions = [_position(2.0), _position(-1.0)]

    result = aggregate_greeks(positions, use_cash=True)

    assert result.delta == pytest.approx(2.0 * 30_000.0 + (-1.0) * 30_000.0)
    assert result.vega == pytest.approx(2.0 * 6_000_000.0 + (-1.0) * 6_000_000.0)


def test_aggregate_greeks_sums_native_greeks_within_one_currency() -> None:
    positions = [_position(2.0), _position(3.0)]

    result = aggregate_greeks(positions, use_cash=False)

    assert result.delta == pytest.approx(5.0 * 0.5)


def test_native_aggregation_raises_on_mixed_currencies() -> None:
    positions = [_position(1.0, currency="XBT"), _position(1.0, currency="XET")]

    with pytest.raises(PortfolioError):
        aggregate_greeks(positions, use_cash=False)


def test_native_aggregation_raises_on_mixed_settlement_convention() -> None:
    positions = [_position(1.0, settles_in_base=True), _position(1.0, settles_in_base=False)]

    with pytest.raises(PortfolioError):
        aggregate_greeks(positions, use_cash=False)


def test_cash_aggregation_never_raises_on_mixed_currencies() -> None:
    positions = [_position(1.0, currency="XBT"), _position(1.0, currency="XET")]

    aggregate_greeks(positions, use_cash=True)  # should not raise


def test_aggregate_greeks_on_empty_portfolio_is_all_zero() -> None:
    result = aggregate_greeks([], use_cash=True)
    assert result.delta == 0.0 and result.vega == 0.0


def test_vega_by_expiry_groups_correctly() -> None:
    positions = [
        _position(1.0, expiry=EXPIRY_1),
        _position(2.0, expiry=EXPIRY_1),
        _position(1.0, expiry=EXPIRY_2),
    ]

    grouped = vega_by_expiry(positions, use_cash=True)

    assert grouped[EXPIRY_1] == pytest.approx(3.0 * 6_000_000.0)
    assert grouped[EXPIRY_2] == pytest.approx(1.0 * 6_000_000.0)
