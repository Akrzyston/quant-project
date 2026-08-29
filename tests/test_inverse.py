"""Inverse option behaviour.

test_both_derivations_agree is the load-bearing one: the two implementations
share no code beyond Black-76, so agreement to machine precision is evidence the
adjustment was derived rather than assumed.
"""

from __future__ import annotations

import math

import pytest

from voltk.models.base import CP
from voltk.models.black76 import Black76
from voltk.models.inverse import InverseOption

RATE = 0.03
FORWARD = 60000.0
STRIKES = [20000.0, 60000.0, 150000.0]
TAUS = [0.02, 1.0]


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("cp", list(CP))
def test_both_derivations_agree(strike, tau, cp) -> None:
    model = InverseOption()
    for vol in (0.3, 1.1):
        direct = model.price(FORWARD, strike, tau, vol, RATE, cp)
        replicated = model.price_via_replication(FORWARD, strike, tau, vol, RATE, cp)
        assert direct == pytest.approx(replicated, rel=1e-13, abs=1e-15)


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("tau", TAUS)
def test_quote_currency_value_recovers_black76(strike, tau) -> None:
    """Multiplying the coin premium by spot returns the ordinary price."""
    inverse, black = InverseOption(), Black76()
    spot = FORWARD * math.exp(-RATE * tau)
    for cp in CP:
        coin = inverse.price(FORWARD, strike, tau, 0.6, RATE, cp)
        assert coin * spot == pytest.approx(
            black.price(FORWARD, strike, tau, 0.6, RATE, cp), rel=1e-12
        )


@pytest.mark.parametrize("strike", STRIKES)
def test_call_never_exceeds_one_coin(strike) -> None:
    model = InverseOption()
    for tau in TAUS:
        assert model.price(FORWARD, strike, tau, 0.8, RATE, CP.CALL) <= 1.0


def test_deep_in_the_money_call_saturates() -> None:
    """Value converges to 1 - K/F: close to one coin, never reaching it."""
    model = InverseOption()
    for strike in (10000.0, 1000.0, 1.0):
        value = model.price(FORWARD, strike, 0.25, 0.6, RATE, CP.CALL)
        assert value == pytest.approx(1.0 - strike / FORWARD, abs=1e-9)
        assert value < 1.0


def test_coin_delta_vanishes_at_both_extremes() -> None:
    """Hump-shaped in strike, unlike the monotone quote-currency delta."""
    model = InverseOption()
    strikes = [1.0, 30000.0, 60000.0, 100000.0, 1e9]
    deltas = [model.greeks(FORWARD, k, 0.25, 0.6, RATE, CP.CALL).delta for k in strikes]
    assert deltas[0] == pytest.approx(0.0, abs=1e-9)
    assert deltas[-1] == pytest.approx(0.0, abs=1e-9)
    assert max(deltas) in deltas[1:4]

    quote = [Black76().greeks(FORWARD, k, 0.25, 0.6, RATE, CP.CALL).delta for k in strikes]
    assert quote == sorted(quote, reverse=True)


def test_doubling_the_forward_halves_remaining_upside() -> None:
    """Value approaches one coin asymptotically, so upside decays like K/F."""
    model = InverseOption()
    remaining = [
        1.0 - model.price(forward, 60000.0, 0.02, 0.4, RATE, CP.CALL)
        for forward in (120000.0, 240000.0, 480000.0)
    ]
    for first, second in zip(remaining, remaining[1:]):
        assert second == pytest.approx(first / 2.0, rel=0.02)


def test_halving_the_forward_kills_the_position() -> None:
    model = InverseOption()
    values = [
        model.price(forward, 60000.0, 0.02, 0.4, RATE, CP.CALL)
        for forward in (60000.0, 30000.0, 15000.0)
    ]
    assert values == sorted(values, reverse=True)
    assert values[-1] < 1e-6


def test_hedging_with_quote_currency_delta_is_materially_wrong() -> None:
    """Deep in the money the two numbers are not close; one is near zero."""
    strike, tau, vol = 40000.0, 0.25, 0.6
    coin = InverseOption().greeks(FORWARD, strike, tau, vol, RATE, CP.CALL).delta
    quote = Black76().greeks(FORWARD, strike, tau, vol, RATE, CP.CALL).delta
    assert quote > 0.7
    assert coin < 1e-4


def test_put_greeks_follow_from_parity() -> None:
    """Parity is 1 - K/F, so vega is shared and the deltas differ by K/F^2."""
    model = InverseOption()
    strike = 70000.0
    call = model.greeks(FORWARD, strike, 0.5, 0.7, RATE, CP.CALL)
    put = model.greeks(FORWARD, strike, 0.5, 0.7, RATE, CP.PUT)
    assert call.vega == pytest.approx(put.vega, rel=1e-15)
    assert call.theta == pytest.approx(put.theta, rel=1e-15)
    assert call.delta - put.delta == pytest.approx(strike / FORWARD**2, rel=1e-14)
