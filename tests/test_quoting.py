from __future__ import annotations

import math

import pytest

from voltk.quoting import (
    QuotingError,
    REHEDGE_INTERVAL_YEARS,
    classify_against_market,
    derive_width,
    quote_ladder,
    reservation_price,
)

UNDERLYING = 60_000.0
VOL = 0.6


def test_derive_width_sums_its_named_terms() -> None:
    width = derive_width(
        vega=100.0, gamma=0.001, fit_residual_vol=0.02, underlying=UNDERLYING, vol=VOL,
        market_half_spread=5.0,
    )
    assert width.half_width == pytest.approx(width.fit_term + width.gamma_term + width.liquidity_term)
    assert width.fit_term == pytest.approx(100.0 * 0.02)


def test_derive_width_respects_the_floor() -> None:
    width = derive_width(
        vega=0.0, gamma=0.0, fit_residual_vol=0.0, underlying=UNDERLYING, vol=VOL,
        market_half_spread=None, floor=1.5,
    )
    assert width.half_width == 1.5


def test_derive_width_gamma_term_matches_the_taylor_rehedge_cost() -> None:
    gamma = 0.002
    width = derive_width(
        vega=0.0, gamma=gamma, fit_residual_vol=0.0, underlying=UNDERLYING, vol=VOL,
        market_half_spread=None, gamma_coef=1.0,
    )
    hedge_move = UNDERLYING * VOL * math.sqrt(REHEDGE_INTERVAL_YEARS)
    assert width.gamma_term == pytest.approx(0.5 * gamma * hedge_move * hedge_move)


def test_reservation_price_shifts_away_from_positive_inventory() -> None:
    mid = 100.0
    shaded = reservation_price(mid, inventory=5.0, risk_aversion=0.1, vol=VOL, tau=0.5)
    assert shaded < mid
    assert reservation_price(mid, inventory=0.0, risk_aversion=0.1, vol=VOL, tau=0.5) == mid


def test_quote_ladder_widens_with_level() -> None:
    ladder = quote_ladder(mid=100.0, half_width=2.0, levels=3, size=1.0, level_growth=0.5)
    assert len(ladder) == 3
    widths = [lvl.ask - lvl.bid for lvl in ladder]
    assert widths == sorted(widths)
    assert widths[0] == pytest.approx(4.0)


def test_quote_ladder_rejects_zero_levels() -> None:
    with pytest.raises(QuotingError):
        quote_ladder(mid=100.0, half_width=1.0, levels=0, size=1.0)


@pytest.mark.parametrize(
    "our_bid,our_ask,expected",
    [
        (99.5, 100.5, "outside"),
        (99.9, 100.1, "inside"),
        (101.0, 102.0, "crossed"),
        (99.9, 100.5, "straddling"),
    ],
)
def test_classify_against_market(our_bid, our_ask, expected) -> None:
    # market is 99.8 / 100.2
    assert classify_against_market(our_bid, our_ask, 99.8, 100.2).value == expected
