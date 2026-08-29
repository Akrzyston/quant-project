"""Skew-adjusted delta: both sticky regimes, routed by the model's own
underlying convention.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from voltk.models.black76 import Black76
from voltk.models.black_scholes import BlackScholes
from voltk.models.base import CP
from voltk.skew import skew_adjusted_delta
from voltk.surface import SVISlice, Surface

EXPIRY = datetime(2026, 9, 19, 8, 0, tzinfo=UTC)
TAU = 30.0 / 365.0
FORWARD = 60_000.0
RATE = 0.05


def _sloped_surface() -> Surface:
    slice_ = SVISlice(
        currency="XBT", expiry=EXPIRY, tau=TAU,
        a=0.05, b=0.10, rho=-0.5, m=0.0, sigma=0.2,
        sse=0.0, points_used=10, points_available=10, k_range=(-1.0, 1.0),
    )
    return Surface(slices=(slice_,))


def _flat_surface() -> Surface:
    slice_ = SVISlice(
        currency="XBT", expiry=EXPIRY, tau=TAU,
        a=0.05, b=0.0, rho=0.0, m=0.0, sigma=0.2,
        sse=0.0, points_used=10, points_available=10, k_range=(-1.0, 1.0),
    )
    return Surface(slices=(slice_,))


@pytest.mark.parametrize("strike", [50_000.0, 60_000.0, 72_000.0])
@pytest.mark.parametrize("cp", list(CP))
def test_sticky_strike_delta_equals_flat_delta_exactly(strike, cp) -> None:
    result = skew_adjusted_delta(Black76(), _sloped_surface(), strike, FORWARD, TAU, RATE, cp)
    assert result.sticky_strike_delta == result.flat_delta


def test_sticky_delta_delta_differs_from_flat_on_a_sloped_smile() -> None:
    result = skew_adjusted_delta(Black76(), _sloped_surface(), 50_000.0, FORWARD, TAU, RATE, CP.CALL)
    assert result.sticky_delta_delta != pytest.approx(result.flat_delta)
    assert result.sticky_delta_adjustment != pytest.approx(0.0, abs=1e-12)


def test_sticky_delta_delta_equals_flat_delta_on_a_flat_smile() -> None:
    result = skew_adjusted_delta(Black76(), _flat_surface(), 50_000.0, FORWARD, TAU, RATE, CP.CALL)
    assert result.sticky_delta_delta == pytest.approx(result.flat_delta)
    assert result.sticky_delta_adjustment == pytest.approx(0.0, abs=1e-12)


def test_skew_adjustment_uses_matching_underlying_sensitivity_for_spot_vs_forward_models() -> None:
    """BlackScholes (spot) and Black76 (forward) route to different Surface
    sensitivity functions -- confirm both compute a real, nonzero adjustment
    on the same sloped smile rather than one silently falling back to zero
    from a mismatched sensitivity.
    """
    forward_result = skew_adjusted_delta(
        Black76(), _sloped_surface(), 50_000.0, FORWARD, TAU, RATE, CP.CALL
    )
    spot_result = skew_adjusted_delta(
        BlackScholes(), _sloped_surface(), 50_000.0, FORWARD, TAU, RATE, CP.CALL
    )

    assert forward_result.sticky_delta_adjustment != pytest.approx(0.0, abs=1e-12)
    assert spot_result.sticky_delta_adjustment != pytest.approx(0.0, abs=1e-12)
