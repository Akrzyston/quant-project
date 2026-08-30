"""Skew-adjusted delta: Delta_eff = Delta_flat + Vega * dsigma/d(underlying).
Zero under sticky-strike (the smile is pinned to absolute strikes by
definition); real under sticky-delta. M6 measures empirically which regime
holds; this just computes both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from voltk.models.base import CP
from voltk.surface import Surface, log_moneyness


@dataclass(frozen=True, slots=True)
class SkewAdjustedDelta:
    flat_delta: float
    sticky_strike_delta: float
    sticky_delta_delta: float
    sticky_delta_adjustment: float


def skew_adjusted_delta(
    model, surface: Surface, strike: float, forward: float, tau: float, rate: float, cp: CP
) -> SkewAdjustedDelta:
    """Routes to the spot- or forward-flavoured sensitivity depending on
    model.underlying_is_forward, so the adjustment stays dimensionally
    consistent with the model's own delta -- mixing a spot sensitivity into
    a forward Greek (or vice versa) would silently misstate the correction.
    """
    k = log_moneyness(strike, forward)
    vol = surface.vol(k, tau)
    underlying = forward if model.underlying_is_forward else forward * math.exp(-rate * tau)
    flat = model.greeks(underlying, strike, tau, vol, rate, cp)

    if model.underlying_is_forward:
        sensitivity = surface.dvol_dforward_sticky_delta(strike, forward, tau)
    else:
        sensitivity = surface.dvol_dspot_sticky_delta(strike, forward, tau, rate)

    adjustment = flat.vega * sensitivity
    return SkewAdjustedDelta(
        flat_delta=flat.delta,
        sticky_strike_delta=flat.delta,
        sticky_delta_delta=flat.delta + adjustment,
        sticky_delta_adjustment=adjustment,
    )
