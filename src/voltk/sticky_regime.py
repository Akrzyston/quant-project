"""Empirical sticky-strike vs sticky-delta test.

Surface already gives the theoretical vol sensitivity to spot under an
assumed sticky-delta regime (dvol_dspot_sticky_delta) and sticky-strike
(zero, by definition). This module measures what the market actually does:
invert one day of intraday option candles to implied vol at a fixed strike,
regress the vol change against the spot change, and report the empirical
slope alongside both predictions. Inverting a candle needs the forward AT
THAT TIMESTAMP, not the forward at capture time, since tau shrinks through
the session -- derived per timestamp via the same put-call parity identity
voltk.forward uses for a whole snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from voltk.forward import forward_from_parity
from voltk.marketdata.parse import CandleSeries
from voltk.models.base import CP
from voltk.models.black76 import Black76
from voltk.models.solver import SolverError, implied_vol_detailed

_SECONDS_PER_YEAR = 365.0 * 24 * 3600
_REFERENCE_MODEL = Black76()


class StickyRegimeError(ValueError):
    """Bad caller input, e.g. too few observations or a session with no spot movement."""


@dataclass(frozen=True, slots=True)
class VolObservation:
    timestamp: datetime
    spot: float
    forward: float
    implied_vol: float


def observations_from_candles(
    call_candles: CandleSeries,
    put_candles: CandleSeries,
    perpetual_candles: CandleSeries,
    *,
    strike: float,
    expiry: datetime,
) -> tuple[VolObservation, ...]:
    """One VolObservation per timestamp present in all three candle series --
    alignment is by exact timestamp match, not position, since a thinly
    traded side can be missing a tick the others have. tau is recomputed at
    every timestamp; rate=0 mirrors smile_points's reasoning for a
    coin-settled price already converted via `price * forward`. Points the
    solver can't identify (vega below floor) are dropped rather than fed
    into the regression as noise.
    """
    puts_by_time = {c.timestamp: c for c in put_candles.candles}
    perp_by_time = {c.timestamp: c for c in perpetual_candles.candles}

    observations: list[VolObservation] = []
    for call in call_candles.candles:
        put = puts_by_time.get(call.timestamp)
        perp = perp_by_time.get(call.timestamp)
        if put is None or perp is None:
            continue
        tau = (expiry - call.timestamp).total_seconds() / _SECONDS_PER_YEAR
        if tau <= 0:
            continue
        forward = forward_from_parity(strike, call.close, put.close, settles_in_base=True)
        if forward is None:
            continue
        price = call.close * forward
        try:
            result = implied_vol_detailed(_REFERENCE_MODEL, price, forward, strike, tau, 0.0, CP.CALL)
        except SolverError:
            continue
        if not result.identified:
            continue
        observations.append(
            VolObservation(timestamp=call.timestamp, spot=perp.close, forward=forward, implied_vol=result.vol)
        )
    observations.sort(key=lambda o: o.timestamp)
    return tuple(observations)


@dataclass(frozen=True, slots=True)
class RegressionResult:
    strike: float
    slope: float
    intercept: float
    r_squared: float
    n_observations: int
    sticky_strike_prediction: float
    sticky_delta_prediction: float


def regress_vol_on_spot(
    observations: Sequence[VolObservation], *, strike: float, sticky_delta_prediction: float
) -> RegressionResult:
    """OLS slope of vol change on spot change (first differences, not
    levels, so a shared session trend doesn't masquerade as the local
    sensitivity Delta_eff needs). sticky_strike_prediction is always 0.0;
    sticky_delta_prediction is whatever the caller supplies. Reports the
    comparison, doesn't declare a winner.
    """
    if len(observations) < 3:
        raise StickyRegimeError(f"Need >=3 observations to regress, got {len(observations)}.")
    ordered = sorted(observations, key=lambda o: o.timestamp)
    d_spot = [b.spot - a.spot for a, b in zip(ordered, ordered[1:])]
    d_vol = [b.implied_vol - a.implied_vol for a, b in zip(ordered, ordered[1:])]
    n = len(d_spot)
    mean_x, mean_y = sum(d_spot) / n, sum(d_vol) / n
    var_x = sum((x - mean_x) ** 2 for x in d_spot)
    if var_x == 0.0:
        raise StickyRegimeError("Spot did not move across the session; the regression is undefined.")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(d_spot, d_vol))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in d_vol)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(d_spot, d_vol))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return RegressionResult(
        strike=strike,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        n_observations=n,
        sticky_strike_prediction=0.0,
        sticky_delta_prediction=sticky_delta_prediction,
    )
