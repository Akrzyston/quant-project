"""Structured shock ladder: level (`a`), skew (`rho`) and curvature
(`sigma`) bumps to the fitted SVI slice, not a parallel vol shift. Not a
true historical PCA (no bulk historical-chain endpoint exists), so this is
model-implied, with only the level shock's magnitude sized from DVOL's
realized vol-of-vol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Sequence

from voltk.marketdata.parse import DvolSeries
from voltk.surface import RHO_EPS, SIGMA_FLOOR, SVISlice, log_moneyness


class ShockFactor(StrEnum):
    LEVEL = "level"
    SKEW = "skew"
    CURVATURE = "curvature"


class ShockError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ShockDefinition:
    factor: ShockFactor
    magnitude: float
    label: str


def shock_slice(slice_: SVISlice, definition: ShockDefinition) -> SVISlice:
    """A direct SVI-parameter perturbation of an already-fitted slice -- no
    re-calibration. Clamped to the same validity box calibrate_svi_slice fits
    within (|rho|<1, sigma>0), and `a` is lifted if needed to keep the
    non-negative-variance constraint a+b*sigma*sqrt(1-rho^2)>=0 satisfied
    everywhere, so a shock is always applicable rather than raising.
    """
    a, b, rho, m, sigma = slice_.a, slice_.b, slice_.rho, slice_.m, slice_.sigma

    if definition.factor is ShockFactor.LEVEL:
        a = a + definition.magnitude
    elif definition.factor is ShockFactor.SKEW:
        rho = min(max(rho + definition.magnitude, -1.0 + RHO_EPS), 1.0 - RHO_EPS)
    elif definition.factor is ShockFactor.CURVATURE:
        sigma = max(sigma + definition.magnitude, SIGMA_FLOOR)
    else:
        raise ShockError(f"Unknown shock factor {definition.factor!r}.")

    vertex = a + b * sigma * math.sqrt(max(1.0 - rho * rho, 0.0))
    if vertex < 0.0:
        a -= vertex

    return replace(slice_, a=a, b=b, rho=rho, m=m, sigma=sigma)


@dataclass(frozen=True, slots=True)
class LadderRung:
    definition: ShockDefinition
    base_vol: float
    shocked_vol: float
    base_price: float
    shocked_price: float
    pnl: float


@dataclass(frozen=True, slots=True)
class ShockLadder:
    currency: str
    expiry: datetime
    strike: float
    rungs: tuple[LadderRung, ...]


def build_ladder(
    slice_: SVISlice,
    model,
    forward: float,
    strike: float,
    rate: float,
    cp,
    *,
    level_magnitudes: Sequence[float],
    skew_magnitudes: Sequence[float] = (-0.10, 0.10),
    curvature_magnitudes: Sequence[float] = (-0.05, 0.05),
) -> ShockLadder:
    k = log_moneyness(strike, forward)
    tau = slice_.tau
    base_vol = slice_.vol(k)
    base_price = model.price(forward, strike, tau, base_vol, rate, cp)

    definitions = (
        [ShockDefinition(ShockFactor.LEVEL, m, f"level {m:+.4f}") for m in level_magnitudes]
        + [ShockDefinition(ShockFactor.SKEW, m, f"skew {m:+.2f}") for m in skew_magnitudes]
        + [
            ShockDefinition(ShockFactor.CURVATURE, m, f"curvature {m:+.2f}")
            for m in curvature_magnitudes
        ]
    )

    rungs = []
    for definition in definitions:
        shocked = shock_slice(slice_, definition)
        shocked_vol = shocked.vol(k)
        shocked_price = model.price(forward, strike, tau, shocked_vol, rate, cp)
        rungs.append(
            LadderRung(
                definition=definition,
                base_vol=base_vol,
                shocked_vol=shocked_vol,
                base_price=base_price,
                shocked_price=shocked_price,
                pnl=shocked_price - base_price,
            )
        )
    return ShockLadder(currency=slice_.currency, expiry=slice_.expiry, strike=strike, rungs=tuple(rungs))


def level_shock_magnitudes_from_dvol(
    dvol_series: DvolSeries | None,
    base_level: float,
    *,
    sigma_multiples: tuple[float, ...] = (-2.0, -1.0, 1.0, 2.0),
) -> tuple[float, ...]:
    """Sizes the level shock from realized vol-of-vol of DVOL closes: since
    w~vol^2, a fractional vol change `x` is roughly `2x` in total variance,
    so the bump to `a` is `base_level * 2 * stdev * multiple`.
    """
    if dvol_series is None or len(dvol_series.points) < 2:
        return ()
    closes = [p.close for p in dvol_series.points]
    log_returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]
    if len(log_returns) < 2:
        return ()
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    stdev = math.sqrt(variance)
    return tuple(base_level * 2.0 * stdev * multiple for multiple in sigma_multiples)
