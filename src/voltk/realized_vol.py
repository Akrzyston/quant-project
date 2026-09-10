"""Realized volatility computed independently from raw candles.

Deribit's own get_historical_volatility number is a black box (undocumented
sampling frequency, window, estimator), so this computes one from raw
perpetual candles to reconcile against instead of trusting blindly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Sequence

from voltk.marketdata.parse import Candle, RealizedVolPoint

_SECONDS_PER_YEAR = 365.0 * 24 * 3600


class RealizedVolError(ValueError):
    """Bad caller input, e.g. too few candles or a degenerate spacing."""


@dataclass(frozen=True, slots=True)
class RealizedVolResult:
    estimator: str
    window_start: datetime
    window_end: datetime
    periods_per_year: float
    n_observations: int
    value: float


def infer_periods_per_year(candles: Sequence[Candle]) -> float:
    """Annualisation factor from the ACTUAL median spacing between candle
    timestamps, not the resolution that was requested -- the venue can
    return a coarser or gappier grid than asked (thin trading, an outage),
    and trusting the request parameter over the data would silently
    mis-annualize the result.
    """
    if len(candles) < 2:
        raise RealizedVolError(f"Need >=2 candles to infer a sampling interval, got {len(candles)}.")
    diffs = sorted((b.timestamp - a.timestamp).total_seconds() for a, b in zip(candles, candles[1:]))
    median = diffs[len(diffs) // 2]
    if median <= 0:
        raise RealizedVolError(f"Non-positive median candle spacing: {median}s.")
    return _SECONDS_PER_YEAR / median


def close_to_close(candles: Sequence[Candle], *, periods_per_year: float) -> RealizedVolResult:
    """Classic close-to-close estimator: sample stdev of log returns, scaled
    by sqrt(periods_per_year). Uses ddof=1 (N-1), the convention behind every
    realized-vol figure a vendor publishes, this project's own DVOL
    cross-check included.
    """
    if len(candles) < 2:
        raise RealizedVolError(f"close_to_close needs >=2 candles, got {len(candles)}.")
    closes = [c.close for c in candles]
    returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    n = len(returns)
    if n < 2:
        raise RealizedVolError("close_to_close needs >=2 returns for a sample stdev.")
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return RealizedVolResult(
        estimator="close_to_close",
        window_start=candles[0].timestamp,
        window_end=candles[-1].timestamp,
        periods_per_year=periods_per_year,
        n_observations=n,
        value=math.sqrt(variance * periods_per_year),
    )


def parkinson(candles: Sequence[Candle], *, periods_per_year: float) -> RealizedVolResult:
    """Range estimator using each candle's own high/low, not just its close.
    More statistically efficient than close-to-close, at the cost of
    assuming pure diffusion (no jumps).

    sigma^2 = (1 / (4*ln2*N)) * sum(ln(H_i/L_i)^2)
    """
    usable = [c for c in candles if c.high > 0 and c.low > 0]
    if not usable:
        raise RealizedVolError("parkinson needs >=1 candle with a positive high and low.")
    n = len(usable)
    total = sum(math.log(c.high / c.low) ** 2 for c in usable)
    variance = total / (4.0 * math.log(2.0) * n) * periods_per_year
    return RealizedVolResult(
        estimator="parkinson",
        window_start=usable[0].timestamp,
        window_end=usable[-1].timestamp,
        periods_per_year=periods_per_year,
        n_observations=n,
        value=math.sqrt(variance),
    )


def rolling_realized_vol(
    candles: Sequence[Candle],
    *,
    window: timedelta,
    step: timedelta,
    estimator: Callable[..., RealizedVolResult] = close_to_close,
) -> tuple[RealizedVolPoint, ...]:
    """One realized-vol estimate per step, computed from the trailing
    `window` of candles ending at that point. Deribit's own realized-vol
    history endpoint caps at ~16 days regardless of what's requested; this
    reaches back as far as the candles do, computed independently -- the
    same principle this module already applies to the single-window
    estimators, extended to a series.
    """
    if not candles:
        return ()
    ordered = sorted(candles, key=lambda c: c.timestamp)
    t = ordered[0].timestamp + window
    end = ordered[-1].timestamp
    points: list[RealizedVolPoint] = []
    while t <= end:
        in_window = [c for c in ordered if t - window < c.timestamp <= t]
        if len(in_window) >= 2:
            try:
                periods = infer_periods_per_year(in_window)
                result = estimator(in_window, periods_per_year=periods)
            except RealizedVolError:
                pass  # too sparse for this estimator (e.g. close_to_close needs >=2 returns) -- skip, don't crash the series
            else:
                points.append(RealizedVolPoint(timestamp=t, value=result.value))
        t += step
    return tuple(points)


@dataclass(frozen=True, slots=True)
class Reconciliation:
    n: int
    mean_abs_diff: float


def reconcile(
    a: Sequence[RealizedVolPoint], b: Sequence[RealizedVolPoint], *, max_gap_seconds: float = 3600.0
) -> Reconciliation | None:
    """How closely two realized-vol series agree where they overlap in time,
    paired by nearest timestamp within max_gap_seconds. The same cross-check
    principle this project already runs against Deribit's DVOL (M3), applied
    here to check rolling_realized_vol against Deribit's own published
    realized-vol series over whatever window the venue's own capped history
    actually covers.
    """
    if not a or not b:
        return None
    b_sorted = sorted(b, key=lambda p: p.timestamp)
    diffs: list[float] = []
    for pa in a:
        nearest = min(b_sorted, key=lambda pb: abs((pb.timestamp - pa.timestamp).total_seconds()))
        gap = abs((nearest.timestamp - pa.timestamp).total_seconds())
        if gap <= max_gap_seconds:
            diffs.append(abs(pa.value - nearest.value))
    if not diffs:
        return None
    return Reconciliation(n=len(diffs), mean_abs_diff=sum(diffs) / len(diffs))
