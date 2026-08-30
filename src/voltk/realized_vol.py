"""Realized volatility computed independently from raw candles.

Deribit's own get_historical_volatility number is a black box (undocumented
sampling frequency, window, estimator), so this computes one from raw
perpetual candles to reconcile against instead of trusting blindly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from voltk.marketdata.parse import Candle

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
