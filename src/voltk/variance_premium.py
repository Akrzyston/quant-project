"""Variance risk premium: implied vol minus realized vol, aligned by time.

DVOL and Deribit's published realized-vol series are independently sampled
(different endpoints, no shared timestamp grid) -- comparing them point by
point needs an explicit alignment rule, not a naive zip of two lists that
happen to be the same length.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from voltk.marketdata.parse import DvolPoint, RealizedVolPoint


@dataclass(frozen=True, slots=True)
class PremiumPoint:
    timestamp: datetime
    implied: float
    realized: float
    premium: float


def variance_risk_premium(
    dvol_points: Sequence[DvolPoint],
    realized_points: Sequence[RealizedVolPoint],
    *,
    max_gap_seconds: float = 3600.0,
) -> tuple[PremiumPoint, ...]:
    """Pairs each DVOL close with the nearest realized-vol observation within
    max_gap_seconds. A DVOL point with no realized observation close enough
    in time is dropped rather than matched to a stale one or interpolated
    across the gap -- a real data outage should show up as a shorter series,
    not a smoothed-over one.
    """
    if not dvol_points or not realized_points:
        return ()
    realized_sorted = sorted(realized_points, key=lambda p: p.timestamp)
    out: list[PremiumPoint] = []
    for dp in dvol_points:
        nearest = min(realized_sorted, key=lambda rp: abs((rp.timestamp - dp.timestamp).total_seconds()))
        gap = abs((nearest.timestamp - dp.timestamp).total_seconds())
        if gap > max_gap_seconds:
            continue
        out.append(PremiumPoint(timestamp=dp.timestamp, implied=dp.close, realized=nearest.value, premium=dp.close - nearest.value))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class PremiumSummary:
    n: int
    mean_premium: float
    fraction_positive: float
    inversions: tuple[datetime, ...]


def summarise_premium(points: Sequence[PremiumPoint]) -> PremiumSummary | None:
    """None on an empty series rather than a summary of nothing -- mirrors
    constant_maturity_variance's own None-on-empty convention.
    """
    if not points:
        return None
    n = len(points)
    return PremiumSummary(
        n=n,
        mean_premium=sum(p.premium for p in points) / n,
        fraction_positive=sum(1 for p in points if p.premium > 0) / n,
        inversions=tuple(p.timestamp for p in points if p.premium < 0),
    )
