"""Vega bucketed by surface control point.

SVI has no literal spline knots, so "control point" means one of the market
smile points the slice was calibrated against, bucketed by rank in
log-moneyness. Bump only that bucket, refit, reprice, read off the
sensitivity. Bucketed vegas sum only approximately to parallel vega -- exact
for a uniform bump (absorbed entirely by `a`), approximate for a subset,
where the residual is the refit's real nonlinearity in the other parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Sequence

import numpy as np

from voltk.surface import SVISlice, SmilePoint, calibrate_svi_slice, log_moneyness

DEFAULT_N_BUCKETS = 3
DEFAULT_DW = 1e-5
_LABELS_3 = ("wing (low k)", "center", "wing (high k)")


class VegaBucketError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VegaBucket:
    label: str
    k_range: tuple[float, float]
    points_used: int
    vega: float


@dataclass(frozen=True, slots=True)
class BucketedVega:
    currency: str
    expiry: datetime
    strike: float
    buckets: tuple[VegaBucket, ...]
    total_bucketed: float
    parallel_vega: float

    @property
    def reconciliation_error(self) -> float:
        floor = 1e-12
        return abs(self.total_bucketed - self.parallel_vega) / max(abs(self.parallel_vega), floor)


def _bumped_price(
    points: Sequence[SmilePoint], indices: Sequence[int], dw: float,
    model, forward: float, strike: float, tau: float, rate: float, cp,
    *, currency: str, expiry: datetime,
) -> float:
    index_set = set(indices)
    bumped = tuple(
        replace(p, total_variance=p.total_variance + dw) if i in index_set else p
        for i, p in enumerate(points)
    )
    fit = calibrate_svi_slice(bumped, currency=currency, expiry=expiry, tau=tau)
    return model.price(forward, strike, tau, fit.vol(log_moneyness(strike, forward)), rate, cp)


def bucket_vega(
    points: Sequence[SmilePoint],
    base_slice: SVISlice,
    model,
    forward: float,
    strike: float,
    tau: float,
    rate: float,
    cp,
    *,
    currency: str,
    expiry: datetime,
    n_buckets: int = DEFAULT_N_BUCKETS,
    dw: float = DEFAULT_DW,
) -> BucketedVega:
    """`base_slice` is the already-fitted slice from calibrate_svi_slice on
    `points`, unbumped -- reused rather than recomputed, so the base price is
    consistent with whatever the caller is already displaying.
    """
    identified = [p for p in points if p.identified]
    if len(identified) < n_buckets:
        raise VegaBucketError(
            f"{currency} {expiry}: only {len(identified)} identified points, need "
            f"at least {n_buckets} to fill {n_buckets} buckets."
        )
    ordered = sorted(identified, key=lambda p: p.k)

    base_k = log_moneyness(strike, forward)
    base_price = model.price(forward, strike, tau, base_slice.vol(base_k), rate, cp)

    groups = [g.tolist() for g in np.array_split(np.arange(len(ordered)), n_buckets)]
    labels = _LABELS_3 if n_buckets == 3 else tuple(f"bucket {i + 1}" for i in range(n_buckets))

    buckets: list[VegaBucket] = []
    for label, idx_group in zip(labels, groups):
        if not idx_group:
            buckets.append(VegaBucket(label=label, k_range=(0.0, 0.0), points_used=0, vega=0.0))
            continue
        price = _bumped_price(
            ordered, idx_group, dw, model, forward, strike, tau, rate, cp,
            currency=currency, expiry=expiry,
        )
        ks = [ordered[i].k for i in idx_group]
        buckets.append(
            VegaBucket(
                label=label, k_range=(min(ks), max(ks)), points_used=len(idx_group),
                vega=(price - base_price) / dw,
            )
        )

    parallel_price = _bumped_price(
        ordered, range(len(ordered)), dw, model, forward, strike, tau, rate, cp,
        currency=currency, expiry=expiry,
    )

    return BucketedVega(
        currency=currency,
        expiry=expiry,
        strike=strike,
        buckets=tuple(buckets),
        total_bucketed=sum(b.vega for b in buckets),
        parallel_vega=(parallel_price - base_price) / dw,
    )
