"""Model-free variance index.

CBOE-style variance-swap replication: sum out-of-the-money option prices
across the strike ladder, weighted by 1/K^2 and the local strike spacing.
No model is fit; the number falls straight out of observed prices, which is
what makes it checkable against Deribit's own published DVOL index rather
than against anything this library computed.

Uses the forward derived from put-call parity (voltk.forward.ImpliedForward),
not Universe.forward_for. That parity-derived forward is exactly what CBOE's
own methodology uses (the smallest |C-P| strike), and it is the reason M2
exists: reading the forward off the index, the way Deribit's own chain
display does, produces a variance number that is quietly wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from voltk.forward import ImpliedForward
from voltk.instruments import Instrument, OptionType
from voltk.universe import Universe

_SECONDS_PER_YEAR = 365.0 * 24 * 3600

# Three usable strikes gives one interior central-difference node plus the two
# one-sided ends -- fewer than that and there is no curvature information in
# the sum, only noise.
MIN_STRIKES_FOR_VARIANCE = 3


class VarianceError(ValueError):
    """Bad caller input, e.g. a non-positive target maturity."""


@dataclass(frozen=True, slots=True)
class VarianceSlice:
    currency: str
    expiry: datetime
    tau: float
    forward: float
    rate: float
    k0: float
    k0_extrapolated: bool
    variance: float
    strikes_used: int
    contributions: tuple[tuple[float, float], ...]

    @property
    def valid(self) -> bool:
        return self.variance >= 0.0

    @property
    def vol(self) -> float | None:
        return math.sqrt(self.variance) if self.variance >= 0.0 else None


def _otm_ladder(
    universe: Universe, currency: str, expiry: datetime, forward: float, marks: Mapping[str, float]
) -> tuple[float, bool, list[tuple[float, float]]]:
    calls: dict[float, Instrument] = {}
    puts: dict[float, Instrument] = {}
    for inst in universe.options(currency):
        if inst.expiry != expiry or inst.strike is None:
            continue
        target = calls if inst.option_type is OptionType.CALL else puts
        target[inst.strike] = inst

    strikes = sorted(set(calls) | set(puts))
    if not strikes:
        return forward, False, []

    below_or_equal = [k for k in strikes if k <= forward]
    k0_extrapolated = not below_or_equal
    k0 = below_or_equal[-1] if below_or_equal else strikes[0]

    ladder: list[tuple[float, float]] = []
    for k in strikes:
        if k < k0:
            inst = puts.get(k)
            price = marks.get(inst.name) if inst else None
        elif k > k0:
            inst = calls.get(k)
            price = marks.get(inst.name) if inst else None
        else:
            call_inst, put_inst = calls.get(k), puts.get(k)
            call_price = marks.get(call_inst.name) if call_inst else None
            put_price = marks.get(put_inst.name) if put_inst else None
            prices = [p for p in (call_price, put_price) if p is not None]
            price = sum(prices) / len(prices) if prices else None
        if price is not None:
            ladder.append((k, price))

    return k0, k0_extrapolated, ladder


def model_free_variance(
    universe: Universe,
    marks: Mapping[str, float],
    implied_forward: ImpliedForward,
) -> VarianceSlice | None:
    """CBOE-style discretized model-free variance for one expiry.

    sigma^2(T) = (2/T) * sum_i [dK_i/K_i^2] * e^(rT) * Q(K_i) - (1/T)*(F/K0 - 1)^2

    Q(K) is the out-of-the-money price: put below K0, call above, the average
    of both at K0 itself. dK_i is the central difference between neighbouring
    usable strikes, one-sided at the two ends. Returns None, not raise, below
    MIN_STRIKES_FOR_VARIANCE usable strikes -- mirrors implied_forward_curve's
    per-expiry skip-not-crash rule.
    """
    currency, expiry, forward = implied_forward.currency, implied_forward.expiry, implied_forward.forward
    k0, k0_extrapolated, ladder = _otm_ladder(universe, currency, expiry, forward, marks)
    if len(ladder) < MIN_STRIKES_FOR_VARIANCE:
        return None

    tau = (expiry - universe.as_of).total_seconds() / _SECONDS_PER_YEAR
    if tau <= 0:
        return None
    rate = universe.implied_rate(currency, expiry)
    discount = math.exp(rate * tau)

    strikes = [k for k, _ in ladder]
    n = len(strikes)
    contributions: list[tuple[float, float]] = []
    total = 0.0
    for i, (k, price) in enumerate(ladder):
        if i == 0:
            delta_k = strikes[1] - strikes[0]
        elif i == n - 1:
            delta_k = strikes[-1] - strikes[-2]
        else:
            delta_k = (strikes[i + 1] - strikes[i - 1]) / 2.0
        term = (delta_k / (k * k)) * discount * price
        contributions.append((k, term))
        total += term

    variance = (2.0 / tau) * total - (1.0 / tau) * (forward / k0 - 1.0) ** 2

    return VarianceSlice(
        currency=currency,
        expiry=expiry,
        tau=tau,
        forward=forward,
        rate=rate,
        k0=k0,
        k0_extrapolated=k0_extrapolated,
        variance=variance,
        strikes_used=n,
        contributions=tuple(contributions),
    )


def variance_term_structure(
    universe: Universe,
    marks: Mapping[str, float],
    forwards: Sequence[ImpliedForward],
) -> tuple[VarianceSlice, ...]:
    """model_free_variance per ImpliedForward, silently skipping thin expiries."""
    slices = (model_free_variance(universe, marks, f) for f in forwards)
    return tuple(s for s in slices if s is not None)


@dataclass(frozen=True, slots=True)
class ConstantMaturityVariance:
    currency: str
    target_tau: float
    variance: float
    vol: float | None
    bracketed: bool
    extrapolated: bool
    single_slice: bool
    left: VarianceSlice | None
    right: VarianceSlice | None


def constant_maturity_variance(
    slices: Sequence[VarianceSlice],
    target_tau: float,
    *,
    currency: str,
) -> ConstantMaturityVariance | None:
    """CBOE interpolation-to-constant-maturity, in year-fractions.

    variance_target = [T1*v1*(T2-Tt) + T2*v2*(Tt-T1)] / (T2-T1) / Tt

    Linear interpolation of TOTAL variance (T*sigma^2) between the two
    nearest usable slices, then re-annualized. Degrades to extrapolation past
    either end (flagged, not raised) and to the single available slice,
    un-interpolated, when only one exists.
    """
    if target_tau <= 0:
        raise VarianceError(f"target_tau must be positive, got {target_tau!r}.")

    usable = sorted(
        (s for s in slices if s.currency == currency and s.valid),
        key=lambda s: s.tau,
    )
    if not usable:
        return None

    if len(usable) == 1:
        only = usable[0]
        return ConstantMaturityVariance(
            currency=currency,
            target_tau=target_tau,
            variance=only.variance,
            vol=only.vol,
            bracketed=False,
            extrapolated=False,
            single_slice=True,
            left=only,
            right=None,
        )

    left = max((s for s in usable if s.tau <= target_tau), key=lambda s: s.tau, default=None)
    right = min((s for s in usable if s.tau >= target_tau), key=lambda s: s.tau, default=None)
    if left is None:
        left, right = usable[0], usable[1]
    elif right is None:
        left, right = usable[-2], usable[-1]
    elif left is right:
        # target_tau lands exactly on one slice; still needs a pair to interpolate.
        idx = usable.index(left)
        left, right = (usable[idx - 1], usable[idx]) if idx > 0 else (usable[idx], usable[idx + 1])

    bracketed = left.tau <= target_tau <= right.tau
    span = right.tau - left.tau
    total_variance = (
        left.tau * left.variance * (right.tau - target_tau)
        + right.tau * right.variance * (target_tau - left.tau)
    ) / span
    variance = total_variance / target_tau

    return ConstantMaturityVariance(
        currency=currency,
        target_tau=target_tau,
        variance=variance,
        vol=math.sqrt(variance) if variance >= 0.0 else None,
        bracketed=bracketed,
        extrapolated=not bracketed,
        single_slice=False,
        left=left,
        right=right,
    )


@dataclass(frozen=True, slots=True)
class DvolComparison:
    currency: str
    model_free_vol: float | None
    dvol: float | None
    surface_atm_vol: float | None
    gap_vs_dvol: float | None
    gap_vs_dvol_bps: float | None
    gap_vs_surface: float | None
    gap_vs_surface_bps: float | None


def compare_to_dvol(
    cmv: ConstantMaturityVariance | None,
    dvol_close: float | None,
    surface_atm_vol: float | None,
    *,
    currency: str,
) -> DvolComparison:
    """Pure comparison: any missing input just leaves the corresponding gap None."""
    model_free_vol = cmv.vol if cmv is not None else None

    gap_vs_dvol = gap_vs_dvol_bps = None
    if model_free_vol is not None and dvol_close is not None and dvol_close > 0:
        gap_vs_dvol = model_free_vol - dvol_close
        gap_vs_dvol_bps = gap_vs_dvol / dvol_close * 10_000

    gap_vs_surface = gap_vs_surface_bps = None
    if model_free_vol is not None and surface_atm_vol is not None and surface_atm_vol > 0:
        gap_vs_surface = model_free_vol - surface_atm_vol
        gap_vs_surface_bps = gap_vs_surface / surface_atm_vol * 10_000

    return DvolComparison(
        currency=currency,
        model_free_vol=model_free_vol,
        dvol=dvol_close,
        surface_atm_vol=surface_atm_vol,
        gap_vs_dvol=gap_vs_dvol,
        gap_vs_dvol_bps=gap_vs_dvol_bps,
        gap_vs_surface=gap_vs_surface,
        gap_vs_surface_bps=gap_vs_surface_bps,
    )
