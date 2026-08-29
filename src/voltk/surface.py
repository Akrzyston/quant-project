"""Volatility surface: raw SVI per expiry, stitched across strike and time.

Fit in total variance against log-moneyness, not raw implied vol against
strike. Total variance w = sigma^2 * tau is the quantity that is additive
under the calendar interpolation this module does (linear in T), and the
calendar no-arbitrage condition is exactly its monotonicity in T -- neither
statement holds for raw vol against strike, and strikes are not even
comparable across expiries the way log-moneyness is.

The structural prior is raw SVI (Gatheral): w(k) = a + b{rho(k-m) + sqrt((k-m)^2+sigma^2)}.
Chosen for three reasons. It is linear in (a, b*rho, b) for a fixed (m, sigma),
so most of the fit is well-conditioned. Its wings grow linearly in total
variance by construction, so vol grows like sqrt(|k|) rather than exploding --
this is what keeps deep-wing extrapolation from producing absurd vols without
any ad hoc capping. And Gatheral's own g(k) function gives an exact, checkable
butterfly-arbitrage test rather than a numerical proxy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from voltk.forward import ImpliedForward
from voltk.instruments import OptionType
from voltk.models.base import CP
from voltk.models.black76 import Black76
from voltk.models.solver import SolverError, implied_vol_detailed
from voltk.universe import Universe

_SECONDS_PER_YEAR = 365.0 * 24 * 3600
_REFERENCE_MODEL = Black76()

# Five free parameters need real over-determination, not an exact fit --
# below this the calibration would just interpolate whatever noise is there.
MIN_POINTS_FOR_SVI = 5
RHO_EPS = 1e-4
SIGMA_FLOOR = 1e-4
VERTEX_SLACK = 1e-9
CALIBRATION_SEEDS = 4

# Checking only the fitted k-range would miss exactly the wing-extrapolation
# failure mode this module exists to avoid.
WING_PAD_FRACTION = 0.20
BUTTERFLY_GRID_POINTS = 121
BUTTERFLY_TOLERANCE = -1e-9
CALENDAR_TOLERANCE = 1e-10


class SurfaceError(ValueError):
    """Too few usable points, no valid fit, or expiries out of order."""


def log_moneyness(strike: float, forward: float) -> float:
    return math.log(strike / forward)


@dataclass(frozen=True, slots=True)
class SmilePoint:
    strike: float
    k: float
    option_type: CP
    market_vol: float
    total_variance: float
    vega: float
    identified: bool


def smile_points(
    universe: Universe, marks: dict[str, float], implied_forward: ImpliedForward
) -> tuple[SmilePoint, ...]:
    """One point per strike with a usable OTM mark: put below the forward, call
    above, call at the forward itself (fallback to put if no call is marked).

    Implied vol always comes from a fixed Black76() reference, never the UI's
    selected model -- accepting a model parameter here would be exactly the
    kind of opening that later lets the point-pricing selector leak into a fit
    whose CBOE-style methodology assumes a stable, forward-based reference.
    Skips strikes the solver can't price (outside the model's no-arbitrage
    band) rather than raising; returns non-identified points too so callers
    can display them, but calibration filters to identified itself.
    """
    currency, expiry, forward = implied_forward.currency, implied_forward.expiry, implied_forward.forward
    tau = (expiry - universe.as_of).total_seconds() / _SECONDS_PER_YEAR
    if tau <= 0:
        return ()
    rate = universe.implied_rate(currency, expiry)

    calls, puts = {}, {}
    for inst in universe.options(currency):
        if inst.expiry != expiry or inst.strike is None:
            continue
        (calls if inst.option_type is OptionType.CALL else puts)[inst.strike] = inst

    points: list[SmilePoint] = []
    for strike in sorted(set(calls) | set(puts)):
        if strike < forward:
            inst, cp = puts.get(strike), CP.PUT
        elif strike > forward:
            inst, cp = calls.get(strike), CP.CALL
        else:
            inst, cp = calls.get(strike), CP.CALL
            if inst is None:
                inst, cp = puts.get(strike), CP.PUT
        if inst is None:
            continue
        price = marks.get(inst.name)
        if price is None:
            continue

        try:
            result = implied_vol_detailed(_REFERENCE_MODEL, price, forward, strike, tau, rate, cp)
        except SolverError:
            continue

        points.append(
            SmilePoint(
                strike=strike,
                k=log_moneyness(strike, forward),
                option_type=cp,
                market_vol=result.vol,
                total_variance=result.vol * result.vol * tau,
                vega=result.vega,
                identified=result.identified,
            )
        )
    return tuple(points)


def _svi_curve(params: np.ndarray, k: np.ndarray) -> np.ndarray:
    a, b, rho, m, sigma = params
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


@dataclass(frozen=True, slots=True)
class SVISlice:
    currency: str
    expiry: datetime
    tau: float
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    sse: float
    points_used: int
    points_available: int
    k_range: tuple[float, float]

    def total_variance(self, k: float) -> float:
        x = k - self.m
        return self.a + self.b * (self.rho * x + math.sqrt(x * x + self.sigma * self.sigma))

    def dw_dk(self, k: float) -> float:
        x = k - self.m
        z = math.sqrt(x * x + self.sigma * self.sigma)
        return self.b * (self.rho + x / z)

    def d2w_dk2(self, k: float) -> float:
        x = k - self.m
        z = math.sqrt(x * x + self.sigma * self.sigma)
        return self.b * self.sigma * self.sigma / (z * z * z)

    def vol(self, k: float) -> float:
        w = self.total_variance(k)
        return math.sqrt(w / self.tau) if w > 0.0 else 0.0


def calibrate_svi_slice(
    points: Sequence[SmilePoint], *, currency: str, expiry: datetime, tau: float
) -> SVISlice:
    """Raw SVI fit via scipy.optimize.least_squares, bounded to stay a valid
    slice (b>=0, |rho|<1, sigma>0). The joint non-negative-variance constraint
    a + b*sigma*sqrt(1-rho^2) >= 0 isn't a per-parameter box, so it's checked
    post-fit; a handful of re-seeded attempts (different initial sigma) covers
    the case where the first seed lands somewhere that violates it.
    """
    identified = [p for p in points if p.identified]
    if len(identified) < MIN_POINTS_FOR_SVI:
        raise SurfaceError(
            f"{currency} {expiry}: only {len(identified)} identified points, need "
            f"at least {MIN_POINTS_FOR_SVI} to fit SVI's 5 parameters."
        )

    k = np.array([p.k for p in identified], dtype=float)
    w = np.array([p.total_variance for p in identified], dtype=float)
    k_span = max(float(k.max() - k.min()), 1e-6)

    lower = [-10.0, 0.0, -1.0 + RHO_EPS, float(k.min()) - 2 * k_span, SIGMA_FLOOR]
    upper = [10.0, 10.0, 1.0 - RHO_EPS, float(k.max()) + 2 * k_span, 5 * k_span]

    m0 = float(k[np.argmin(w)])
    a0 = max(float(w.min()), SIGMA_FLOOR)
    b0 = max(float(w.max() - w.min()) / k_span, 1e-3)

    best: tuple[float, np.ndarray] | None = None
    for attempt in range(CALIBRATION_SEEDS):
        sigma0 = min(max(k_span * 0.1 * (attempt + 1), SIGMA_FLOOR), upper[4])
        x0 = np.clip([a0, b0, 0.0, m0, sigma0], lower, upper)

        result = least_squares(
            lambda params: _svi_curve(params, k) - w, x0, bounds=(lower, upper), max_nfev=2000
        )
        a, b, rho, m, sigma = (float(v) for v in result.x)
        vertex = a + b * sigma * math.sqrt(max(1.0 - rho * rho, 0.0))
        if vertex < -VERTEX_SLACK:
            continue
        sse = float(np.sum(result.fun**2))
        if best is None or sse < best[0]:
            best = (sse, result.x)

    if best is None:
        raise SurfaceError(
            f"{currency} {expiry}: no SVI fit over {CALIBRATION_SEEDS} attempts satisfied "
            "the non-negative-variance constraint a + b*sigma*sqrt(1-rho^2) >= 0."
        )

    sse, params = best
    a, b, rho, m, sigma = (float(v) for v in params)
    return SVISlice(
        currency=currency,
        expiry=expiry,
        tau=tau,
        a=a, b=b, rho=rho, m=m, sigma=sigma,
        sse=sse,
        points_used=len(identified),
        points_available=len(points),
        k_range=(float(k.min()), float(k.max())),
    )


@dataclass(frozen=True, slots=True)
class SurfaceCalibrationFailure:
    currency: str
    expiry: datetime
    reason: str


def calibrate_surface(
    universe: Universe, marks: dict[str, float], forwards: Sequence[ImpliedForward]
) -> tuple[tuple[SVISlice, ...], tuple[SurfaceCalibrationFailure, ...]]:
    """calibrate_svi_slice per expiry, catching failures per-expiry instead of
    aborting the whole surface -- same "counted, never silently discarded"
    discipline as voltk.chain.
    """
    slices: list[SVISlice] = []
    failures: list[SurfaceCalibrationFailure] = []
    for f in forwards:
        tau = (f.expiry - universe.as_of).total_seconds() / _SECONDS_PER_YEAR
        if tau <= 0:
            failures.append(SurfaceCalibrationFailure(f.currency, f.expiry, "expiry has passed"))
            continue
        points = smile_points(universe, marks, f)
        try:
            slices.append(calibrate_svi_slice(points, currency=f.currency, expiry=f.expiry, tau=tau))
        except SurfaceError as exc:
            failures.append(SurfaceCalibrationFailure(f.currency, f.expiry, str(exc)))
    return tuple(sorted(slices, key=lambda s: s.tau)), tuple(failures)


@dataclass(frozen=True, slots=True)
class ButterflyNode:
    k: float
    g: float

    @property
    def violated(self) -> bool:
        return self.g < BUTTERFLY_TOLERANCE


@dataclass(frozen=True, slots=True)
class ButterflyReport:
    currency: str
    expiry: datetime
    nodes: tuple[ButterflyNode, ...]

    @property
    def violations(self) -> tuple[ButterflyNode, ...]:
        return tuple(n for n in self.nodes if n.violated)

    @property
    def clean(self) -> bool:
        return not self.violations


def butterfly_check(slice_: SVISlice, *, grid_points: int = BUTTERFLY_GRID_POINTS) -> ButterflyReport:
    """Gatheral's g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2 >= 0."""
    lo, hi = slice_.k_range
    pad = WING_PAD_FRACTION * max(hi - lo, 1e-6)
    nodes = []
    for k in np.linspace(lo - pad, hi + pad, grid_points):
        k = float(k)
        w = slice_.total_variance(k)
        wp = slice_.dw_dk(k)
        wpp = slice_.d2w_dk2(k)
        g = (1.0 - k * wp / (2.0 * w)) ** 2 - (wp * wp / 4.0) * (1.0 / w + 0.25) + wpp / 2.0
        nodes.append(ButterflyNode(k=k, g=g))
    return ButterflyReport(currency=slice_.currency, expiry=slice_.expiry, nodes=tuple(nodes))


@dataclass(frozen=True, slots=True)
class CalendarNode:
    k: float
    w_near: float
    w_far: float

    @property
    def violated(self) -> bool:
        return self.w_far < self.w_near - CALENDAR_TOLERANCE


@dataclass(frozen=True, slots=True)
class CalendarReport:
    currency: str
    near_expiry: datetime
    far_expiry: datetime
    nodes: tuple[CalendarNode, ...]

    @property
    def violations(self) -> tuple[CalendarNode, ...]:
        return tuple(n for n in self.nodes if n.violated)

    @property
    def clean(self) -> bool:
        return not self.violations


def calendar_check(
    near: SVISlice, far: SVISlice, *, grid_points: int = BUTTERFLY_GRID_POINTS
) -> CalendarReport:
    """w must be non-decreasing in T at fixed k, checked on the shared,
    padded k-range of the two adjacent fitted slices.
    """
    if near.tau >= far.tau:
        raise SurfaceError(f"calendar_check needs near.tau < far.tau, got {near.tau} >= {far.tau}.")

    lo = min(near.k_range[0], far.k_range[0])
    hi = max(near.k_range[1], far.k_range[1])
    pad = WING_PAD_FRACTION * max(hi - lo, 1e-6)
    nodes = [
        CalendarNode(k=float(k), w_near=near.total_variance(float(k)), w_far=far.total_variance(float(k)))
        for k in np.linspace(lo - pad, hi + pad, grid_points)
    ]
    return CalendarReport(
        currency=near.currency, near_expiry=near.expiry, far_expiry=far.expiry, nodes=tuple(nodes)
    )


def calendar_reports(slices: Sequence[SVISlice]) -> tuple[CalendarReport, ...]:
    ordered = sorted(slices, key=lambda s: s.tau)
    return tuple(calendar_check(near, far) for near, far in zip(ordered, ordered[1:]))


@dataclass(frozen=True, slots=True)
class Surface:
    """A continuous strike/time surface stitched from independently-fit slices.

    Interpolates total variance linearly in T between the two bracketing
    slices at fixed k -- equivalent to the CBOE constant-maturity formula, and
    automatically consistent with calendar no-arbitrage whenever the two
    endpoints already are, since a linear interpolant cannot dip below the
    lower of two values it connects.

    Extrapolates outside the fitted maturity range by holding the
    instantaneous variance rate w(k,T)/T constant at the nearest slice, the
    same "flat rate" idea in the time direction that SVI's own linear-in-k
    wings already apply in the strike direction -- one documented rule,
    avoiding the wing/tail blowup failure mode both ways.
    """

    slices: tuple[SVISlice, ...]

    def __post_init__(self) -> None:
        if not self.slices:
            raise SurfaceError("Surface has no calibrated slices.")

    def total_variance(self, k: float, tau: float) -> float:
        slices = self.slices
        if tau <= slices[0].tau:
            near = slices[0]
            return near.total_variance(k) * tau / near.tau
        if tau >= slices[-1].tau:
            far = slices[-1]
            return far.total_variance(k) * tau / far.tau
        for left, right in zip(slices, slices[1:]):
            if left.tau <= tau <= right.tau:
                w_left, w_right = left.total_variance(k), right.total_variance(k)
                weight = (tau - left.tau) / (right.tau - left.tau)
                return w_left + weight * (w_right - w_left)
        raise SurfaceError(f"tau {tau} could not be bracketed within calibrated slices.")

    def vol(self, k: float, tau: float) -> float:
        w = self.total_variance(k, tau)
        return math.sqrt(w / tau) if w > 0.0 else 0.0

    def dvol_dforward_sticky_delta(
        self, strike: float, forward: float, tau: float, *, bump: float = 1e-3
    ) -> float:
        """d(vol)/d(forward) under sticky-delta: the fitted curve is held fixed
        in relative log-moneyness k = ln(K/F), so as the forward moves, vol at
        a fixed absolute strike moves with it -- the smile "follows the
        underlying." This is sticky-delta, not sticky-strike: genuine
        sticky-strike means the smile is pinned to absolute strikes, so vol at
        a fixed K does not move at all (see dvol_sticky_strike).
        """
        up, down = forward * (1.0 + bump), forward * (1.0 - bump)
        vol_up = self.vol(log_moneyness(strike, up), tau)
        vol_down = self.vol(log_moneyness(strike, down), tau)
        return (vol_up - vol_down) / (up - down)

    def dvol_dspot_sticky_delta(
        self, strike: float, forward: float, tau: float, rate: float, *, bump: float = 1e-3
    ) -> float:
        """d(vol)/d(spot) under sticky-delta, chain-ruled through the exact
        futures-basis relationship F = S*exp(rate*tau) this project already
        uses (Universe.implied_rate): dF/dS = exp(rate*tau), holding rate and
        tau fixed as spot moves.
        """
        return self.dvol_dforward_sticky_delta(strike, forward, tau, bump=bump) * math.exp(rate * tau)

    def dvol_sticky_strike(self, *args: object, **kwargs: object) -> float:
        """Identically zero: sticky-strike means the fitted sigma(K) curve
        itself does not move as spot or forward move, by definition. Same
        call shape as the other two so a caller can select a regime
        generically without branching on it.
        """
        return 0.0
