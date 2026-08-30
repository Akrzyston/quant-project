"""M4: skew-adjusted delta across strikes, both sticky regimes against flat
delta -- sticky-strike collapses to exactly flat delta by definition, so
any daylight in this chart is the sticky-delta correction alone.

Uses the coin-settled inverse model, not a quote-settled one: this is
supposed to be the delta of a real Deribit position, and M4's whole finding
is that the coin-settled delta isn't a units conversion away from the
quote-settled one -- pricing this with Black76 would silently reintroduce
exactly the mistake that milestone exists to catch.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from common import load_universe_and_marks, savefig
from voltk.forward import implied_forward_curve
from voltk.models.base import CP
from voltk.models.inverse import InverseOption
from voltk.skew import skew_adjusted_delta
from voltk.surface import SurfaceError, Surface, calibrate_svi_slice, smile_points

_SECONDS_PER_YEAR = 365.0 * 24 * 3600
MODEL = InverseOption()


def main() -> None:
    universe, marks = load_universe_and_marks()
    currency = universe.spec.currencies[0]
    curve = implied_forward_curve(universe, marks, currency=currency)
    if not curve:
        print("No implied forward curve; skipping.")
        return
    forward_point = sorted(curve, key=lambda f: f.expiry)[len(curve) // 2]

    tau = (forward_point.expiry - universe.as_of).total_seconds() / _SECONDS_PER_YEAR
    points = smile_points(universe, marks, forward_point)
    try:
        fit = calibrate_svi_slice(points, currency=currency, expiry=forward_point.expiry, tau=tau)
    except SurfaceError as exc:
        print(f"Could not fit a slice: {exc}; skipping.")
        return
    surface = Surface(slices=(fit,))

    strikes = sorted(universe.strikes(currency, forward_point.expiry))
    rate = universe.implied_rate(currency, forward_point.expiry)

    flat, sticky_delta = [], []
    for strike in strikes:
        cp = CP.CALL if strike >= forward_point.forward else CP.PUT
        adj = skew_adjusted_delta(MODEL, surface, strike, forward_point.forward, tau, rate, cp)
        flat.append(adj.flat_delta)
        sticky_delta.append(adj.sticky_delta_delta)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(strikes, flat, "o-", label="Flat delta (= sticky-strike)")
    ax.plot(strikes, sticky_delta, "s--", label="Sticky-delta adjusted")
    ax.axvline(forward_point.forward, color="gray", linestyle=":", linewidth=1, label="Forward")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Delta (coin)")
    ax.set_title(
        f"{currency} {forward_point.expiry:%d %b %Y}: skew-adjusted delta, "
        f"{universe.as_of:%Y-%m-%d %H:%M UTC}"
    )
    ax.legend()
    savefig(fig, "skew_delta")


if __name__ == "__main__":
    main()
