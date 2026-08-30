"""M3: fitted SVI smiles across a short, a medium, and a long expiry.

Fit in total variance against log-moneyness, but shown here in implied vol
-- re-expressed for display, not what was fitted, the same distinction the
Smile dashboard panel already draws for the same readability reason: total
variance spans a very different scale at a week to expiry than at a year,
which would otherwise squash the short-dated curve to a flat line.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import load_universe_and_marks, savefig
from voltk.forward import implied_forward_curve
from voltk.surface import SurfaceError, calibrate_svi_slice, smile_points

_SECONDS_PER_YEAR = 365.0 * 24 * 3600
MIN_TAU_DAYS = 3.0


def main() -> None:
    universe, marks = load_universe_and_marks()
    currency = universe.spec.currencies[0]
    curve = sorted(implied_forward_curve(universe, marks, currency=currency), key=lambda f: f.expiry)
    curve = [
        f for f in curve
        if (f.expiry - universe.as_of).total_seconds() / _SECONDS_PER_YEAR * 365.0 >= MIN_TAU_DAYS
    ]
    if len(curve) < 2:
        print("Not enough expiries past the minimum tenor to pick a short/medium/long spread; skipping.")
        return

    picks = {curve[0], curve[len(curve) // 2], curve[-1]}

    fig, ax = plt.subplots(figsize=(8, 5))
    for forward_point in sorted(picks, key=lambda f: f.expiry):
        tau = (forward_point.expiry - universe.as_of).total_seconds() / _SECONDS_PER_YEAR
        points = smile_points(universe, marks, forward_point)
        identified = [p for p in points if p.identified]
        if len(identified) < 5:
            continue
        try:
            fit = calibrate_svi_slice(points, currency=currency, expiry=forward_point.expiry, tau=tau)
        except SurfaceError:
            continue

        k_grid = np.linspace(*fit.k_range, 121)
        label = f"{forward_point.expiry:%d %b %Y} ({tau * 365:.0f}d)"
        line = ax.plot(k_grid, [fit.vol(float(k)) for k in k_grid], label=label)[0]
        ax.scatter([p.k for p in identified], [p.market_vol for p in identified], color=line.get_color(), s=14)

    ax.set_xlabel("Log-moneyness k = ln(K/F)")
    ax.set_ylabel("Implied vol")
    ax.set_title(f"{currency}: fitted SVI smiles, {universe.as_of:%Y-%m-%d %H:%M UTC}")
    ax.legend()
    savefig(fig, "smile")


if __name__ == "__main__":
    main()
