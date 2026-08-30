"""M5: P&L attribution as a waterfall, for a stated scenario (spot +1%, vol
+2 points, one day of theta) applied to an ATM option from the fitted
surface. A stated scenario rather than two real snapshots so this figure
reproduces from a single capture, the same as every other figure here.

Coin-settled, like every other figure that prices a real Deribit position
here -- Black76 would price a quote-settled hypothetical instead, and the
whole point of this milestone's own coin/cash distinction is that the two
aren't interchangeable.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from common import load_universe_and_marks, savefig
from voltk.forward import implied_forward_curve
from voltk.models.base import CP
from voltk.models.inverse import InverseOption
from voltk.pnl import attribute_pnl
from voltk.surface import SurfaceError, calibrate_svi_slice, log_moneyness, smile_points

_SECONDS_PER_YEAR = 365.0 * 24 * 3600
MODEL = InverseOption()
SPOT_SHOCK = 0.01
VOL_SHOCK = 0.02
DAYS_ELAPSED = 1.0


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

    strike = min(universe.strikes(currency, forward_point.expiry), key=lambda k: abs(k - forward_point.forward))
    vol = fit.vol(log_moneyness(strike, forward_point.forward))
    rate = universe.implied_rate(currency, forward_point.expiry)

    price_a = MODEL.price(forward_point.forward, strike, tau, vol, rate, CP.CALL)
    greeks_a = MODEL.greeks(forward_point.forward, strike, tau, vol, rate, CP.CALL)

    forward_b = forward_point.forward * (1.0 + SPOT_SHOCK)
    vol_b = vol + VOL_SHOCK
    d_t = DAYS_ELAPSED / 365.0
    tau_b = max(tau - d_t, 0.0)
    price_b = MODEL.price(forward_b, strike, tau_b, vol_b, rate, CP.CALL)

    result = attribute_pnl(greeks_a, price_a, price_b, forward_b - forward_point.forward, vol_b - vol, d_t)

    labels = ["Delta", "Gamma", "Vega", "Theta", "Residual"]
    values = [result.delta_term, result.gamma_term, result.vega_term, result.theta_term, result.residual]
    cumulative = 0.0
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, value) in enumerate(zip(labels, values)):
        color = "tab:green" if value >= 0 else "tab:red"
        ax.bar(label, value, bottom=cumulative if value >= 0 else cumulative + value, color=color)
        cumulative += value
    ax.axhline(result.actual_pnl, color="black", linestyle="--", linewidth=1, label=f"Actual P&L ({result.actual_pnl:+.6f})")
    ax.set_ylabel(f"{currency} (coin)")
    ax.set_title(
        f"{currency} {strike:g} {forward_point.expiry:%d %b %Y} call: "
        f"P&L attribution, +{SPOT_SHOCK:.0%} spot / +{VOL_SHOCK:.0%} vol / {DAYS_ELAPSED:g}d"
    )
    ax.legend()
    savefig(fig, "pnl_attribution")


if __name__ == "__main__":
    main()
