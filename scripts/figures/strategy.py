"""M8: expected 1-day edge of a delta-hedged short straddle, as a function
of assumed realized vol -- crosses zero exactly at the vol the position was
struck at, the same relationship the strategy memo derives.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import load_universe_and_marks, savefig
from voltk.forward import implied_forward_curve
from voltk.models.base import CP
from voltk.models.inverse import InverseOption
from voltk.portfolio import Position, aggregate_greeks
from voltk.strategy import expected_daily_edge, hedge_delta, with_hedge
from voltk.surface import SurfaceError, calibrate_svi_slice, log_moneyness, smile_points

_SECONDS_PER_YEAR = 365.0 * 24 * 3600
MODEL = InverseOption()
VOL_RANGE_MULTIPLE = 2.0


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
    entry_vol = fit.vol(log_moneyness(strike, forward_point.forward))
    rate = universe.implied_rate(currency, forward_point.expiry)

    call_greeks = MODEL.greeks(forward_point.forward, strike, tau, entry_vol, rate, CP.CALL)
    put_greeks = MODEL.greeks(forward_point.forward, strike, tau, entry_vol, rate, CP.PUT)
    call_pos = Position(
        instrument_name="call", currency=currency, expiry=forward_point.expiry, strike=strike,
        cp=CP.CALL, size=-1.0, settles_in_base=True, coin_greeks=call_greeks, cash_greeks=call_greeks,
    )
    put_pos = Position(
        instrument_name="put", currency=currency, expiry=forward_point.expiry, strike=strike,
        cp=CP.PUT, size=-1.0, settles_in_base=True, coin_greeks=put_greeks, cash_greeks=put_greeks,
    )
    option_greeks = aggregate_greeks([call_pos, put_pos], use_cash=False)
    combined = with_hedge(option_greeks, hedge_delta(option_greeks.delta))

    realized_vols = np.linspace(entry_vol / VOL_RANGE_MULTIPLE, entry_vol * VOL_RANGE_MULTIPLE, 61)
    edges = [expected_daily_edge(combined.gamma, combined.theta, forward_point.forward, float(rv)).total for rv in realized_vols]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(realized_vols, edges)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(entry_vol, color="gray", linestyle=":", label=f"Entry vol ({entry_vol:.1%})")
    ax.set_xlabel("Assumed realized vol")
    ax.set_ylabel(f"Expected 1-day edge ({currency}, coin)")
    ax.set_title(f"{currency} {strike:g} straddle: expected edge vs realized vol assumption")
    ax.legend()
    savefig(fig, "strategy")


if __name__ == "__main__":
    main()
