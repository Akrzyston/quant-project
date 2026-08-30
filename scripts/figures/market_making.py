"""M7: a simulated market-making session's P&L, decomposed into edge
captured, adverse selection, and vega P&L.

Live-only: the session walks real intraday perpetual candles, which (like
M6's DVOL/realized-vol history) have no snapshot/replay concept here.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import matplotlib.pyplot as plt

from common import load_universe_and_marks, savefig
from voltk.forward import implied_forward_curve
from voltk.marketdata import parse
from voltk.marketdata.deribit import DeribitClient
from voltk.market_making import simulate_session, SessionStep
from voltk.models.base import CP
from voltk.models.inverse import InverseOption
from voltk.quoting import derive_width
from voltk.surface import SurfaceError, calibrate_svi_slice, log_moneyness, smile_points

_SECONDS_PER_YEAR = 365.0 * 24 * 3600
MODEL = InverseOption()  # coin-settled, matching how Deribit actually quotes
WINDOW_HOURS = 12
RESOLUTION = "15"
COIN_WIDTH_FLOOR = 0.0001


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
    greeks = MODEL.greeks(forward_point.forward, strike, tau, vol, rate, CP.CALL)

    width = derive_width(
        vega=greeks.vega, gamma=greeks.gamma, fit_residual_vol=0.0,
        underlying=forward_point.forward, vol=vol, market_half_spread=None, floor=COIN_WIDTH_FLOOR,
    )

    client = DeribitClient()
    perpetual = next((i for i in universe.futures(currency) if i.expiry is None), None)
    if perpetual is None:
        print("No perpetual instrument; skipping.")
        return
    end = datetime.now(UTC)
    start = end - timedelta(hours=WINDOW_HOURS)
    candles = parse.candle_series(
        client.fetch_candles(perpetual.name, start=start, end=end, resolution=RESOLUTION),
        instrument_name=perpetual.name,
    )
    if len(candles.candles) < 3:
        print("Not enough intraday candles; skipping.")
        return

    steps = [
        SessionStep(
            timestamp=c.timestamp,
            underlying=c.close * math.exp(rate * max((forward_point.expiry - c.timestamp).total_seconds() / _SECONDS_PER_YEAR, 0.0)),
            tau=max((forward_point.expiry - c.timestamp).total_seconds() / _SECONDS_PER_YEAR, 0.0),
            vol=vol,
        )
        for c in candles.candles
    ]
    result = simulate_session(MODEL, steps, strike=strike, rate=rate, cp=CP.CALL, half_width=width.half_width, markout_steps=2)

    labels = ["Edge captured", "Adverse selection", "Vega P&L", "Theta P&L", "Residual"]
    values = [result.total_edge, result.total_adverse_selection, result.total_vega_pnl, result.total_theta_pnl, result.total_residual]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:green" if v >= 0 else "tab:red" for v in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(result.total_pnl, color="black", linestyle="--", linewidth=1, label=f"Total P&L ({result.total_pnl:+.4f})")
    ax.set_ylabel(f"{currency} (coin)")
    ax.set_title(
        f"{currency} {strike:g} call: simulated session, {len(result.fills)} fills "
        f"over trailing {WINDOW_HOURS}h"
    )
    ax.legend()
    savefig(fig, "market_making")


if __name__ == "__main__":
    main()
