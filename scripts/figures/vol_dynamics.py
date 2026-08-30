"""M6: implied (DVOL) vs realized vol, and the variance risk premium.

Live-only by necessity, not by oversight: DVOL and realized-vol history
have no snapshot/replay concept anywhere in this project (data.dvol_for and
data.historical_vol_for are both live-only in the dashboard too), so this
is the one figure make all cannot reproduce identically on every run --
only the same live discipline every other DVOL/realized-vol consumer here
already follows.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import matplotlib.pyplot as plt

from common import load_universe_and_marks, savefig
from voltk.marketdata import parse
from voltk.marketdata.deribit import DeribitClient
from voltk.variance_premium import summarise_premium, variance_risk_premium

LOOKBACK_DAYS = 14


def main() -> None:
    universe, _ = load_universe_and_marks()
    currency = universe.spec.currencies[0]
    client = DeribitClient()

    end = datetime.now(UTC)
    start = end - timedelta(days=LOOKBACK_DAYS)
    dvol = parse.dvol_series(client.fetch_dvol(currency, start=start, end=end, resolution="3600"), currency=currency)
    realized = parse.historical_volatility_series(client.fetch_historical_volatility(currency), currency=currency)
    if not dvol.points or not realized.points:
        print("DVOL or realized-vol history unavailable; skipping.")
        return

    premium_points = variance_risk_premium(dvol.points, realized.points)
    summary = summarise_premium(premium_points)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[2, 1])
    ax1.plot([p.timestamp for p in dvol.points], [p.close for p in dvol.points], label="DVOL (implied)")
    ax1.plot([p.timestamp for p in realized.points], [p.value for p in realized.points], label="Realized")
    ax1.set_ylabel("Annualized vol")
    ax1.set_title(f"{currency}: implied vs realized, {LOOKBACK_DAYS}d trailing")
    ax1.legend()

    ax2.plot([p.timestamp for p in premium_points], [p.premium for p in premium_points], color="tab:purple")
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Premium")
    if summary:
        ax2.set_title(f"mean {summary.mean_premium:+.2%}, positive {summary.fraction_positive:.0%} of window")
    fig.autofmt_xdate()

    savefig(fig, "vol_dynamics")


if __name__ == "__main__":
    main()
