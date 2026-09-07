"""Delta against strike, settlement currency versus quote currency.

The plot the milestone asks for. The two curves come from the same contract at
the same inputs; only the currency the risk is measured in differs.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import data, state
from app.registry import Slot, panel
from voltk.marketdata import MarketDataError
from voltk.models.base import CP
from voltk.models.black76 import Black76
from voltk.models.inverse import InverseOption
from voltk.universe import UniverseError

GRID_POINTS = 160


@panel(
    key="delta_profile",
    title="Delta vs Strike",
    slot=Slot.MAIN,
    order=15,
    milestone="M1",
)
def render() -> None:
    s = state.get()
    if not s.currencies:
        st.caption("Select a currency first.")
        return

    try:
        universe, _ = data.active_universe()
    except (MarketDataError, UniverseError) as exc:
        data.show_error(exc)
        return

    currency = s.currency or s.currencies[0]
    expiries = universe.expiries(currency)
    if not expiries:
        st.caption("No dated options captured.")
        return

    expiry = st.selectbox(
        "Expiry", expiries, format_func=lambda e: e.strftime("%d %b %Y"), key="delta_expiry"
    )
    vol = st.slider("Vol", 0.1, 2.0, 0.6, 0.05, key="delta_vol")

    forward = universe.forward_for(currency, expiry)
    rate = universe.implied_rate(currency, expiry)
    tau = max((expiry - universe.as_of).total_seconds() / (365 * 24 * 3600), 1e-6)

    inverse, black = InverseOption(), Black76()
    strikes = [forward * (0.05 + 3.45 * i / (GRID_POINTS - 1)) for i in range(GRID_POINTS)]
    coin = [inverse.greeks(forward, k, tau, vol, rate, CP.CALL).delta for k in strikes]
    quote = [black.greeks(forward, k, tau, vol, rate, CP.CALL).delta for k in strikes]

    # The two are in different units, so the coin curve is scaled onto the same
    # axis purely for shape comparison. The scale factor is shown below.
    peak = max(coin) or 1.0
    scaled = [value / peak for value in coin]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=strikes, y=quote, name="Quote currency (Black-76)", mode="lines")
    )
    figure.add_trace(
        go.Scatter(x=strikes, y=scaled, name="Settlement currency (scaled)", mode="lines")
    )
    figure.add_vline(x=forward, line_dash="dot", annotation_text="forward")
    figure.update_layout(
        xaxis_title="Strike",
        yaxis_title="Call delta",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(figure, width="stretch")

    st.caption(f"Settlement-currency delta scaled by {peak:.3e} to share the axis.")

    st.markdown(
        """
**Why the shapes differ.** Settlement-currency delta is `K N(d2) / F²`, which is
zero at both ends of the strike axis and humped in between. At low strikes `N(d2)`
tends to one while `K` tends to zero, so the product vanishes -- a deep in-the-money
coin-settled call is worth about one coin whatever the underlying does, and one coin
is worth one coin, so there is no exposure left to measure. At high strikes `N(d2)`
decays faster than `K` grows, the ordinary reason an out-of-the-money option has no
delta.

The quote-currency curve is the familiar monotone fall from one to zero.

The trap is where they disagree most. Hedging a coin-settled book with
quote-currency deltas is most wrong deep in the money, exactly where the
quote-currency number reads near one and feels safest.
"""
    )
