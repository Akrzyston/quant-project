"""Implied forward against the traded future, and the basis between them.

Deribit's own chain display converts USD bid/ask off the index, not the
forward. A large basis here against a small index-versus-future gap is that
mistake's signature.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import data, state
from app.registry import Slot, panel
from voltk.forward import implied_forward_curve
from voltk.marketdata import MarketDataError
from voltk.universe import UniverseError


@panel(
    key="forward_curve",
    title="Forward Curve",
    slot=Slot.MAIN,
    order=17,
    milestone="M2",
    caption="The forward from put-call parity, not the index Deribit's own chain display quietly substitutes for it. Pricing off the index instead skews every strike's moneyness the same direction.",
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
    curve = implied_forward_curve(universe, data.marks_for(currency), currency=currency)
    if not curve:
        st.caption("No two-sided marks; nothing to imply a forward from.")
        return

    expiries = [point.expiry for point in curve]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=expiries, y=[point.traded_future for point in curve],
            name="Traded future", mode="lines+markers",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=expiries, y=[point.forward for point in curve],
            name="Parity-implied forward", mode="lines+markers",
        )
    )
    try:
        figure.add_hline(y=universe.index(currency), line_dash="dash", annotation_text="index")
    except UniverseError:
        pass
    figure.update_layout(
        xaxis_title="Expiry",
        yaxis_title="Forward (quote currency)",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(figure, width="stretch")

    st.dataframe(
        [
            {
                "Expiry": point.expiry.strftime("%Y-%m-%d"),
                "Traded future (quote currency)": round(point.traded_future, 2),
                "Implied forward (quote currency)": round(point.forward, 2),
                "Basis (quote currency)": round(point.basis, 2),
                "Basis (bps)": round(point.basis_bps, 1),
                "Pairs used": point.pairs_used,
            }
            for point in curve
        ],
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "Deribit's own chain shows USD bid/ask off the index, not the forward. A "
        "large basis here against a small index-vs-future gap is that mistake's "
        "signature. When no futures are captured, traded future falls back to the "
        "index, so this comparison carries no signal."
    )
