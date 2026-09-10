"""Quoting parameters: the stated inputs to the width formula, not a width
itself -- see voltk.quoting.derive_width.
"""

from __future__ import annotations

import streamlit as st

from app import state
from app.registry import Slot, panel


@panel(
    key="quoting_parameters",
    title="Mock Quoter",
    slot=Slot.QUOTER_RAIL,
    order=10,
    milestone="M7",
    caption="The inputs behind the quote width below, stated directly rather than hand-set.",
)
def render() -> None:
    q = state.get().quote
    q.levels = st.number_input("Levels", min_value=1, max_value=10, value=q.levels, step=1)
    q.size = st.number_input("Size (contracts)", min_value=0.1, value=q.size, step=0.1)
    q.level_growth = st.slider("Level growth", 0.0, 2.0, q.level_growth, 0.1)

    st.caption("Width formula coefficients")
    q.fit_coef = st.slider("Fit-residual weight", 0.0, 5.0, q.fit_coef, 0.1)
    q.gamma_coef = st.slider("Rehedge (gamma) weight", 0.0, 5.0, q.gamma_coef, 0.1)
    q.liquidity_coef = st.slider("Illiquidity weight", 0.0, 2.0, q.liquidity_coef, 0.1)
    q.floor_coin = st.number_input("Minimum half-width (coin)", min_value=0.0, value=q.floor_coin, step=0.0001, format="%.4f")

    st.caption("Inventory skew (Avellaneda-Stoikov)")
    q.risk_aversion = st.slider("Risk aversion", 0.0, 50.0, q.risk_aversion, 0.5)
