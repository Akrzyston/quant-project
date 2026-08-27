"""Quoting parameters. Bound to state now; the quoter arrives at M7."""

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
)
def render() -> None:
    q = state.get().quote
    q.levels = st.number_input("Levels", min_value=1, max_value=10, value=q.levels, step=1)
    q.width_bps = st.number_input(
        "Width (bps)", min_value=1.0, max_value=1000.0, value=q.width_bps, step=5.0
    )
    q.size = st.number_input("Size (contracts)", min_value=0.1, value=q.size, step=0.1)
    q.inventory_skew = st.slider("Inventory skew", -1.0, 1.0, q.inventory_skew, 0.05)
