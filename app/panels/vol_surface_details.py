"""Volatility surface. Registers into the main slot alongside the dossier."""

from __future__ import annotations

import streamlit as st

from app import state
from app.panels._placeholder import stub
from app.registry import Slot, panel


@panel(
    key="vol_surface_details",
    title="Volatility Surface",
    slot=Slot.MAIN,
    order=20,
    milestone="M3",
)
def render() -> None:
    s = state.get()
    st.caption(f"All active splines for {s.currency or 'the selected symbol'}.")
    stub(
        "M3",
        [
            "Interactive 3D surface of implied vol against strike and expiry",
            "Smile per expiry in delta terms, with a 2D strike view",
            "Click-through on an expiry that fills Option and Greeks Details",
            "Quality flags on nodes violating no-arbitrage bounds",
        ],
        needs="Requires the chain view and forward curve from M2.",
    )
