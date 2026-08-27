"""Option details."""

from __future__ import annotations

import streamlit as st

from app import state
from app.panels._placeholder import stub
from app.registry import Slot, panel


@panel(
    key="option_details",
    title="Option Details",
    slot=Slot.LEFT_RAIL,
    order=30,
    milestone="M1",
    caption="Populates when an expiry is selected from the volatility surface.",
)
def render() -> None:
    if not state.get().instrument_name:
        st.caption("No contract selected.")
    stub(
        "M1",
        [
            "Contract metadata: multiplier, tick size, settlement currency",
            "Bid, ask and venue mark",
            "Model price and its difference against mark",
            "Implied vol from mid, with the round-trip residual",
        ],
    )
