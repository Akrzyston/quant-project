"""Mock orderbook."""

from __future__ import annotations

import streamlit as st

from app import state
from app.panels._placeholder import stub
from app.registry import Slot, panel


@panel(
    key="mock_orderbook",
    title="Mock Orderbook",
    slot=Slot.QUOTER_MAIN,
    order=10,
    milestone="M7",
)
def render() -> None:
    if not state.get().instrument_name:
        st.caption("Select a contract to see its book.")
    stub(
        "M7",
        [
            "Bid levels descending and ask levels ascending",
            "Model mid overlaid on the book",
            "Generated quotes against resting size",
        ],
    )
