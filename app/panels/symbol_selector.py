"""Symbol selector.

Multi-select rather than single, because a cross-currency view needs both chains
captured in the same window. Currencies come from discovered instruments, so the
list only ever holds underlyings that actually have options.
"""

from __future__ import annotations

import streamlit as st

from app import data, state
from app.registry import Slot, panel
from voltk.marketdata import MarketDataError


@panel(
    key="symbol_selector",
    title="Symbol Selector",
    slot=Slot.LEFT_RAIL,
    order=10,
    milestone="M0",
)
def render() -> None:
    s = state.get()

    try:
        available = data.option_currencies()
    except MarketDataError as exc:
        data.show_error(exc)
        return

    if not available:
        st.warning("The venue returned no live options. Use Refresh data.")
        return

    default = list(s.currencies) if s.currencies else available[:1]
    chosen = st.multiselect(
        "Currencies",
        available,
        default=[c for c in default if c in available],
        key="universe_currencies",
        help="Select more than one to capture them in a single window.",
    )
    if not chosen:
        st.caption("Select at least one currency.")
        s.currencies = ()
        return

    s.currencies = tuple(chosen)
    if s.currency not in s.currencies:
        s.currency = s.currencies[0]

    if len(s.currencies) > 1:
        s.currency = st.radio(
            "Focus",
            s.currencies,
            index=s.currencies.index(s.currency),
            horizontal=True,
            key="focus_currency",
        )
        st.caption("Cross-currency: all selected chains share one capture window.")
