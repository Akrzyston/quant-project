"""Chain forensics: what got dropped before pricing, and why.

Every filter rule states the fraction of the chain it removed. Nothing is
silently discarded -- every dropped instrument stays visible below the funnel.
"""

from __future__ import annotations

import streamlit as st

from app import data, state
from app.registry import Slot, panel
from voltk.chain import filter_chain
from voltk.marketdata import MarketDataError
from voltk.universe import UniverseError


@panel(
    key="chain_view",
    title="Chain View",
    slot=Slot.MAIN,
    order=12,
    milestone="M2",
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
    report = filter_chain(
        universe.options(currency), data.marks_for(currency), data.quotes_for(currency)
    )

    st.caption(report.summary())
    st.dataframe(
        [
            {
                "Rule": outcome.name,
                "Description": outcome.description,
                "Dropped": outcome.dropped_count,
                "% of chain": f"{outcome.dropped_fraction:.1%}",
            }
            for outcome in report.outcomes
        ],
        hide_index=True,
        width="stretch",
    )

    with st.expander("Dropped instruments by rule"):
        if not any(outcome.dropped for outcome in report.outcomes):
            st.caption("Nothing dropped.")
        for outcome in report.outcomes:
            if outcome.dropped:
                st.markdown(f"**{outcome.name}** ({outcome.dropped_count})")
                st.caption(", ".join(outcome.dropped))
