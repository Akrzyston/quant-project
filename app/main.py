"""Dashboard entry point.

Owns the layout regions and nothing else. Panel content comes from the
registry, so adding a milestone does not touch this file. The page opens on
the Strategy tab (M8, Slot.FEATURED) -- everything else lives one tab over
under Pricing Engine, unchanged internally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app import data, state  # noqa: E402
from app.registry import Slot, discover, panels_for  # noqa: E402
from voltk.marketdata import MarketDataError  # noqa: E402

RAIL_RATIO = [0.32, 0.68]


def _render_global(slot: Slot) -> None:
    for spec in panels_for(slot):
        spec.render()


def _render_rail(slot: Slot) -> None:
    specs = panels_for(slot)
    if not specs:
        st.caption("No panels in this slot yet.")
        return
    for spec in specs:
        with st.container(border=True):
            st.markdown(f"**{spec.title}**")
            if spec.caption:
                st.caption(spec.caption)
            spec.render()


def _render_main(slot: Slot) -> None:
    specs = panels_for(slot)
    if not specs:
        st.caption("No panels in this slot yet.")
        return
    with st.container(border=True):
        if len(specs) == 1:
            spec = specs[0]
            st.markdown(f"### {spec.title}")
            if spec.caption:
                st.caption(spec.caption)
            spec.render()
            return
        for tab, spec in zip(st.tabs([spec.title for spec in specs]), specs):
            with tab:
                if spec.caption:
                    st.caption(spec.caption)
                spec.render()


def _engine_intro() -> None:
    st.markdown(
        "Everything here is what the Strategy tab's one trade depends on: "
        "a forward before a smile, a correctly fit smile before its Greeks "
        "mean anything, the right unit before a hedge ratio can be "
        "trusted. The rail on the left stays constant no matter which tab "
        "is open on the right: pick the pricing model here, read whatever "
        "contract is currently selected here, capture or replay a "
        "snapshot here."
    )
    st.markdown(
        "The tabs walk through that chain in order: capture, chain "
        "hygiene, delta in both settlement units, the forward from "
        "parity, a smile fit checked against Deribit's own index, the "
        "surface assembled from every expiry, a snapshot comparison, "
        "and vol dynamics against what actually realized. Each one is "
        "a specific, checkable claim about the same chain, not an "
        "arbitrary chart."
    )
    st.divider()


def _header() -> None:
    s = state.get()
    st.title("Dashboard")
    left, right = st.columns([0.75, 0.25])
    with left:
        if s.is_live:
            st.caption(f"{data.client().label} · live")
        else:
            st.caption(f"replay · {s.active_snapshot_id or 'no snapshot selected'}")
    with right:
        if st.button("Refresh data", width="stretch"):
            st.cache_data.clear()
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Volatility Dashboard", layout="wide")
    discover()
    state.get()
    _header()
    _render_global(Slot.GLOBAL)

    try:
        strategy_tab, engine_tab = st.tabs(["Strategy", "Pricing Engine"])
        with strategy_tab:
            _render_main(Slot.FEATURED)

        with engine_tab:
            _engine_intro()
            top_rail, top_main = st.columns(RAIL_RATIO, gap="medium")
            with top_rail:
                _render_rail(Slot.LEFT_RAIL)
            with top_main:
                _render_main(Slot.MAIN)

            st.divider()
            st.markdown(
                "The surface above becomes something tradeable below: a "
                "two-sided quote, and a simulated session showing what "
                "happens once the market actually trades against it."
            )

            bottom_rail, bottom_main = st.columns(RAIL_RATIO, gap="medium")
            with bottom_rail:
                _render_rail(Slot.QUOTER_RAIL)
            with bottom_main:
                _render_main(Slot.QUOTER_MAIN)
    except MarketDataError as exc:
        data.show_error(exc)


if __name__ == "__main__":
    main()
