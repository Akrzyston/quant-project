"""Dashboard entry point.

Owns the four regions of the layout and nothing else. Panel content comes from
the registry, so adding a milestone does not touch this file.
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

    try:
        top_rail, top_main = st.columns(RAIL_RATIO, gap="medium")
        with top_rail:
            _render_rail(Slot.LEFT_RAIL)
        with top_main:
            _render_main(Slot.MAIN)

        st.divider()

        bottom_rail, bottom_main = st.columns(RAIL_RATIO, gap="medium")
        with bottom_rail:
            _render_rail(Slot.QUOTER_RAIL)
        with bottom_main:
            _render_main(Slot.QUOTER_MAIN)
    except MarketDataError as exc:
        data.show_error(exc)


if __name__ == "__main__":
    main()
