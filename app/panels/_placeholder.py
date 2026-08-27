"""Placeholder for panels whose milestone has not landed."""

from __future__ import annotations

import streamlit as st


def stub(milestone: str, will_show: list[str], needs: str = "") -> None:
    st.caption(f"Arrives at {milestone}.")
    for item in will_show:
        st.markdown(f"- {item}")
    if needs:
        st.caption(needs)
