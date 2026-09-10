"""Model selector, populated from the library registry."""

from __future__ import annotations

import streamlit as st

from app import state
from app.registry import Slot, panel
from voltk import models


@panel(
    key="model_selector",
    title="Model Selector",
    slot=Slot.LEFT_RAIL,
    order=20,
    milestone="M1",
    caption="The model pricing every panel to the right. Black-Scholes leads on purpose, so its mistakes on a coin-settled contract stay visible instead of hidden.",
)
def render() -> None:
    s = state.get()
    specs = models.all_models()
    by_key = {spec.key: spec for spec in specs}
    keys = list(by_key)

    index = keys.index(s.model_key) if s.model_key in keys else 0
    s.model_key = st.selectbox(
        "Model",
        keys,
        index=index,
        format_func=lambda k: by_key[k].display_name,
        key="model_key_select",
    )

    spec = by_key[s.model_key]
    st.caption(f"{spec.underlying.value}-based · {spec.vol_convention} vol")
    st.caption(spec.rationale)
    if not spec.implemented:
        st.info(f"Not implemented yet. Scheduled for {spec.milestone}.")
