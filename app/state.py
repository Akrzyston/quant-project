"""Cross-panel state.

One typed object under one Streamlit key. Panels read and write attributes and
never invent keys of their own, so the rail selection and the main panel cannot
drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import streamlit as st

_STATE_KEY = "voltk_state"


@dataclass
class QuoteParams:
    levels: int = 3
    width_bps: float = 50.0
    size: float = 1.0
    inventory_skew: float = 0.0


@dataclass
class SessionState:
    currencies: tuple[str, ...] = ()
    currency: str | None = None
    instrument_name: str | None = None
    expiry_label: str | None = None
    model_key: str | None = None

    data_mode: str = "live"
    active_snapshot_id: str | None = None

    quote: QuoteParams = field(default_factory=QuoteParams)
    scratch: dict[str, Any] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        return self.data_mode == "live"


def get() -> SessionState:
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = SessionState()
    return st.session_state[_STATE_KEY]


def reset() -> None:
    st.session_state[_STATE_KEY] = SessionState()
