"""Market data access for the dashboard.

Caching lives here rather than in voltk so the library stays framework-free.
This module also decides whether the active source is the live venue or a
snapshot replay; panels never construct a source themselves.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app import state
from voltk.capture import Capture, dossier
from voltk.capture import capture as run_capture
from voltk.instruments import Instrument, Kind
from voltk.marketdata import ALL_CURRENCIES, DeribitClient, MarketDataError
from voltk.marketdata import parse
from voltk.snapshots import (
    ReplaySource,
    SnapshotStore,
    rebuild_universe,
)
from voltk.universe import Universe, UniverseSpec, currencies_with

DISCOVERY_TTL = 3600
CAPTURE_TTL = 30


@st.cache_resource
def client() -> DeribitClient:
    return DeribitClient()


@st.cache_resource
def store() -> SnapshotStore:
    return SnapshotStore()


@st.cache_data(ttl=DISCOVERY_TTL, show_spinner="Discovering instruments...")
def _discovery() -> list[Instrument]:
    """One read of the whole option universe, used only to populate selectors."""
    return parse.instruments(client().fetch_instruments(ALL_CURRENCIES, Kind.OPTION))


def option_currencies() -> list[str]:
    return currencies_with(_discovery(), Kind.OPTION)


@st.cache_data(ttl=CAPTURE_TTL, show_spinner="Capturing chain, index and futures...")
def _live_capture(spec_key: str, currencies: tuple[str, ...]) -> Capture:
    return run_capture(client(), UniverseSpec.of(currencies))


def active_universe() -> tuple[Universe, dict[str, Any]]:
    """The universe the dashboard is currently showing, live or replayed."""
    s = state.get()
    if not s.currencies:
        raise MarketDataError("No currencies selected.")

    if s.is_live:
        spec = UniverseSpec.of(s.currencies)
        capture = _live_capture(spec.key(), spec.currencies)
        return capture.universe, {
            "source": capture.source_label,
            "quality": capture.quality,
            "capture": capture,
        }

    if not s.active_snapshot_id:
        raise MarketDataError("Snapshot mode is on but no snapshot is selected.")

    snapshot = store().load(s.active_snapshot_id)
    return rebuild_universe(snapshot), {
        "source": ReplaySource(snapshot).label,
        "quality": snapshot.meta.quality,
        "snapshot": snapshot,
    }


def capture_and_save(currencies: tuple[str, ...], note: str = "") -> str:
    spec = UniverseSpec.of(currencies)
    captured = run_capture(client(), spec, note=note)
    return store().save(captured)


def marks_for(currency: str) -> dict[str, float]:
    """Venue mark prices for the current source, keyed by instrument name."""
    s = state.get()
    if s.is_live:
        try:
            response = client().fetch_book_summary(currency, Kind.OPTION)
        except MarketDataError:
            return {}
        return dict(parse.mark_prices(response))

    if not s.active_snapshot_id:
        return {}
    snapshot = store().load(s.active_snapshot_id)
    component = f"summary:{currency}:{Kind.OPTION}"
    if component not in snapshot.responses:
        return {}
    return dict(parse.mark_prices(snapshot.responses[component]))


def quotes_for(currency: str) -> dict[str, parse.Quote]:
    """Venue bid/ask/mark for the current source, keyed by instrument name."""
    s = state.get()
    if s.is_live:
        try:
            response = client().fetch_book_summary(currency, Kind.OPTION)
        except MarketDataError:
            return {}
        return dict(parse.quotes(response))

    if not s.active_snapshot_id:
        return {}
    snapshot = store().load(s.active_snapshot_id)
    component = f"summary:{currency}:{Kind.OPTION}"
    if component not in snapshot.responses:
        return {}
    return dict(parse.quotes(snapshot.responses[component]))


def summarise(universe: Universe) -> dict[str, Any]:
    return dossier(universe)


def show_error(exc: Exception) -> None:
    st.error(str(exc))
    st.caption("Data is cached briefly. Use Refresh data in the header to clear it.")
