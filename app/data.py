"""Market data access for the dashboard.

Caching lives here rather than in voltk so the library stays framework-free.
This module also decides whether the active source is the live venue or a
snapshot replay; panels never construct a source themselves.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
# DVOL is Deribit's own already-published index, not something this app
# derives -- it isn't drift/window-gated like a bracketed capture (30s), and
# it's only a periodic visual cross-check, so it doesn't need discovery's
# full hour of staleness tolerance either.
DVOL_TTL = 900


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


@st.cache_data(ttl=DVOL_TTL, show_spinner="Fetching DVOL history...")
def dvol_for(
    currency: str, lookback_hours: int = 48, resolution: str = "3600"
) -> parse.DvolSeries | None:
    """Deribit's own published DVOL, live-only and best-effort: unavailable
    is None, never a crash on the cross-check panels. now() is computed
    inside the cached body so the cache key stays (currency, lookback_hours,
    resolution) rather than a start/end pair that would shift every rerun.
    """
    end = datetime.now(UTC)
    start = end - timedelta(hours=lookback_hours)
    try:
        response = client().fetch_dvol(currency, start=start, end=end, resolution=resolution)
    except MarketDataError:
        return None
    return parse.dvol_series(response, currency=currency)


def summarise(universe: Universe) -> dict[str, Any]:
    return dossier(universe)


def show_error(exc: Exception) -> None:
    st.error(str(exc))
    st.caption("Data is cached briefly. Use Refresh data in the header to clear it.")
