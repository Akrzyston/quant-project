"""Market data access for the dashboard.

Caching lives here rather than in voltk so the library stays framework-free.
This module also decides whether the active source is the live venue or a
snapshot replay; panels never construct a source themselves.
"""

from __future__ import annotations

import pickle
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import streamlit as st
from streamlit.runtime.caching.cache_errors import UnserializableReturnValueError

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
# DVOL is Deribit's own published index, a periodic cross-check rather than
# a drift/window-gated capture, so it doesn't need discovery's full hour.
DVOL_TTL = 900
HISTORICAL_VOL_TTL = 900  # same reasoning as DVOL_TTL
# Candles change faster than either published index but still don't need
# capture's 30s tightness -- these feed a reconciliation chart, not a live quote.
CANDLE_TTL = 300


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


def _find_unpicklable_path(obj: Any, path: str = "capture", seen: set[int] | None = None, depth: int = 0) -> str | None:
    """Walk a dataclass/dict/list/tuple tree and return a dotted path to the
    first value that fails to pickle on its own -- diagnostic only, for
    tracking down an UnserializableReturnValueError that hasn't reproduced
    outside a live session. Not called on any success path.
    """
    if seen is None:
        seen = set()
    if depth > 10 or id(obj) in seen:
        return None
    try:
        pickle.dumps(obj)
        return None
    except Exception:
        pass
    seen.add(id(obj))
    if is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            found = _find_unpicklable_path(getattr(obj, f.name), f"{path}.{f.name}", seen, depth + 1)
            if found:
                return found
        return path
    if isinstance(obj, dict):
        for k, v in obj.items():
            found = _find_unpicklable_path(v, f"{path}[{k!r}]", seen, depth + 1)
            if found:
                return found
        return path
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found = _find_unpicklable_path(v, f"{path}[{i}]", seen, depth + 1)
            if found:
                return found
        return path
    return f"{path} (leaf, type {type(obj).__name__})"


def active_universe() -> tuple[Universe, dict[str, Any]]:
    """The universe the dashboard is currently showing, live or replayed."""
    s = state.get()
    if not s.currencies:
        raise MarketDataError("No currencies selected.")

    if s.is_live:
        spec = UniverseSpec.of(s.currencies)
        try:
            capture = _live_capture(spec.key(), spec.currencies)
        except UnserializableReturnValueError:
            fresh = run_capture(client(), spec)
            bad_path = _find_unpicklable_path(fresh)
            raise MarketDataError(
                f"Live capture built but the cache couldn't pickle it (first bad field: {bad_path}). "
                "This is a diagnostic message, not the normal failure mode -- please report this path."
            ) from None
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


def universe_for_snapshot(snapshot_id: str) -> Universe:
    """A specific, explicitly-named snapshot's universe -- unlike
    active_universe(), not tied to the session's current replay selection.
    Needed to look at two snapshots (A and B) at once.
    """
    return rebuild_universe(store().load(snapshot_id))


def marks_for_snapshot(snapshot_id: str, currency: str) -> dict[str, float]:
    """Mark prices from a specific snapshot, independent of session state."""
    snapshot = store().load(snapshot_id)
    component = f"summary:{currency}:{Kind.OPTION}"
    if component not in snapshot.responses:
        return {}
    return dict(parse.mark_prices(snapshot.responses[component]))


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


@st.cache_data(ttl=HISTORICAL_VOL_TTL, show_spinner="Fetching realized volatility history...")
def historical_vol_for(currency: str) -> parse.RealizedVolSeries | None:
    """Deribit's own published realized volatility, live-only and
    best-effort like dvol_for. No start/end control -- the venue always
    returns its own trailing history.
    """
    try:
        response = client().fetch_historical_volatility(currency)
    except MarketDataError:
        return None
    return parse.historical_volatility_series(response, currency=currency)


@st.cache_data(ttl=CANDLE_TTL, show_spinner="Fetching candles...")
def candles_for(
    instrument_name: str, start: datetime, end: datetime, resolution: str = "60"
) -> parse.CandleSeries | None:
    """OHLCV candles for one instrument -- live-only and best-effort, same
    None-on-failure convention as historical_vol_for and dvol_for.
    """
    try:
        response = client().fetch_candles(instrument_name, start=start, end=end, resolution=resolution)
    except MarketDataError:
        return None
    return parse.candle_series(response, instrument_name=instrument_name)


def summarise(universe: Universe) -> dict[str, Any]:
    return dossier(universe)


def show_error(exc: Exception) -> None:
    st.error(str(exc))
    st.caption("Data is cached briefly. Use Refresh data in the header to clear it.")
