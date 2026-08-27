"""Capture tests: bracketing, gating, and the forward curve."""

from __future__ import annotations

import pytest

from tests.conftest import ALT, BASE, FakeVenue
from voltk.capture import capture, dossier
from voltk.instruments import Kind
from voltk.marketdata.base import MarketDataError
from voltk.universe import UniverseSpec


def test_index_is_read_before_and_after_the_chain(venue: FakeVenue) -> None:
    capture(venue, UniverseSpec.of([BASE]))
    index_calls = [c for c in venue.calls if c.endswith("get_index_price")]
    instrument_calls = [i for i, c in enumerate(venue.calls) if c.endswith("get_instruments")]
    first_index = venue.calls.index("public/get_index_price")
    last_index = len(venue.calls) - 1 - venue.calls[::-1].index("public/get_index_price")

    assert len(index_calls) >= 2
    assert first_index > max(instrument_calls), "definitions are read before the index bracket"
    assert last_index > first_index, "the bracket must close after quotes are read"


def test_quiet_market_passes_the_gate(venue: FakeVenue) -> None:
    result = capture(venue, UniverseSpec.of([BASE]))
    assert result.quality.ok
    assert result.quality.worst_drift_bps == pytest.approx(0.0)


def test_moving_index_is_flagged_degraded() -> None:
    venue = FakeVenue(index_path=[60000.0, 60600.0])
    result = capture(venue, UniverseSpec.of([BASE]), max_index_drift_bps=15.0)
    assert not result.quality.ok
    assert any("moved" in reason for reason in result.quality.reasons)
    assert result.quality.worst_drift_bps == pytest.approx(100.0, rel=1e-3)


def test_slow_capture_is_flagged_degraded(venue: FakeVenue) -> None:
    result = capture(venue, UniverseSpec.of([BASE]), max_window_ms=0.0)
    assert not result.quality.ok
    assert any("took" in reason for reason in result.quality.reasons)


def test_as_of_is_the_midpoint_of_the_bracket(venue: FakeVenue) -> None:
    result = capture(venue, UniverseSpec.of([BASE]))
    assert result.started_at <= result.as_of <= result.completed_at


def test_missing_index_metadata_refuses_to_capture(venue: FakeVenue) -> None:
    class NoIndex(FakeVenue):
        def fetch_instruments(self, currency, kind):
            response = super().fetch_instruments(currency, kind)
            stripped = [dict(item, price_index=None) for item in response.result()]
            return self._wrap("public/get_instruments", stripped, {"currency": currency})

    with pytest.raises(MarketDataError, match="price index"):
        capture(NoIndex(), UniverseSpec.of([BASE]))


def test_cross_currency_shares_one_window() -> None:
    venue = FakeVenue(currencies=(BASE, ALT))
    result = capture(venue, UniverseSpec.of([BASE, ALT]))
    assert result.universe.spec.is_cross_currency
    assert set(result.universe.index_prices) == {BASE, ALT}
    assert result.universe.options(BASE) and result.universe.options(ALT)
    assert result.as_of == result.universe.as_of


def test_forward_curve_is_built_from_futures_marks(venue: FakeVenue) -> None:
    result = capture(venue, UniverseSpec.of([BASE]))
    forwards = result.universe.forwards(BASE)
    assert forwards
    assert [p.expiry for p in forwards] == sorted(p.expiry for p in forwards)
    assert all(p.forward > 0 for p in forwards)


def test_forward_interpolates_between_futures(venue: FakeVenue) -> None:
    universe = capture(venue, UniverseSpec.of([BASE])).universe
    points = universe.forwards(BASE)
    midpoint = points[0].expiry + (points[1].expiry - points[0].expiry) / 2
    interpolated = universe.forward_for(BASE, midpoint)
    assert min(points[0].forward, points[1].forward) < interpolated
    assert interpolated < max(points[0].forward, points[1].forward)


def test_forward_falls_back_to_index_without_futures(venue: FakeVenue) -> None:
    universe = capture(venue, UniverseSpec.of([BASE], kinds=[Kind.OPTION])).universe
    expiry = universe.expiries(BASE)[0]
    assert universe.forward_for(BASE, expiry) == universe.index(BASE)


def test_dossier_is_derived_only_from_the_universe(venue: FakeVenue) -> None:
    universe = capture(venue, UniverseSpec.of([BASE])).universe
    assert dossier(universe)["digest"] == dossier(universe)["digest"]
