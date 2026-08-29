"""Snapshot tests: storage, integrity, and reproducible replay."""

from __future__ import annotations

import time

import pytest

import voltk
from tests.conftest import ALT, BASE, FakeVenue
from voltk.capture import capture, dossier
from voltk.canonical import canonical_json
from voltk.marketdata.base import MarketDataError
from voltk.snapshots import (
    ReplaySource,
    SnapshotError,
    SnapshotStore,
    payload_hashes,
    rebuild_universe,
)
from voltk.universe import UniverseSpec


def _saved(venue: FakeVenue, store: SnapshotStore, currencies=(BASE,)) -> str:
    return store.save(capture(venue, UniverseSpec.of(currencies)))


def test_round_trip_preserves_payload_bytes(venue: FakeVenue, store: SnapshotStore) -> None:
    original = capture(venue, UniverseSpec.of([BASE]))
    snapshot_id = store.save(original)
    reloaded = store.load(snapshot_id)

    assert payload_hashes(reloaded) == {
        component: response.sha256 for component, response in original.responses
    }
    for component, response in original.responses:
        assert reloaded.responses[component].body == response.body


def test_replay_reproduces_the_dossier_exactly(venue: FakeVenue, store: SnapshotStore) -> None:
    snapshot_id = _saved(venue, store)
    first = dossier(rebuild_universe(store.load(snapshot_id)))
    time.sleep(0.05)
    second = dossier(rebuild_universe(store.load(snapshot_id)))

    assert canonical_json(first) == canonical_json(second)
    assert first["digest"] == second["digest"]


def test_replay_does_not_drift_with_wall_clock(venue: FakeVenue, store: SnapshotStore) -> None:
    """tau must come from the snapshot, not from now()."""
    snapshot_id = _saved(venue, store)
    first = rebuild_universe(store.load(snapshot_id))
    time.sleep(0.2)
    second = rebuild_universe(store.load(snapshot_id))

    assert first.as_of == second.as_of
    option = first.options(BASE)[0]
    same = next(i for i in second.options(BASE) if i.name == option.name)
    assert option.tau(first.as_of) == same.tau(second.as_of)


def test_live_and_replay_agree_on_the_same_payloads(
    venue: FakeVenue, store: SnapshotStore
) -> None:
    original = capture(venue, UniverseSpec.of([BASE]))
    snapshot_id = store.save(original)
    replayed = rebuild_universe(store.load(snapshot_id))

    assert dossier(original.universe)["digest"] == dossier(replayed)["digest"]


def test_corrupted_payload_is_refused(venue: FakeVenue, store: SnapshotStore) -> None:
    snapshot_id = _saved(venue, store)
    import sqlite3

    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE payloads SET body = ? WHERE snapshot_id = ?",
            (b'{"result": []}', snapshot_id),
        )

    with pytest.raises(SnapshotError, match="hash check"):
        store.load(snapshot_id)


def test_metadata_carries_source_and_retrieval_time(
    venue: FakeVenue, store: SnapshotStore
) -> None:
    snapshot_id = _saved(venue, store)
    universe = rebuild_universe(store.load(snapshot_id))
    instrument = universe.instruments[0]

    assert instrument.provenance is not None
    assert instrument.provenance.source
    assert instrument.retrieved_at is not None
    assert instrument.provenance.endpoint


def test_snapshot_is_versioned_and_timestamped(venue: FakeVenue, store: SnapshotStore) -> None:
    snapshot_id = _saved(venue, store)
    meta = store.load(snapshot_id).meta

    assert meta.schema_version >= 1
    assert meta.library_version == voltk.__version__
    assert meta.as_of.tzinfo is not None
    assert meta.content_hash
    assert snapshot_id.startswith(meta.as_of.strftime("%Y%m%dT%H%M%SZ"))


def test_reload_refuses_a_snapshot_from_a_different_library_version(
    venue: FakeVenue, store: SnapshotStore
) -> None:
    snapshot_id = _saved(venue, store)
    import sqlite3

    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE snapshots SET library_version = ? WHERE snapshot_id = ?",
            ("0.0.0-stale", snapshot_id),
        )

    with pytest.raises(SnapshotError, match="0.0.0-stale"):
        store.load(snapshot_id)


def test_replay_source_satisfies_the_market_data_protocol(
    venue: FakeVenue, store: SnapshotStore
) -> None:
    snapshot = store.load(_saved(venue, store))
    source = ReplaySource(snapshot)

    assert not source.is_live
    assert source.now() == snapshot.meta.as_of
    replayed = capture(source, snapshot.meta.spec)
    assert dossier(replayed.universe)["digest"] == dossier(rebuild_universe(snapshot))["digest"]


def test_replay_refuses_a_universe_it_did_not_capture(
    venue: FakeVenue, store: SnapshotStore
) -> None:
    snapshot = store.load(_saved(venue, store, currencies=(BASE,)))
    source = ReplaySource(snapshot)

    with pytest.raises(MarketDataError, match="no component"):
        capture(source, UniverseSpec.of([ALT]))


def test_cross_currency_snapshot_round_trips(store: SnapshotStore) -> None:
    venue = FakeVenue(currencies=(BASE, ALT))
    snapshot_id = store.save(capture(venue, UniverseSpec.of([BASE, ALT])))
    universe = rebuild_universe(store.load(snapshot_id))

    assert universe.spec.is_cross_currency
    assert universe.options(BASE) and universe.options(ALT)
    assert set(universe.index_prices) == {BASE, ALT}


def test_listing_orders_newest_first(venue: FakeVenue, store: SnapshotStore) -> None:
    ids = [_saved(venue, store) for _ in range(3)]
    listed = [meta.snapshot_id for meta in store.list()]

    assert set(listed) == set(ids)
    as_of = [meta.as_of for meta in store.list()]
    assert as_of == sorted(as_of, reverse=True)


def test_degraded_capture_is_marked_in_storage(store: SnapshotStore) -> None:
    venue = FakeVenue(index_path=[60000.0, 61000.0])
    snapshot_id = store.save(capture(venue, UniverseSpec.of([BASE])))
    meta = store.load(snapshot_id).meta

    assert meta.degraded
    assert meta.quality.reasons


def test_delete_removes_payloads(venue: FakeVenue, store: SnapshotStore) -> None:
    snapshot_id = _saved(venue, store)
    store.delete(snapshot_id)

    with pytest.raises(SnapshotError):
        store.load(snapshot_id)

