"""Replay must not read the wall clock.

The failure mode this guards against is silent: a snapshot that reloads fine but
produces slightly different tau on every run, so nothing downstream reconciles.
"""

from __future__ import annotations

import time

from tests.conftest import BASE, FakeVenue
from voltk.capture import capture, dossier
from voltk.snapshots import ReplaySource, SnapshotStore, rebuild_universe
from voltk.universe import UniverseSpec


def test_capture_through_replay_source_matches_the_original(
    venue: FakeVenue, store: SnapshotStore
) -> None:
    original = capture(venue, UniverseSpec.of([BASE]))
    snapshot = store.load(store.save(original))

    time.sleep(0.1)
    replayed = capture(ReplaySource(snapshot), snapshot.meta.spec)

    assert replayed.as_of == original.as_of
    assert replayed.quality.replayed
    assert dossier(replayed.universe)["digest"] == dossier(original.universe)["digest"]


def test_recorded_drift_survives_replay(store: SnapshotStore) -> None:
    venue = FakeVenue(index_path=[60000.0, 60600.0])
    original = capture(venue, UniverseSpec.of([BASE]))
    snapshot = store.load(store.save(original))

    replayed = capture(ReplaySource(snapshot), snapshot.meta.spec)
    assert replayed.quality.index_drift_bps == original.quality.index_drift_bps


def test_repeated_replay_is_bit_for_bit_stable(venue: FakeVenue, store: SnapshotStore) -> None:
    snapshot_id = store.save(capture(venue, UniverseSpec.of([BASE])))

    digests = set()
    for _ in range(4):
        time.sleep(0.02)
        digests.add(dossier(rebuild_universe(store.load(snapshot_id)))["digest"])

    assert len(digests) == 1
