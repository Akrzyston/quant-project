"""Registry tests.

Discovery is automatic, so these tests supply the noise: a panel that stops
registering fails here rather than disappearing from the screen.
"""

from __future__ import annotations

import pytest

from app.registry import Slot, all_panels, discover, manifest, panels_for

EXPECTED = {
    ("symbol_selector", "left_rail", 10),
    ("model_selector", "left_rail", 20),
    ("option_details", "left_rail", 30),
    ("greeks_details", "left_rail", 40),
    ("snapshot_control", "left_rail", 50),
    ("instrument_dossier", "main", 10),
    ("vol_surface_details", "main", 20),
    ("quoting_parameters", "quoter_rail", 10),
    ("mock_orderbook", "quoter_main", 10),
}


@pytest.fixture(scope="module", autouse=True)
def _discovered() -> None:
    discover()


def test_manifest_matches_expected() -> None:
    assert set(manifest()) == EXPECTED


def test_every_slot_is_populated() -> None:
    assert not [slot for slot in Slot if not panels_for(slot)]


def test_panel_keys_are_unique() -> None:
    keys = [spec.key for spec in all_panels()]
    assert len(keys) == len(set(keys))


def test_ordering_within_a_slot_is_unambiguous() -> None:
    for slot in Slot:
        orders = [spec.order for spec in panels_for(slot)]
        assert orders == sorted(orders)
        assert len(orders) == len(set(orders))


def test_every_panel_declares_a_milestone() -> None:
    for spec in all_panels():
        assert spec.milestone
