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
    ("chain_view", "main", 12),
    ("delta_profile", "main", 15),
    ("forward_curve", "main", 17),
    ("smile", "main", 18),
    ("surface_3d", "main", 19),
    ("snapshot_browser", "main", 20),
    ("vol_history", "main", 22),
    ("position_view", "main", 24),
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
