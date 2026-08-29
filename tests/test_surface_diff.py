"""Surface comparison: matched by expiry, compared in ATM vol terms."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from voltk.surface import SVISlice, Surface
from voltk.surface_diff import compare_surfaces

EXPIRY_1 = datetime(2026, 9, 19, 8, 0, tzinfo=UTC)
EXPIRY_2 = datetime(2026, 10, 17, 8, 0, tzinfo=UTC)


def _slice(expiry, tau, a, m=0.0) -> SVISlice:
    return SVISlice(
        currency="XBT", expiry=expiry, tau=tau,
        a=a, b=0.10, rho=-0.3, m=m, sigma=0.2,
        sse=0.0, points_used=10, points_available=10, k_range=(-1.0, 1.0),
    )


def test_compare_surfaces_matches_by_expiry_and_computes_atm_vol_change() -> None:
    surface_a = Surface(slices=(_slice(EXPIRY_1, 10 / 365, a=0.05), _slice(EXPIRY_2, 40 / 365, a=0.08)))
    surface_b = Surface(slices=(_slice(EXPIRY_1, 10 / 365, a=0.09), _slice(EXPIRY_2, 40 / 365, a=0.08)))

    diffs = compare_surfaces(surface_a, surface_b)

    assert [d.expiry for d in diffs] == [EXPIRY_1, EXPIRY_2]
    moved, unmoved = diffs
    assert moved.atm_vol_change > 0.0
    assert unmoved.atm_vol_change == pytest.approx(0.0, abs=1e-12)
    assert moved.atm_vol_a == pytest.approx(_slice(EXPIRY_1, 10 / 365, a=0.05).vol(0.0))
    assert moved.atm_vol_b == pytest.approx(_slice(EXPIRY_1, 10 / 365, a=0.09).vol(0.0))


def test_expiry_present_in_only_one_surface_is_excluded() -> None:
    surface_a = Surface(slices=(_slice(EXPIRY_1, 10 / 365, a=0.05),))
    surface_b = Surface(slices=(_slice(EXPIRY_2, 40 / 365, a=0.08),))

    diffs = compare_surfaces(surface_a, surface_b)

    assert diffs == ()
