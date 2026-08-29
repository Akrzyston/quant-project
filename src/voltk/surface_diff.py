"""Surface comparison: what moved between two fitted surfaces.

Deliberately not a raw SVI-parameter diff -- SVI is not identifiable, so two
parameterizations can differ wildly in (a,b,rho,m,sigma) while producing
nearly identical smiles. The only safe comparison unit is the vol the
surface actually implies, evaluated at its vertex k=0 (the ATM point in
log-moneyness terms).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from voltk.surface import Surface


@dataclass(frozen=True, slots=True)
class SlicePairDiff:
    expiry: datetime
    tau_a: float
    tau_b: float
    atm_vol_a: float
    atm_vol_b: float
    atm_vol_change: float


def compare_surfaces(surface_a: Surface, surface_b: Surface) -> tuple[SlicePairDiff, ...]:
    """Per-expiry ATM vol change, matched by expiry present in both surfaces,
    sorted by expiry. An expiry fitted in only one surface is excluded --
    there is nothing to compare it against, not a data-quality problem to
    surface.
    """
    slices_a = {s.expiry: s for s in surface_a.slices}
    slices_b = {s.expiry: s for s in surface_b.slices}
    shared = sorted(set(slices_a) & set(slices_b))

    diffs = []
    for expiry in shared:
        a, b = slices_a[expiry], slices_b[expiry]
        vol_a, vol_b = a.vol(0.0), b.vol(0.0)
        diffs.append(
            SlicePairDiff(
                expiry=expiry,
                tau_a=a.tau,
                tau_b=b.tau,
                atm_vol_a=vol_a,
                atm_vol_b=vol_b,
                atm_vol_change=vol_b - vol_a,
            )
        )
    return tuple(diffs)
