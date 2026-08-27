"""Greeks details."""

from __future__ import annotations

from app.panels._placeholder import stub
from app.registry import Slot, panel


@panel(
    key="greeks_details",
    title="Greeks Details",
    slot=Slot.LEFT_RAIL,
    order=40,
    milestone="M4",
    caption="Populates when an expiry is selected from the volatility surface.",
)
def render() -> None:
    stub(
        "M4",
        [
            "Delta, gamma, vega, theta, rho in quote and base units",
            "Bucketed vega by expiry",
            "Analytic against finite-difference, residual per Greek",
        ],
    )
