"""Interactive 3D surface: every calibrated smile stitched across strike and
time, with the no-arbitrage checks that make it trustworthy laid out
explicitly rather than left implicit in the picture.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app import data, state
from app.registry import Slot, panel
from voltk.forward import implied_forward_curve
from voltk.marketdata import MarketDataError
from voltk.surface import Surface, butterfly_check, calendar_reports, calibrate_surface
from voltk.universe import UniverseError

GRID_POINTS = 41


@panel(
    key="surface_3d",
    title="3D Surface",
    slot=Slot.MAIN,
    order=19,
    milestone="M3",
    caption="Every smile stitched into one surface across strike and time, with the no-arbitrage checks stated outright rather than left for the picture to imply.",
)
def render() -> None:
    s = state.get()
    if not s.currencies:
        st.caption("Select a currency first.")
        return

    try:
        universe, _ = data.active_universe()
    except (MarketDataError, UniverseError) as exc:
        data.show_error(exc)
        return

    currency = s.currency or s.currencies[0]
    marks = data.marks_for(currency)
    curve = implied_forward_curve(universe, marks, currency=currency)
    if not curve:
        st.caption("No two-sided marks; nothing to fit a surface from.")
        return

    slices, failures = calibrate_surface(universe, marks, curve)
    if not slices:
        st.caption("No expiry had enough marked strikes to calibrate.")
        _quality_expander(slices, failures)
        return

    surface = Surface(slices=slices)
    _surface_chart(surface, slices)
    _quality_expander(slices, failures)


def _surface_chart(surface: Surface, slices) -> None:
    k_lo = min(s.k_range[0] for s in slices)
    k_hi = max(s.k_range[1] for s in slices)
    pad = 0.20 * max(k_hi - k_lo, 1e-6)
    k_grid = np.linspace(k_lo - pad, k_hi + pad, GRID_POINTS)
    tau_grid = np.linspace(slices[0].tau, slices[-1].tau, max(len(slices), 2) * 4)

    z = np.array([[surface.vol(float(k), float(tau)) for k in k_grid] for tau in tau_grid])

    figure = go.Figure()
    figure.add_trace(
        go.Surface(x=k_grid, y=tau_grid, z=z, colorscale="Viridis", showscale=True, opacity=0.85)
    )
    figure.add_trace(
        go.Scatter3d(
            x=[0.0 for _ in slices],
            y=[s.tau for s in slices],
            z=[s.vol(0.0) for s in slices],
            mode="markers",
            marker=dict(size=4, color="black"),
            name="Fitted ATM vol",
        )
    )
    figure.update_layout(
        scene=dict(
            xaxis_title="Log-moneyness k",
            yaxis_title="Tau (years)",
            zaxis_title="Implied vol",
        ),
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Shaded band beyond the fitted k-range and outside [near, far] tau is "
        "extrapolated: flat variance-rate in time, SVI's own linear-in-k wings "
        "in strike. Neither direction is fit to data past that boundary."
    )


def _quality_expander(slices, failures) -> None:
    with st.expander("No-arbitrage quality flags"):
        any_flag = False
        for s in slices:
            report = butterfly_check(s)
            if not report.clean:
                any_flag = True
                st.warning(
                    f"{s.expiry:%Y-%m-%d}: butterfly arbitrage at "
                    f"{len(report.violations)} of {len(report.nodes)} grid nodes."
                )
        for report in calendar_reports(slices):
            if not report.clean:
                any_flag = True
                st.warning(
                    f"{report.near_expiry:%Y-%m-%d} -> {report.far_expiry:%Y-%m-%d}: "
                    f"total variance decreases at {len(report.violations)} of "
                    f"{len(report.nodes)} grid nodes."
                )
        for failure in failures:
            any_flag = True
            st.info(f"{failure.expiry:%Y-%m-%d}: not calibrated -- {failure.reason}")
        if not any_flag:
            st.success("Butterfly and calendar no-arbitrage hold across every calibrated expiry.")
