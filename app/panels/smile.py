"""Per-expiry smile: total variance against log-moneyness, fitted with SVI,
cross-checked against Deribit's published DVOL and the model-free variance
index built straight from the chain.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app import data, state
from app.registry import Slot, panel
from voltk.forward import implied_forward_curve
from voltk.marketdata import MarketDataError
from voltk.surface import SurfaceError, butterfly_check, calibrate_svi_slice, smile_points
from voltk.universe import UniverseError
from voltk.variance import compare_to_dvol, constant_maturity_variance, variance_term_structure

TARGET_TENOR_DAYS = 30.0


@panel(
    key="smile",
    title="Smile",
    slot=Slot.MAIN,
    order=18,
    milestone="M3",
    caption="The per-expiry smile in total variance against log-moneyness, checked against Deribit's own DVOL: a number that's actually verifiable, not just plausible-looking.",
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
        st.caption("No two-sided marks; nothing to fit a smile from.")
        return

    _dvol_block(universe, marks, curve, currency)

    expiry_by_label = {f.expiry.strftime("%d %b %Y"): f for f in curve}
    label = st.selectbox("Expiry", list(expiry_by_label), key="smile_expiry")
    implied_forward = expiry_by_label[label]
    tau = (implied_forward.expiry - universe.as_of).total_seconds() / (365 * 24 * 3600)

    points = smile_points(universe, marks, implied_forward)
    try:
        fit = calibrate_svi_slice(points, currency=currency, expiry=implied_forward.expiry, tau=tau)
    except SurfaceError as exc:
        st.warning(str(exc))
        return

    _smile_chart(points, fit)
    _strike_view_expander(points, fit, implied_forward.forward)
    _residual_table(points, fit)

    st.caption(
        "Total variance (vol^2 * tau) against log-moneyness ln(K/F), not raw vol "
        "against strike: it is the quantity that is linear under time "
        "interpolation and whose monotonicity in maturity IS the calendar "
        "no-arbitrage condition -- neither is true of vol against strike, and "
        "strikes are not even comparable across expiries the way moneyness is."
    )


def _smile_chart(points, fit) -> None:
    identified = [p for p in points if p.identified]
    unidentified = [p for p in points if not p.identified]

    k_lo, k_hi = fit.k_range
    pad = 0.20 * max(k_hi - k_lo, 1e-6)
    k_grid = np.linspace(k_lo - pad, k_hi + pad, 121)
    fitted_w = [fit.total_variance(float(k)) for k in k_grid]

    report = butterfly_check(fit)
    violations = report.violations

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=[p.k for p in identified], y=[p.total_variance for p in identified],
                    name="Market (identified)", mode="markers")
    )
    if unidentified:
        figure.add_trace(
            go.Scatter(x=[p.k for p in unidentified], y=[p.total_variance for p in unidentified],
                        name="Market (not identified)", mode="markers",
                        marker=dict(symbol="circle-open"))
        )
    figure.add_trace(go.Scatter(x=list(k_grid), y=fitted_w, name="Fitted SVI", mode="lines"))
    if violations:
        figure.add_trace(
            go.Scatter(x=[n.k for n in violations], y=[fit.total_variance(n.k) for n in violations],
                        name="Butterfly violation", mode="markers",
                        marker=dict(symbol="x", size=10, color="red"))
        )
    figure.add_vrect(x0=k_lo, x1=k_hi, fillcolor="LightGrey", opacity=0.15, line_width=0,
                      annotation_text="fitted range", annotation_position="top left")
    figure.update_layout(
        xaxis_title="Log-moneyness k = ln(K/F)",
        yaxis_title="Total variance w(k)",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(figure, width="stretch")
    if violations:
        st.warning(f"{len(violations)} of {len(report.nodes)} grid nodes breach butterfly no-arbitrage.")
    else:
        st.caption("Butterfly no-arbitrage holds across the checked range.")


def _strike_view_expander(points, fit, forward: float) -> None:
    with st.expander("Strike / vol view (re-expressed for display, not what was fitted)"):
        k_lo, k_hi = fit.k_range
        k_grid = np.linspace(k_lo, k_hi, 81)
        strikes = [forward * float(np.exp(k)) for k in k_grid]
        vols = [fit.vol(float(k)) for k in k_grid]

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=[p.strike for p in points], y=[p.market_vol for p in points],
                        name="Market", mode="markers")
        )
        figure.add_trace(go.Scatter(x=strikes, y=vols, name="Fitted SVI", mode="lines"))
        figure.add_vline(x=forward, line_dash="dot", annotation_text="forward")
        figure.update_layout(
            xaxis_title="Strike", yaxis_title="Implied vol", height=360,
            margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(figure, width="stretch")


def _residual_table(points, fit) -> None:
    rows = [
        {
            "Strike": p.strike,
            "k": round(p.k, 4),
            "Right": p.option_type.value,
            "Market vol": round(p.market_vol, 4),
            "Fitted vol": round(fit.vol(p.k), 4),
            "Residual (vol)": round(p.market_vol - fit.vol(p.k), 4),
            "Identified": p.identified,
        }
        for p in points
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    residuals = [abs(r["Residual (vol)"]) for r in rows if r["Identified"]]
    if residuals:
        st.caption(
            f"Mean absolute residual {sum(residuals) / len(residuals):.4f} vol; "
            "residuals concentrate at the wings, where a single strike carries the "
            "curvature term and the fit has the least data to constrain it."
        )


def _dvol_block(universe, marks, curve, currency: str) -> None:
    target_tau = TARGET_TENOR_DAYS / 365.0
    slices = variance_term_structure(universe, marks, curve)
    cmv = constant_maturity_variance(slices, target_tau, currency=currency) if slices else None

    dvol_series = data.dvol_for(currency)
    dvol_close = dvol_series.latest.close if dvol_series and dvol_series.latest else None

    comparison = compare_to_dvol(cmv, dvol_close, None, currency=currency)

    cols = st.columns(3)
    cols[0].metric(
        f"Model-free {TARGET_TENOR_DAYS:g}d vol",
        f"{comparison.model_free_vol:.2%}" if comparison.model_free_vol is not None else "—",
    )
    cols[1].metric(
        "Deribit DVOL (latest)",
        f"{comparison.dvol:.2%}" if comparison.dvol is not None else "—",
        delta=f"{comparison.gap_vs_dvol_bps:.0f} bps" if comparison.gap_vs_dvol_bps is not None else None,
    )
    extrapolated = cmv.extrapolated if cmv else False
    cols[2].metric("Expiries used", len(slices), help="Extrapolated past the fitted range" if extrapolated else None)
    if cmv is None:
        st.caption("Not enough marked strikes to build a model-free variance index yet.")
    elif dvol_close is None:
        st.caption("DVOL unavailable right now; showing the model-free index alone.")
