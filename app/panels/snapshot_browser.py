"""Browse stored snapshots, compare two of them (what moved on the surface),
and attribute a held position's P&L between them via the Greeks-based
Taylor identity.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app import data, pricing, state
from app.registry import Slot, panel
from voltk import models
from voltk.forward import implied_forward_curve
from voltk.greeks import cash_greeks_from_coin
from voltk.models.bounds import ArbitrageError
from voltk.models.solver import SolverError
from voltk.pnl import attribute_pnl
from voltk.snapshots import SnapshotError
from voltk.surface import Surface, SurfaceError, calibrate_svi_slice, log_moneyness, smile_points
from voltk.surface_diff import compare_surfaces

GRID_POINTS = 41


@panel(
    key="snapshot_browser",
    title="Snapshot Browser",
    slot=Slot.MAIN,
    order=20,
    milestone="M5",
)
def render() -> None:
    s = state.get()
    saved = data.store().list()
    if not saved:
        st.caption("No snapshots yet. Take one from the Snapshots panel.")
        return

    _browser_table(saved)

    if len(saved) < 2 or not s.currencies:
        st.caption("Need at least two snapshots (and a selected currency) to compare.")
        return

    currency = s.currency or s.currencies[0]
    ids = [meta.snapshot_id for meta in saved]
    labels = {meta.snapshot_id: meta.display_name() for meta in saved}

    col_a, col_b = st.columns(2)
    snapshot_b_id = col_b.selectbox(
        "Snapshot B (later)", ids, index=0, format_func=lambda k: labels[k], key="browser_b"
    )
    snapshot_a_id = col_a.selectbox(
        "Snapshot A (earlier)", ids, index=min(1, len(ids) - 1),
        format_func=lambda k: labels[k], key="browser_a",
    )
    if snapshot_a_id == snapshot_b_id:
        st.caption("Pick two different snapshots.")
        return

    try:
        universe_a = data.universe_for_snapshot(snapshot_a_id)
        universe_b = data.universe_for_snapshot(snapshot_b_id)
    except SnapshotError as exc:
        st.error(str(exc))
        return
    marks_a = data.marks_for_snapshot(snapshot_a_id, currency)
    marks_b = data.marks_for_snapshot(snapshot_b_id, currency)

    _compare_block(universe_a, marks_a, universe_b, marks_b, currency)
    _attribution_block(universe_a, marks_a, universe_b, marks_b, currency, s)


def _browser_table(saved) -> None:
    rows = [
        {
            "Snapshot": meta.snapshot_id,
            "As of": meta.as_of.strftime("%Y-%m-%d %H:%M:%S"),
            "Currencies": "/".join(meta.spec.currencies),
            "Degraded": meta.degraded,
            "Note": meta.note,
        }
        for meta in saved
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def _fit_slice(universe, marks, currency: str, expiry):
    forwards = implied_forward_curve(universe, marks, currency=currency, expiries=[expiry])
    if not forwards:
        return None
    points = smile_points(universe, marks, forwards[0])
    tau = (expiry - universe.as_of).total_seconds() / (365 * 24 * 3600)
    try:
        return calibrate_svi_slice(points, currency=currency, expiry=expiry, tau=tau)
    except SurfaceError:
        return None


def _fit_surface(universe, marks, currency: str) -> Surface | None:
    slices = [
        _fit_slice(universe, marks, currency, expiry) for expiry in universe.expiries(currency)
    ]
    slices = tuple(s for s in slices if s is not None)
    return Surface(slices=slices) if slices else None


def _compare_block(universe_a, marks_a, universe_b, marks_b, currency: str) -> None:
    with st.expander("A vs B: what moved", expanded=True):
        surface_a = _fit_surface(universe_a, marks_a, currency)
        surface_b = _fit_surface(universe_b, marks_b, currency)
        if surface_a is None or surface_b is None:
            st.caption("Not enough marked strikes in one of the two snapshots to compare.")
            return

        all_slices = list(surface_a.slices) + list(surface_b.slices)
        k_lo = min(sl.k_range[0] for sl in all_slices)
        k_hi = max(sl.k_range[1] for sl in all_slices)
        pad = 0.20 * max(k_hi - k_lo, 1e-6)
        k_grid = np.linspace(k_lo - pad, k_hi + pad, GRID_POINTS)
        tau_lo = min(sl.tau for sl in all_slices)
        tau_hi = max(sl.tau for sl in all_slices)
        tau_grid = np.linspace(tau_lo, tau_hi, max(len(all_slices), 2) * 4)

        z = np.array(
            [
                [
                    surface_b.vol(float(k), float(tau)) - surface_a.vol(float(k), float(tau))
                    for k in k_grid
                ]
                for tau in tau_grid
            ]
        )
        figure = go.Figure(
            go.Surface(x=k_grid, y=tau_grid, z=z, colorscale="RdBu", colorbar=dict(title="Δ vol"))
        )
        figure.update_layout(
            scene=dict(
                xaxis_title="Log-moneyness k", yaxis_title="Tau (years)",
                zaxis_title="Δ implied vol (B - A)",
            ),
            height=500,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(figure, width="stretch")

        diffs = compare_surfaces(surface_a, surface_b)
        if diffs:
            st.dataframe(
                [
                    {
                        "Expiry": d.expiry.strftime("%Y-%m-%d"),
                        "ATM vol A": round(d.atm_vol_a, 4),
                        "ATM vol B": round(d.atm_vol_b, 4),
                        "Change": round(d.atm_vol_change, 4),
                    }
                    for d in diffs
                ],
                hide_index=True, width="stretch",
            )
        st.caption(
            "Grid is unioned across both snapshots' fitted strike ranges before "
            "differencing, so a point extrapolated in one is never compared as if "
            "it were as trustworthy as a fitted point in the other."
        )


def _attribution_block(universe_a, marks_a, universe_b, marks_b, currency: str, s) -> None:
    with st.expander("P&L attribution"):
        names_a = {i.name: i for i in universe_a.options(currency) if i.strike is not None}
        names_b = {i.name: i for i in universe_b.options(currency) if i.strike is not None}
        shared_names = sorted(set(names_a) & set(names_b))
        if not shared_names:
            st.caption("No contract is listed in both snapshots.")
            return

        selected = st.multiselect("Positions", shared_names, key="browser_positions")
        if not selected:
            st.caption("Select one or more contracts held between A and B.")
            return

        spec = models.get(s.model_key or models.all_models()[0].key)
        if not spec.implemented:
            st.info(f"{spec.display_name} arrives at {spec.milestone}.")
            return
        model = spec.build()

        d_t_years = (universe_b.as_of - universe_a.as_of).total_seconds() / (365 * 24 * 3600)

        rows = []
        totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "residual": 0.0, "actual": 0.0}
        for name in selected:
            size = st.number_input(f"Size: {name}", value=1.0, step=1.0, key=f"browser_size_{name}")
            attribution = _attribute_one(
                universe_a, marks_a, universe_b, marks_b, currency,
                names_a[name], names_b[name], spec, model, d_t_years,
            )
            if attribution is None:
                st.caption(f"{name}: could not be priced at both snapshots.")
                continue
            rows.append(
                {
                    "Instrument": name, "Price A": round(attribution.price_a, 6),
                    "Price B": round(attribution.price_b, 6),
                    "Actual PnL": round(size * attribution.actual_pnl, 6),
                    "Delta term": round(size * attribution.delta_term, 6),
                    "Gamma term": round(size * attribution.gamma_term, 6),
                    "Vega term": round(size * attribution.vega_term, 6),
                    "Theta term": round(size * attribution.theta_term, 6),
                    "Residual": round(size * attribution.residual, 6),
                }
            )
            totals["delta"] += size * attribution.delta_term
            totals["gamma"] += size * attribution.gamma_term
            totals["vega"] += size * attribution.vega_term
            totals["theta"] += size * attribution.theta_term
            totals["residual"] += size * attribution.residual
            totals["actual"] += size * attribution.actual_pnl

        if not rows:
            return
        st.dataframe(rows, hide_index=True, width="stretch")
        st.metric("Portfolio actual P&L", f"{totals['actual']:,.6f}")
        st.caption(
            f"Explained: delta {totals['delta']:,.4f} + gamma {totals['gamma']:,.4f} + "
            f"vega {totals['vega']:,.4f} + theta {totals['theta']:,.4f} = "
            f"{totals['actual'] - totals['residual']:,.4f}. "
            f"Residual (never absorbed): {totals['residual']:,.4f}."
        )


def _attribute_one(universe_a, marks_a, universe_b, marks_b, currency, inst_a, inst_b, spec, model, d_t_years):
    slice_a = _fit_slice(universe_a, marks_a, currency, inst_a.expiry)
    slice_b = _fit_slice(universe_b, marks_b, currency, inst_b.expiry)
    if slice_a is None or slice_b is None:
        return None

    args_a = pricing.inputs_for(universe_a, inst_a, spec)
    args_b = pricing.inputs_for(universe_b, inst_b, spec)
    forward_a = universe_a.forward_for(currency, inst_a.expiry)
    forward_b = universe_b.forward_for(currency, inst_b.expiry)
    vol_a = slice_a.vol(log_moneyness(inst_a.strike, forward_a))
    vol_b = slice_b.vol(log_moneyness(inst_b.strike, forward_b))

    try:
        price_a = model.price(args_a.underlying, args_a.strike, args_a.tau, vol_a, args_a.rate, args_a.cp)
        price_b = model.price(args_b.underlying, args_b.strike, args_b.tau, vol_b, args_b.rate, args_b.cp)
        coin_a = model.greeks(args_a.underlying, args_a.strike, args_a.tau, vol_a, args_a.rate, args_a.cp)
    except (ArbitrageError, SolverError):
        return None

    if spec.settles_in_base:
        # cash_greeks_from_coin's delta/gamma are SPOT derivatives
        # (dV_cash/dS), not forward derivatives -- d_underlying must match
        # that, not the model's own forward-parameterised convention.
        spot_a = universe_a.index(currency)
        spot_b = universe_b.index(currency)
        greeks_a = cash_greeks_from_coin(coin_a, price_a, forward_a, spot_a)
        price_a_cash = price_a * spot_a
        price_b_cash = price_b * spot_b
        d_underlying = spot_b - spot_a
    else:
        greeks_a = coin_a
        price_a_cash = price_a
        price_b_cash = price_b
        d_underlying = args_b.underlying - args_a.underlying

    d_vol = vol_b - vol_a
    return attribute_pnl(greeks_a, price_a_cash, price_b_cash, d_underlying, d_vol, d_t_years)
