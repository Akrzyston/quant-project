"""Greeks in both units, skew-adjusted delta under both sticky regimes,
bucketed vega, a structured shock ladder, and a minimal session-local
portfolio view.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import data, pricing, state
from app.registry import Slot, panel
from voltk import models
from voltk.forward import implied_forward_curve
from voltk.greeks import cash_greeks_from_coin
from voltk.marketdata import MarketDataError
from voltk.models.bounds import ArbitrageError
from voltk.models.solver import SolverError
from voltk.portfolio import Position, PortfolioError, aggregate_greeks
from voltk.shocks import build_ladder, level_shock_magnitudes_from_dvol
from voltk.skew import skew_adjusted_delta
from voltk.surface import Surface, SurfaceError, calibrate_svi_slice, log_moneyness, smile_points
from voltk.universe import UniverseError
from voltk.vega_buckets import bucket_vega

DEFAULT_LEVEL_MULTIPLES = (-0.4, -0.2, 0.2, 0.4)


@panel(
    key="greeks_details",
    title="Greeks Details",
    slot=Slot.LEFT_RAIL,
    order=40,
    milestone="M4",
    caption="Every Greek in both settlement units, skew-adjusted delta under both sticky regimes, bucketed vega, and a structured shock ladder for whatever contract is selected above.",
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
    if not s.instrument_name:
        st.caption("Select a contract in Option Details first.")
        return

    instrument = next(
        (i for i in universe.options(currency) if i.name == s.instrument_name), None
    )
    if instrument is None or instrument.strike is None or instrument.expiry is None:
        st.caption("No listed contract at that selection.")
        return

    spec = models.get(s.model_key or models.all_models()[0].key)
    if not spec.implemented:
        st.info(f"{spec.display_name} arrives at {spec.milestone}.")
        return
    model = spec.build()
    args = pricing.inputs_for(universe, instrument, spec)

    marks = data.marks_for(currency)
    slice_ = _fit_slice(universe, marks, currency, instrument.expiry)
    if slice_ is None:
        st.warning("Not enough marked strikes at this expiry to fit a smile.")
        return

    forward = universe.forward_for(currency, instrument.expiry)
    k = log_moneyness(instrument.strike, forward)
    vol = slice_.vol(k)
    surface = Surface(slices=(slice_,))
    spot = universe.index(currency)

    try:
        coin_price = model.price(args.underlying, args.strike, args.tau, vol, args.rate, args.cp)
        coin = model.greeks(args.underlying, args.strike, args.tau, vol, args.rate, args.cp)
    except ArbitrageError as exc:
        st.error(str(exc))
        return

    cash = (
        cash_greeks_from_coin(coin, coin_price, forward, spot) if spec.settles_in_base else coin
    )

    _greeks_table(spec, coin, cash)
    _skew_block(model, surface, instrument, forward, args)
    _bucket_block(universe, marks, currency, instrument, model, spec, forward, args, slice_)
    _shock_block(currency, model, spec, instrument, slice_, forward, args)
    _portfolio_block(universe, currency, marks, spec, model)


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


def _greeks_table(spec, coin, cash) -> None:
    with st.expander("Greeks table (coin and cash)", expanded=True):
        fields = ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]
        if spec.settles_in_base:
            rows = [
                {"Greek": f, "Coin": round(getattr(coin, f), 8), "Cash": round(getattr(cash, f), 6)}
                for f in fields
            ]
            st.dataframe(rows, hide_index=True, width="stretch")
            if coin.gamma:
                st.caption(f"Theta per unit gamma (coin): {coin.theta / coin.gamma:,.6f}")
            st.caption(
                "Cash delta/gamma are derived via the self-quanto chain rule "
                "(voltk.greeks.cash_greeks_from_coin), not a coin*spot conversion."
            )
        else:
            rows = [{"Greek": f, "Value (quote currency)": round(getattr(cash, f), 6)} for f in fields]
            st.dataframe(rows, hide_index=True, width="stretch")
            if cash.gamma:
                st.caption(f"Theta per unit gamma: {cash.theta / cash.gamma:,.6f}")
            st.caption(f"{spec.display_name} settles in the quote currency; coin and cash coincide.")


def _skew_block(model, surface, instrument, forward, args) -> None:
    with st.expander("Skew-adjusted delta"):
        result = skew_adjusted_delta(
            model, surface, instrument.strike, forward, args.tau, args.rate, args.cp
        )
        rows = [
            {"Regime": "Flat (model, no adjustment)", "Delta": round(result.flat_delta, 6), "vs flat": 0.0},
            {
                "Regime": "Sticky-strike",
                "Delta": round(result.sticky_strike_delta, 6),
                "vs flat": round(result.sticky_strike_delta - result.flat_delta, 6),
            },
            {
                "Regime": "Sticky-delta",
                "Delta": round(result.sticky_delta_delta, 6),
                "vs flat": round(result.sticky_delta_adjustment, 6),
            },
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "Sticky-strike leaves delta exactly flat: the fitted smile is pinned to "
            "absolute strikes by definition, so it doesn't move as spot moves. "
            "Sticky-delta picks up Vega x dsigma/d(underlying) from the fitted smile. "
            "Which regime the market actually follows is an empirical question, "
            "answered against realized data in the vol dynamics tab."
        )


def _bucket_block(universe, marks, currency, instrument, model, spec, forward, args, slice_) -> None:
    with st.expander("Bucketed vega"):
        forwards = implied_forward_curve(
            universe, marks, currency=currency, expiries=[instrument.expiry]
        )
        if not forwards:
            st.caption("No two-sided marks; nothing to bucket.")
            return
        points = smile_points(universe, marks, forwards[0])
        try:
            result = bucket_vega(
                points, slice_, model, forward, instrument.strike, args.tau, args.rate, args.cp,
                currency=currency, expiry=instrument.expiry,
            )
        except Exception as exc:
            st.caption(f"Could not bucket vega: {exc}")
            return

        # instrument.quote_currency is the venue's inverse-instrument metadata
        # (BTC for a BTC option), not the unit a quote-settled model's output
        # is actually in -- that's a genuinely dollar-scale linear payoff.
        unit = instrument.settlement_currency if spec.settles_in_base else "quote currency"
        rows = [
            {
                "Bucket": b.label, "k range": f"[{b.k_range[0]:.2f}, {b.k_range[1]:.2f}]",
                "Points": b.points_used, f"Vega ({unit})": round(b.vega, 6),
            }
            for b in result.buckets
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            f"Bucketed sum {result.total_bucketed:,.6f} vs parallel vega "
            f"{result.parallel_vega:,.6f} {unit} ({result.reconciliation_error:.2%} apart)."
        )


def _shock_block(currency, model, spec, instrument, slice_, forward, args) -> None:
    with st.expander("Shock ladder"):
        dvol_series = data.dvol_for(currency)
        level_magnitudes = level_shock_magnitudes_from_dvol(dvol_series, slice_.a)
        sourced_from_dvol = bool(level_magnitudes)
        if not level_magnitudes:
            level_magnitudes = tuple(slice_.a * m for m in DEFAULT_LEVEL_MULTIPLES)

        ladder = build_ladder(
            slice_, model, forward, args.strike, args.rate, args.cp,
            level_magnitudes=level_magnitudes,
        )
        # instrument.quote_currency is the venue's inverse-instrument metadata
        # (BTC for a BTC option), not the unit a quote-settled model's output
        # is actually in -- that's a genuinely dollar-scale linear payoff.
        unit = instrument.settlement_currency if spec.settles_in_base else "quote currency"
        rows = [
            {
                "Shock": r.definition.label, "Factor": r.definition.factor.value,
                "Vol": round(r.shocked_vol, 4), f"Price ({unit})": round(r.shocked_price, 6),
                f"PnL ({unit})": round(r.pnl, 6),
            }
            for r in ladder.rungs
        ]
        figure = go.Figure(
            go.Bar(x=[r["Shock"] for r in rows], y=[r[f"PnL ({unit})"] for r in rows])
        )
        figure.update_layout(
            xaxis_title="Shock", yaxis_title=f"P&L ({unit})", height=360,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(figure, width="stretch")
        st.dataframe(rows, hide_index=True, width="stretch")
        source = "Deribit's own DVOL history" if sourced_from_dvol else "a documented default"
        st.caption(
            "Level, skew, and curvature bump the fitted SVI parameters (a, rho, sigma) "
            f"directly: three distinctly-shaped moves, not a parallel vol shift. "
            f"Level magnitude sourced from {source}. Skew and curvature shock "
            "magnitudes are model-implied, not empirically decomposed: Deribit's public "
            "API has no bulk historical-chain endpoint to decompose."
        )


def _portfolio_block(universe, currency, marks, spec, model) -> None:
    with st.expander("Portfolio (session-only, not persisted)"):
        options = universe.options(currency)
        names = [i.name for i in options if i.strike is not None and i.expiry is not None]
        selected = st.multiselect("Positions", names, key="greeks_portfolio_positions")
        if not selected:
            st.caption("Select contracts to aggregate their risk.")
            return

        slice_cache: dict = {}
        positions: list[Position] = []
        for name in selected:
            inst = next(i for i in options if i.name == name)
            size = st.number_input(f"Size: {name}", value=1.0, step=1.0, key=f"greeks_size_{name}")
            slice_ = slice_cache.setdefault(
                inst.expiry, _fit_slice(universe, marks, currency, inst.expiry)
            )
            if slice_ is None:
                continue
            forward = universe.forward_for(currency, inst.expiry)
            spot = universe.index(currency)
            args = pricing.inputs_for(universe, inst, spec)
            vol = slice_.vol(log_moneyness(inst.strike, forward))
            try:
                coin_price = model.price(args.underlying, args.strike, args.tau, vol, args.rate, args.cp)
                coin = model.greeks(args.underlying, args.strike, args.tau, vol, args.rate, args.cp)
            except (ArbitrageError, SolverError):
                continue
            cash = cash_greeks_from_coin(coin, coin_price, forward, spot) if spec.settles_in_base else coin
            positions.append(
                Position(
                    instrument_name=name, currency=currency, expiry=inst.expiry, strike=inst.strike,
                    cp=args.cp, size=size, settles_in_base=spec.settles_in_base,
                    coin_greeks=coin, cash_greeks=cash,
                )
            )

        if not positions:
            st.caption("None of the selected contracts could be priced.")
            return

        cash_totals = aggregate_greeks(positions, use_cash=True)
        st.caption("Cash-denominated Greeks, summable across positions even if they settle in different units.")
        st.dataframe(
            [
                {
                    "Delta (cash)": round(cash_totals.delta, 4), "Gamma (cash)": round(cash_totals.gamma, 6),
                    "Vega (cash)": round(cash_totals.vega, 4), "Theta (cash)": round(cash_totals.theta, 4),
                    "Vanna (cash)": round(cash_totals.vanna, 6), "Volga (cash)": round(cash_totals.volga, 6),
                }
            ],
            hide_index=True, width="stretch",
        )
        try:
            native_totals = aggregate_greeks(positions, use_cash=False)
            st.caption(f"Native-unit (coin) delta, only valid if every position shares one settlement currency: {native_totals.delta:,.4f}")
        except PortfolioError as exc:
            st.caption(f"Native-unit aggregation unavailable: {exc}")
