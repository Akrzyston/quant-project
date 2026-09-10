"""Implied vs realized volatility, the variance risk premium, and an
empirical test of sticky-strike against sticky-delta.

All three sections read from Deribit's history endpoints directly -- they are
live-only, the same established choice as the DVOL cross-check on the Smile
panel (data.dvol_for), since a snapshot stores one instant, not a history.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import plotly.graph_objects as go
import streamlit as st

from app import data, state
from app.registry import Slot, panel
from voltk.forward import implied_forward_curve
from voltk.instruments import OptionType
from voltk.marketdata import MarketDataError
from voltk.realized_vol import RealizedVolError, close_to_close, infer_periods_per_year, parkinson
from voltk.sticky_regime import StickyRegimeError, observations_from_candles, regress_vol_on_spot
from voltk.surface import Surface, calibrate_surface
from voltk.universe import UniverseError
from voltk.variance_premium import summarise_premium, variance_risk_premium

REALIZED_LOOKBACK_DAYS = 16
PERPETUAL_CANDLE_RESOLUTION = "60"
INTRADAY_CANDLE_RESOLUTION = "15"
INTRADAY_WINDOW_HOURS = 24
STRIKE_LADDER_SIZE = 3


@panel(
    key="vol_history",
    title="Vol History",
    slot=Slot.MAIN,
    order=22,
    milestone="M6",
    caption="Implied against realized vol over time, the resulting premium, and which sticky-vol regime actually holds. This is the signal the Strategy tab trades on.",
)
def render() -> None:
    s = state.get()
    if not s.currencies:
        st.caption("Select a currency first.")
        return
    currency = s.currency or s.currencies[0]

    _implied_vs_realized_block(currency)
    _variance_premium_block(currency)
    _sticky_regime_block(currency)


def _implied_vs_realized_block(currency: str) -> None:
    with st.expander("Implied vs realized", expanded=True):
        dvol_series = data.dvol_for(currency, lookback_hours=REALIZED_LOOKBACK_DAYS * 24)
        realized_series = data.historical_vol_for(currency)

        if not dvol_series or not dvol_series.points:
            st.caption("DVOL history unavailable right now.")
            return
        if not realized_series or not realized_series.points:
            st.caption("Deribit's realized-vol history unavailable right now.")
            return

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=[p.timestamp for p in dvol_series.points], y=[p.close for p in dvol_series.points],
                name="DVOL (implied)", mode="lines",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[p.timestamp for p in realized_series.points], y=[p.value for p in realized_series.points],
                name="Deribit realized vol", mode="lines",
            )
        )
        figure.update_layout(
            xaxis_title="Time", yaxis_title="Annualized vol", height=380,
            margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(figure, width="stretch")

        _reconciliation(currency, realized_series)

        st.caption(
            "Deribit's own realized figure is a black box: undocumented sampling "
            "frequency, window length and annualisation convention. It will not "
            "match a figure computed independently even from the same underlying "
            "price path; the gap is expected, not a bug in either number."
        )


def _reconciliation(currency: str, realized_series) -> None:
    universe_futures = None
    try:
        universe, _ = data.active_universe()
        universe_futures = [f for f in universe.futures(currency) if f.expiry is None]
    except (MarketDataError, UniverseError):
        pass
    if not universe_futures:
        st.caption("Perpetual instrument not found; skipping the own-computed reconciliation.")
        return
    perpetual_name = universe_futures[0].name

    end = datetime.now(UTC)
    start = end - timedelta(days=REALIZED_LOOKBACK_DAYS)
    candles = data.candles_for(perpetual_name, start, end, PERPETUAL_CANDLE_RESOLUTION)
    if candles is None or len(candles.candles) < 2:
        st.caption("Perpetual candle history unavailable right now.")
        return

    try:
        periods_per_year = infer_periods_per_year(candles.candles)
        own_close_to_close = close_to_close(candles.candles, periods_per_year=periods_per_year)
        own_parkinson = parkinson(candles.candles, periods_per_year=periods_per_year)
    except RealizedVolError as exc:
        st.caption(f"Could not compute an own realized-vol estimate: {exc}")
        return

    deribit_latest = realized_series.points[-1].value
    cols = st.columns(3)
    cols[0].metric("Deribit realized vol (latest)", f"{deribit_latest:.2%}")
    cols[1].metric(
        "Own close-to-close", f"{own_close_to_close.value:.2%}",
        delta=f"{(own_close_to_close.value - deribit_latest) * 10_000:.0f} bps",
    )
    cols[2].metric(
        "Own Parkinson (range)", f"{own_parkinson.value:.2%}",
        delta=f"{(own_parkinson.value - deribit_latest) * 10_000:.0f} bps",
    )
    st.caption(
        f"Own estimates from {own_close_to_close.n_observations} hourly perpetual candles "
        f"over the trailing {REALIZED_LOOKBACK_DAYS} days, close-to-close and Parkinson "
        "range estimators. Deribit's own figure is not disclosed to use the same window "
        "or sampling frequency, so an exact match is not expected."
    )


def _variance_premium_block(currency: str) -> None:
    with st.expander("Variance risk premium (implied - realized)"):
        dvol_series = data.dvol_for(currency, lookback_hours=REALIZED_LOOKBACK_DAYS * 24)
        realized_series = data.historical_vol_for(currency)
        if not dvol_series or not realized_series:
            st.caption("Need both DVOL and realized-vol history to compute a premium.")
            return

        points = variance_risk_premium(dvol_series.points, realized_series.points)
        summary = summarise_premium(points)
        if summary is None:
            st.caption("No aligned observations within the matching tolerance.")
            return

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=[p.timestamp for p in points], y=[p.premium for p in points],
                        name="Premium (implied - realized)", mode="lines")
        )
        figure.add_hline(y=0.0, line_dash="dot")
        figure.update_layout(
            xaxis_title="Time", yaxis_title="Vol premium", height=300,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(figure, width="stretch")

        cols = st.columns(3)
        cols[0].metric("Mean premium", f"{summary.mean_premium:+.2%}")
        cols[1].metric("Fraction of window positive", f"{summary.fraction_positive:.0%}")
        cols[2].metric("Inversions", len(summary.inversions))
        st.caption(
            "A persistently positive premium is the textbook variance risk premium: "
            "sellers of volatility are compensated on average for bearing realized-vol "
            "risk. Inversions are the timestamps that compensation ran negative."
        )


def _sticky_regime_block(currency: str) -> None:
    with st.expander("Sticky-strike vs sticky-delta, measured"):
        try:
            universe, _ = data.active_universe()
        except (MarketDataError, UniverseError) as exc:
            data.show_error(exc)
            return

        marks = data.marks_for(currency)
        curve = implied_forward_curve(universe, marks, currency=currency)
        if not curve:
            st.caption("No two-sided marks; nothing to fit a reference surface from.")
            return

        expiry_by_label = {f.expiry.strftime("%d %b %Y"): f for f in curve}
        label = st.selectbox("Expiry", list(expiry_by_label), key="vol_history_expiry")
        implied_forward = expiry_by_label[label]
        expiry, forward = implied_forward.expiry, implied_forward.forward

        slices, _ = calibrate_surface(universe, marks, curve)
        if not slices:
            st.caption("No expiry calibrated; cannot compute the theoretical sticky-delta prediction.")
            return
        surface = Surface(slices=slices)

        strikes = universe.strikes(currency, expiry)
        if len(strikes) < STRIKE_LADDER_SIZE:
            st.caption("Not enough listed strikes at this expiry for a ladder.")
            return
        ladder = sorted(strikes, key=lambda k: abs(k - forward))[:STRIKE_LADDER_SIZE]

        futures = [f for f in universe.futures(currency) if f.expiry is None]
        if not futures:
            st.caption("Perpetual instrument not found; needed as the spot leg.")
            return
        perpetual_name = futures[0].name

        end = datetime.now(UTC)
        start = end - timedelta(hours=INTRADAY_WINDOW_HOURS)
        perp_candles = data.candles_for(perpetual_name, start, end, INTRADAY_CANDLE_RESOLUTION)
        if perp_candles is None or len(perp_candles.candles) < 3:
            st.caption("Not enough intraday perpetual candles right now.")
            return

        rate = universe.implied_rate(currency, expiry)
        tau = (expiry - universe.as_of).total_seconds() / (365 * 24 * 3600)

        rows = []
        scatter = go.Figure()
        for strike in ladder:
            call_inst = next(
                (i for i in universe.options(currency) if i.expiry == expiry and i.strike == strike
                 and i.option_type is OptionType.CALL),
                None,
            )
            put_inst = next(
                (i for i in universe.options(currency) if i.expiry == expiry and i.strike == strike
                 and i.option_type is OptionType.PUT),
                None,
            )
            if call_inst is None or put_inst is None:
                continue
            call_candles = data.candles_for(call_inst.name, start, end, INTRADAY_CANDLE_RESOLUTION)
            put_candles = data.candles_for(put_inst.name, start, end, INTRADAY_CANDLE_RESOLUTION)
            if call_candles is None or put_candles is None:
                continue

            observations = observations_from_candles(
                call_candles, put_candles, perp_candles, strike=strike, expiry=expiry
            )
            if len(observations) < 3:
                continue

            theoretical = surface.dvol_dspot_sticky_delta(strike, forward, tau, rate)
            try:
                result = regress_vol_on_spot(observations, strike=strike, sticky_delta_prediction=theoretical)
            except StickyRegimeError:
                continue

            rows.append(
                {
                    "Strike": strike,
                    "Observations": result.n_observations,
                    "Empirical d(vol)/d(spot)": result.slope,
                    "R^2": round(result.r_squared, 3),
                    "Sticky-strike predicts": result.sticky_strike_prediction,
                    "Sticky-delta predicts": round(result.sticky_delta_prediction, 8),
                }
            )
            scatter.add_trace(
                go.Scatter(
                    x=[o.spot for o in observations], y=[o.implied_vol for o in observations],
                    name=f"K={strike:g}", mode="markers",
                )
            )

        if not rows:
            st.caption(
                "Not enough aligned intraday candle observations across the strike "
                "ladder to run the regression right now."
            )
            return

        st.dataframe(rows, hide_index=True, width="stretch")
        scatter.update_layout(
            xaxis_title="Spot (perpetual close)", yaxis_title="Implied vol", height=340,
            margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(scatter, width="stretch")
        st.caption(
            "Empirical slope near the sticky-strike prediction (0) means vol at a fixed "
            "strike held still as spot moved; a slope near the sticky-delta prediction "
            "means the whole smile translated with the forward instead. Regression is "
            "run on first differences (vol change vs spot change) over one session, "
            "not on the levels."
        )
