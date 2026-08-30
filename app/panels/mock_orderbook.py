"""Mock orderbook: theoretical value and a derived two-sided quote, rendered
against the live market, plus a simulated market-making session with the
resulting P&L attributed rather than assumed.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import streamlit as st

from app import data, pricing, state
from app.registry import Slot, panel
from voltk import models
from voltk.forward import implied_forward_curve
from voltk.market_making import SessionStep, simulate_session
from voltk.marketdata import MarketDataError
from voltk.models.bounds import ArbitrageError
from voltk.quoting import classify_against_market, derive_width, quote_ladder, reservation_price
from voltk.surface import SurfaceError, calibrate_svi_slice, log_moneyness, smile_points
from voltk.universe import UniverseError

SESSION_WINDOW_HOURS = 6
SESSION_RESOLUTION = "15"
MARKOUT_STEPS = 2


@panel(
    key="mock_orderbook",
    title="Mock Orderbook",
    slot=Slot.QUOTER_MAIN,
    order=10,
    milestone="M7",
)
def render() -> None:
    s = state.get()
    if not s.currencies or not s.instrument_name:
        st.caption("Select a contract to see its book.")
        return

    try:
        universe, _ = data.active_universe()
    except (MarketDataError, UniverseError) as exc:
        data.show_error(exc)
        return

    currency = s.currency or s.currencies[0]
    instrument = next((i for i in universe.options(currency) if i.name == s.instrument_name), None)
    if instrument is None or instrument.strike is None or instrument.expiry is None:
        st.caption("No listed contract at that selection.")
        return

    spec = models.get(s.model_key or models.all_models()[0].key)
    if not spec.implemented:
        st.info(f"{spec.display_name} arrives at {spec.milestone}.")
        return
    if not spec.settles_in_base:
        st.caption(
            "Quoting against Deribit's own market needs a coin-settled model -- "
            "select the inverse model to quote this contract."
        )
        return
    model = spec.build()
    args = pricing.inputs_for(universe, instrument, spec)

    marks = data.marks_for(currency)
    curve = implied_forward_curve(universe, marks, currency=currency, expiries=[instrument.expiry])
    if not curve:
        st.caption("No implied forward for this expiry yet.")
        return
    implied_forward = curve[0]

    points = smile_points(universe, marks, implied_forward)
    try:
        fit = calibrate_svi_slice(points, currency=currency, expiry=instrument.expiry, tau=args.tau)
    except SurfaceError as exc:
        st.warning(str(exc))
        return

    vol = fit.vol(log_moneyness(instrument.strike, implied_forward.forward))
    match = next(
        (p for p in points if p.strike == instrument.strike and p.option_type == args.cp), None
    )
    fit_residual = match.market_vol - vol if match and match.identified else 0.0

    try:
        theo = model.price(args.underlying, args.strike, args.tau, vol, args.rate, args.cp)
        greeks = model.greeks(args.underlying, args.strike, args.tau, vol, args.rate, args.cp)
    except ArbitrageError as exc:
        st.error(str(exc))
        return

    quote = data.quotes_for(currency).get(instrument.name)
    market_half_spread = (
        (quote.ask - quote.bid) / 2.0 if quote and quote.bid is not None and quote.ask is not None else None
    )

    q = s.quote
    width = derive_width(
        vega=greeks.vega, gamma=greeks.gamma, fit_residual_vol=fit_residual,
        underlying=args.underlying, vol=vol, market_half_spread=market_half_spread,
        fit_coef=q.fit_coef, gamma_coef=q.gamma_coef, liquidity_coef=q.liquidity_coef, floor=q.floor_dollar,
    )

    _width_block(theo, width, fit_residual)
    _ladder_block(theo, width, q, args.tau, vol, quote)
    _session_block(universe, model, instrument, currency, args, vol, width.half_width)


def _width_block(theo: float, width, fit_residual: float) -> None:
    cols = st.columns(4)
    cols[0].metric("Theo (coin)", f"{theo:.6f}")
    cols[1].metric("Fit term", f"{width.fit_term:.6f}", help=f"fit residual {fit_residual:+.4f} vol")
    cols[2].metric("Gamma term", f"{width.gamma_term:.6f}")
    cols[3].metric("Liquidity term", f"{width.liquidity_term:.6f}")


def _ladder_block(theo: float, width, q, tau: float, vol: float, quote) -> None:
    inventory = st.number_input("Preview inventory (contracts)", value=0.0, step=1.0)
    mid = reservation_price(theo, inventory, q.risk_aversion, vol, tau)
    ladder = quote_ladder(mid, width.half_width, levels=q.levels, size=q.size, level_growth=q.level_growth)

    rows = [{"Level": lvl.level, "Bid": round(lvl.bid, 6), "Ask": round(lvl.ask, 6), "Size": lvl.size} for lvl in ladder]
    st.dataframe(rows, hide_index=True, width="stretch")

    if quote and quote.bid is not None and quote.ask is not None:
        position = classify_against_market(ladder[0].bid, ladder[0].ask, quote.bid, quote.ask)
        st.metric("Vs. live market", position.value, help=f"live {quote.bid:.6f} / {quote.ask:.6f}")
    else:
        st.caption("No two-sided live market to compare against right now.")


def _session_block(universe, model, instrument, currency: str, args, vol: float, half_width: float) -> None:
    with st.expander("Simulated market-making session"):
        futures = [f for f in universe.futures(currency) if f.expiry is None]
        if not futures:
            st.caption("Perpetual instrument not found; can't build a spot path.")
            return

        end = datetime.now(UTC)
        start = end - timedelta(hours=SESSION_WINDOW_HOURS)
        candles = data.candles_for(futures[0].name, start, end, SESSION_RESOLUTION)
        if candles is None or len(candles.candles) < MARKOUT_STEPS + 2:
            st.caption("Not enough intraday candles right now.")
            return

        rate = universe.implied_rate(currency, instrument.expiry)
        year_seconds = 365.0 * 24 * 3600

        def _step(c) -> SessionStep:
            tau = max((instrument.expiry - c.timestamp).total_seconds() / year_seconds, 0.0)
            forward = c.close * math.exp(rate * tau)  # perpetual close, basis-adjusted like everywhere else
            return SessionStep(timestamp=c.timestamp, underlying=forward, tau=tau, vol=vol)

        steps = tuple(_step(c) for c in candles.candles)

        result = simulate_session(
            model, steps, strike=args.strike, rate=rate, cp=args.cp,
            half_width=half_width, risk_aversion=state.get().quote.risk_aversion, markout_steps=MARKOUT_STEPS,
        )

        if not result.fills:
            st.caption(f"No fills over the trailing {SESSION_WINDOW_HOURS}h at this width.")
            return

        cols = st.columns(5)
        cols[0].metric("Fills", len(result.fills))
        cols[1].metric("Edge captured", f"{result.total_edge:+.6f}")
        cols[2].metric("Adverse selection", f"{result.total_adverse_selection:+.6f}")
        cols[3].metric("Vega P&L", f"{result.total_vega_pnl:+.6f}")
        cols[4].metric("Total P&L", f"{result.total_pnl:+.6f}")
        st.caption(
            f"Ending inventory {result.ending_inventory:+g} contracts. Residual "
            f"{result.total_residual:+.6f} (theta plus omitted cross-terms), never folded "
            "into the other figures. Fills assume a maximally-informed counterparty -- an "
            "upper bound on adverse selection, not an average session."
        )
