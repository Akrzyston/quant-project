"""Vol-carry position: a delta-hedged short straddle sized off the measured
variance risk premium, with capacity, risk, kill conditions, and live P&L
against an entry marked in this session.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import streamlit as st

from app import data, state
from app.registry import Slot, panel
from voltk import models
from voltk.forward import implied_forward_curve
from voltk.greeks import cash_greeks_from_coin
from voltk.instruments import OptionType
from voltk.marketdata import MarketDataError
from voltk.models.base import CP
from voltk.models.bounds import ArbitrageError
from voltk.pnl import attribute_pnl
from voltk.portfolio import Position, aggregate_greeks
from voltk.realized_vol import close_to_close, infer_periods_per_year
from voltk.strategy import (
    capacity_from_open_interest,
    check_kill_conditions,
    expected_daily_edge,
    hedge_delta,
    size_for_vega_budget,
    with_hedge,
)
from voltk.surface import SurfaceError, calibrate_svi_slice, log_moneyness, smile_points
from voltk.universe import UniverseError

REALIZED_LOOKBACK_DAYS = 7
REALIZED_CANDLE_RESOLUTION = "60"
SCRATCH_KEY = "m8_straddle_entry"


@panel(
    key="position_view",
    title="Position",
    slot=Slot.MAIN,
    order=24,
    milestone="M8",
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
    spec = models.get(s.model_key or models.all_models()[0].key)
    if not spec.implemented:
        st.info(f"{spec.display_name} arrives at {spec.milestone}.")
        return
    if not spec.settles_in_base:
        st.caption("A vol-carry position on a Deribit option needs a coin-settled model -- select the inverse model.")
        return
    model = spec.build()

    marks = data.marks_for(currency)
    curve = implied_forward_curve(universe, marks, currency=currency)
    if not curve:
        st.caption("No two-sided marks; nothing to build a straddle from.")
        return

    expiry_by_label = {f.expiry.strftime("%d %b %Y"): f for f in curve}
    label = st.selectbox("Expiry", list(expiry_by_label), key="position_expiry")
    implied_forward = expiry_by_label[label]

    points = smile_points(universe, marks, implied_forward)
    try:
        fit = calibrate_svi_slice(points, currency=currency, expiry=implied_forward.expiry, tau=_tau(implied_forward.expiry, universe.as_of))
    except SurfaceError as exc:
        st.warning(str(exc))
        return

    strikes = universe.strikes(currency, implied_forward.expiry)
    if not strikes:
        st.caption("No listed strikes at this expiry.")
        return
    default_strike = min(strikes, key=lambda k: abs(k - implied_forward.forward))
    strike = st.selectbox("Strike", sorted(strikes), index=sorted(strikes).index(default_strike), key="position_strike")

    call_inst, put_inst = _find_legs(universe, currency, implied_forward.expiry, strike)
    if call_inst is None or put_inst is None:
        st.caption("Both legs of the straddle need to be listed at this strike.")
        return

    tau = _tau(implied_forward.expiry, universe.as_of)
    vol = fit.vol(log_moneyness(strike, implied_forward.forward))
    rate = universe.implied_rate(currency, implied_forward.expiry)
    underlying = implied_forward.forward

    try:
        call_price = model.price(underlying, strike, tau, vol, rate, CP.CALL)
        call_greeks = model.greeks(underlying, strike, tau, vol, rate, CP.CALL)
        put_price = model.price(underlying, strike, tau, vol, rate, CP.PUT)
        put_greeks = model.greeks(underlying, strike, tau, vol, rate, CP.PUT)
    except ArbitrageError as exc:
        st.error(str(exc))
        return

    quotes = data.quotes_for(currency)
    call_oi = getattr(quotes.get(call_inst.name), "open_interest", None) or 0.0
    put_oi = getattr(quotes.get(put_inst.name), "open_interest", None) or 0.0

    dvol_series = data.dvol_for(currency, lookback_hours=REALIZED_LOOKBACK_DAYS * 24)
    entry_vol = dvol_series.latest.close if dvol_series and dvol_series.latest else vol
    realized_vol = _realized_vol(universe, currency)

    _thesis_block(entry_vol, realized_vol)

    size, vega_budget = _sizing_block(call_greeks, put_greeks, currency)
    if size == 0:
        return

    spot = universe.index(currency)
    call_cash = cash_greeks_from_coin(call_greeks, call_price, underlying, spot)
    put_cash = cash_greeks_from_coin(put_greeks, put_price, underlying, spot)
    call_position = Position(
        instrument_name=call_inst.name, currency=currency, expiry=implied_forward.expiry, strike=strike,
        cp=CP.CALL, size=-size, settles_in_base=True,
        coin_greeks=call_greeks, cash_greeks=call_cash,
    )
    put_position = Position(
        instrument_name=put_inst.name, currency=currency, expiry=implied_forward.expiry, strike=strike,
        cp=CP.PUT, size=-size, settles_in_base=True,
        coin_greeks=put_greeks, cash_greeks=put_cash,
    )
    option_greeks = aggregate_greeks([call_position, put_position], use_cash=False)
    hedge = hedge_delta(option_greeks.delta)
    combined = with_hedge(option_greeks, hedge)

    _risk_block(combined, hedge, underlying, realized_vol)
    _capacity_block(size, call_oi, put_oi)
    kill_checks = _kill_block(realized_vol, entry_vol, tau, vega_budget)
    _entry_pnl_block(
        model, underlying, strike, tau, vol, rate, size, hedge,
        call_price, put_price, call_greeks, put_greeks,
    )

    st.caption(
        "Loses if realized vol runs hot enough that gamma losses from "
        "delta-hedging outpace theta collected -- a vol spike is the losing "
        "scenario this position is exposed to, not a slow grind against it. "
        f"{sum(1 for c in kill_checks if c.breached)} of {len(kill_checks)} "
        "kill conditions currently breached."
    )


def _tau(expiry: datetime, as_of: datetime) -> float:
    return max((expiry - as_of).total_seconds() / (365.0 * 24 * 3600), 0.0)


def _find_legs(universe, currency, expiry, strike):
    call_inst = next(
        (i for i in universe.options(currency) if i.expiry == expiry and i.strike == strike and i.option_type is OptionType.CALL),
        None,
    )
    put_inst = next(
        (i for i in universe.options(currency) if i.expiry == expiry and i.strike == strike and i.option_type is OptionType.PUT),
        None,
    )
    return call_inst, put_inst


def _realized_vol(universe, currency: str) -> float | None:
    futures = [f for f in universe.futures(currency) if f.expiry is None]
    if not futures:
        return None
    end = datetime.now(UTC)
    start = end - timedelta(days=REALIZED_LOOKBACK_DAYS)
    candles = data.candles_for(futures[0].name, start, end, REALIZED_CANDLE_RESOLUTION)
    if candles is None or len(candles.candles) < 2:
        return None
    periods = infer_periods_per_year(candles.candles)
    return close_to_close(candles.candles, periods_per_year=periods).value


def _thesis_block(entry_vol: float, realized_vol: float | None) -> None:
    cols = st.columns(3)
    cols[0].metric("Implied (DVOL)", f"{entry_vol:.2%}")
    cols[1].metric(
        f"Realized ({REALIZED_LOOKBACK_DAYS}d)",
        f"{realized_vol:.2%}" if realized_vol is not None else "—",
    )
    premium = entry_vol - realized_vol if realized_vol is not None else None
    cols[2].metric("Premium", f"{premium:+.2%}" if premium is not None else "—")
    st.caption(
        "Thesis: sell vol when implied sits above realized, delta-hedge to isolate "
        "the bet from direction. A negative premium here is a reason not to put "
        "this position on, not a number to explain away."
    )


def _sizing_block(call_greeks, put_greeks, currency: str) -> tuple[float, float]:
    vega_per_straddle = abs(call_greeks.vega) + abs(put_greeks.vega)
    vega_budget = st.number_input(
        f"Vega risk budget ({currency}, coin)", min_value=0.0001, value=1.0, step=0.1, key="position_vega_budget"
    )
    size = size_for_vega_budget(vega_per_straddle, vega_budget)
    st.metric("Straddles (short)", f"{size:g}", help=f"vega/straddle {vega_per_straddle:.6f}")
    return size, vega_budget


def _risk_block(combined, hedge: float, underlying: float, realized_vol: float | None) -> None:
    with st.expander("Combined risk (options + hedge)", expanded=True):
        rows = [{"Greek": f, "Value": round(getattr(combined, f), 8)} for f in
                ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(f"Hedge: {hedge:+.6f} units of the underlying to bring delta to zero.")

        if realized_vol is not None:
            edge = expected_daily_edge(combined.gamma, combined.theta, underlying, realized_vol)
            cols = st.columns(3)
            cols[0].metric("Expected gamma P&L (1d)", f"{edge.gamma_pnl:+.6f}")
            cols[1].metric("Expected theta P&L (1d)", f"{edge.theta_pnl:+.6f}")
            cols[2].metric("Expected edge (1d)", f"{edge.total:+.6f}")


def _capacity_block(size: float, call_oi: float, put_oi: float) -> None:
    binding_oi = min(call_oi, put_oi)
    if binding_oi <= 0:
        st.caption("No open interest reported for one leg; capacity unknown.")
        return
    capacity = capacity_from_open_interest(binding_oi)
    cols = st.columns(2)
    cols[0].metric("Capacity (10% of OI)", f"{capacity.contracts:.1f}")
    cols[1].metric("Sized position vs. capacity", f"{size / capacity.contracts:.0%}" if capacity.contracts else "—")


def _kill_block(realized_vol: float | None, entry_vol: float, tau: float, vega_budget: float):
    st.caption("Kill conditions")
    entry = state.get().scratch.get(SCRATCH_KEY)
    checks = check_kill_conditions(
        realized_vol=realized_vol if realized_vol is not None else 0.0,
        entry_vol=entry_vol, vol_stop_multiple=1.5,
        pnl=entry.get("pnl", 0.0) if entry else 0.0,
        risk_budget=vega_budget, loss_stop_fraction=0.5,
        tau=tau, min_tau=3.0 / 365.0,
    )
    rows = [{"Condition": c.name, "Breached": c.breached, "Detail": c.detail} for c in checks]
    st.dataframe(rows, hide_index=True, width="stretch")
    return checks


def _entry_pnl_block(
    model, underlying, strike, tau, vol, rate, size, hedge,
    call_price, put_price, call_greeks, put_greeks,
) -> None:
    with st.expander("Entry and live P&L (session-only, not persisted)"):
        scratch = state.get().scratch
        entry = scratch.get(SCRATCH_KEY)
        if st.button("Mark position as entered here"):
            entry = {
                "underlying": underlying, "tau": tau, "vol": vol,
                "call_price": call_price, "put_price": put_price,
                "call_greeks": call_greeks, "put_greeks": put_greeks,
                "size": size, "hedge": hedge,
            }
            scratch[SCRATCH_KEY] = entry

        if not entry:
            st.caption("Not entered yet -- everything above is a live preview.")
            return

        d_underlying = underlying - entry["underlying"]
        d_t = max(entry["tau"] - tau, 0.0)
        try:
            call_now = model.price(underlying, strike, tau, vol, rate, CP.CALL)
            put_now = model.price(underlying, strike, tau, vol, rate, CP.PUT)
        except ArbitrageError:
            st.caption("Can't reprice at the current market; leaving the last known P&L.")
            return

        call_pnl = attribute_pnl(entry["call_greeks"], entry["call_price"], call_now, d_underlying, vol - entry["vol"], d_t)
        put_pnl = attribute_pnl(entry["put_greeks"], entry["put_price"], put_now, d_underlying, vol - entry["vol"], d_t)
        option_pnl = -entry["size"] * (call_pnl.actual_pnl + put_pnl.actual_pnl)
        hedge_pnl = entry["hedge"] * d_underlying
        total_pnl = option_pnl + hedge_pnl
        scratch[SCRATCH_KEY]["pnl"] = total_pnl

        cols = st.columns(3)
        cols[0].metric("Option P&L", f"{option_pnl:+.6f}")
        cols[1].metric("Hedge P&L", f"{hedge_pnl:+.6f}")
        cols[2].metric("Total P&L", f"{total_pnl:+.6f}")
