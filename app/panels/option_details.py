"""Option details: model price, implied vol, and the round-trip residual."""

from __future__ import annotations

import streamlit as st

from app import data, pricing, state
from app.registry import Slot, panel
from voltk import models
from voltk.marketdata import MarketDataError
from voltk.models.bounds import ArbitrageError
from voltk.universe import UniverseError

@panel(
    key="option_details",
    title="Option Details",
    slot=Slot.LEFT_RAIL,
    order=30,
    milestone="M1",
    caption="One contract, live: price, implied vol, and the gap between them on the round trip.",
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
    expiries = universe.expiries(currency)
    if not expiries:
        st.caption("No dated options captured.")
        return

    expiry = st.selectbox(
        "Expiry", expiries, format_func=lambda e: e.strftime("%d %b %Y"), key="detail_expiry"
    )
    strikes = universe.strikes(currency, expiry)
    if not strikes:
        st.caption("No strikes at this expiry.")
        return

    forward = universe.forward_for(currency, expiry)
    nearest = min(range(len(strikes)), key=lambda i: abs(strikes[i] - forward))
    strike = st.selectbox("Strike", strikes, index=nearest, key="detail_strike")
    right = st.radio("Right", ["call", "put"], horizontal=True, key="detail_right")

    instrument = next(
        (
            i
            for i in universe.options(currency)
            if i.expiry == expiry
            and i.strike == strike
            and i.option_type is not None
            and i.option_type.value == right
        ),
        None,
    )
    if instrument is None:
        st.caption("No listed contract at that strike and right.")
        return

    s.instrument_name = instrument.name
    spec = models.get(s.model_key or models.all_models()[0].key)
    if not spec.implemented:
        st.info(f"{spec.display_name} arrives at {spec.milestone}.")
        return

    model = spec.build()
    args = pricing.inputs_for(universe, instrument, spec)

    st.caption(
        f"{instrument.name} · settles {instrument.settlement_currency} · "
        f"size {instrument.contract_size} · tick {instrument.tick_size}"
    )
    st.caption(
        f"tau {args.tau:.4f}y · forward {forward:,.2f} · rate {args.rate:.3%} "
        f"(basis-implied) · {spec.underlying.value} basis"
    )

    vol = st.slider("Vol", 0.05, 3.0, 0.6, 0.01, key="detail_vol")
    try:
        price = model.price(args.underlying, args.strike, args.tau, vol, args.rate, args.cp)
    except Exception as exc:
        st.error(str(exc))
        return

    # instrument.quote_currency is the venue's own metadata for the traded
    # contract (BTC for a BTC option, even though it's an inverse instrument),
    # not a statement about what unit a chosen quote-settled *model*'s output
    # is in -- Black-Scholes etc. price a linear payoff on (forward, strike)
    # and land on a genuinely dollar-scale number, so the label is generic.
    unit = instrument.settlement_currency if spec.settles_in_base else "quote currency"
    st.metric(f"Model price ({unit})", f"{price:,.6f}")

    result = pricing.solve_implied(model, price, args)
    if result is None:
        st.warning("No implied vol exists for that price.")
    elif not result.identified:
        st.warning(result.describe())
    else:
        st.caption(f"Round trip: {result.describe()}")

    try:
        bounds = model.bounds(args.underlying, args.strike, args.tau, args.rate, args.cp)
        st.caption(f"No-arbitrage band ({unit}) [{bounds.lower:,.6f}, {bounds.upper:,.6f}]")
    except ArbitrageError as exc:
        st.error(str(exc))
