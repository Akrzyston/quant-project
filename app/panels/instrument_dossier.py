"""Instrument dossier.

Shows what was captured, how clean the capture was, and the contract
specification behind it. This is the M0 deliverable made visible.
"""

from __future__ import annotations

import streamlit as st

from app import data, state
from app.registry import Slot, panel
from voltk import conventions
from voltk.conventions import ConventionError
from voltk.marketdata import MarketDataError
from voltk.validation import parity_report
from voltk.universe import UniverseError

VENUE_SPEC = "deribit"


@panel(
    key="instrument_dossier",
    title="Instrument Dossier",
    slot=Slot.MAIN,
    order=10,
    milestone="M0",
    caption="What actually got captured, in one bracketed window, and how clean that capture was. Everything downstream assumes this is a real instant, not two reads stitched together.",
)
def render() -> None:
    s = state.get()
    if not s.currencies:
        st.caption("Select a currency to capture a universe.")
        return

    try:
        universe, meta = data.active_universe()
    except (MarketDataError, UniverseError) as exc:
        data.show_error(exc)
        return

    quality = meta["quality"]
    st.caption(f"{meta['source']} · as of {universe.as_of:%Y-%m-%d %H:%M:%S}Z")

    cols = st.columns(3)
    cols[0].metric("Capture window", f"{quality.window_ms:.0f} ms")
    cols[1].metric("Index drift", f"{quality.worst_drift_bps:.1f} bps")
    cols[2].metric("Instruments", len(universe.instruments))

    if not quality.ok:
        for reason in quality.reasons:
            st.warning(reason)
        st.caption("A degraded capture is still usable, but the surface may be smeared.")

    summary = data.summarise(universe)
    st.caption(f"Dossier digest {summary['digest'][:16]}")

    focus = s.currency or s.currencies[0]
    if len(s.currencies) > 1:
        tabs = st.tabs(list(s.currencies))
        for tab, currency in zip(tabs, s.currencies):
            with tab:
                _currency_block(universe, summary, currency)
    else:
        _currency_block(universe, summary, focus)

    _parity_block(universe, focus)
    _conventions_block(universe)


def _currency_block(universe, summary, currency: str) -> None:
    block = summary["currencies"].get(currency)
    if not block:
        st.caption("Nothing captured for this currency.")
        return

    cols = st.columns(4)
    cols[0].metric("Options", block["option_count"])
    cols[1].metric("Expiries", block["expiry_count"])
    cols[2].metric("Strikes", block["strike_count"])
    cols[3].metric("Futures", block["future_count"])

    index_price = block["index_price"]
    if index_price is not None:
        st.metric("Index (quote currency)", f"{index_price:,.2f}")

    settlement = "inverse" if block["inverse_count"] else "linear"
    if block["inverse_count"] and block["linear_count"]:
        settlement = f"mixed ({block['inverse_count']} inverse, {block['linear_count']} linear)"
    st.caption(
        f"Settlement: {settlement} · settles in {', '.join(block['settlement_currencies'])} · "
        f"contract size {', '.join(str(c) for c in block['contract_sizes'])} · "
        f"tick {', '.join(str(t) for t in block['tick_sizes'])}"
    )

    forwards = block["forward_points"]
    if forwards:
        st.markdown("**Forward curve**")
        st.dataframe(
            [
                {
                    "Expiry": point["expiry"][:10],
                    "Forward": round(point["forward"], 2),
                    "Basis vs index": (
                        round(point["forward"] - index_price, 2)
                        if index_price is not None
                        else None
                    ),
                    "Instrument": point["instrument"],
                }
                for point in forwards
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No dated futures marked, so forwards fall back to the index.")

    expiries = universe.expiries(currency)
    if expiries:
        rows = [
            {
                "Expiry": expiry.strftime("%Y-%m-%d"),
                "Tau (yrs)": round(
                    (expiry - universe.as_of).total_seconds() / (365 * 24 * 3600), 4
                ),
                "Strikes": len(universe.strikes(currency, expiry)),
                "Forward": round(universe.forward_for(currency, expiry), 2),
            }
            for expiry in expiries
        ]
        st.markdown("**Expiries**")
        st.dataframe(rows, hide_index=True, width="stretch")


def _parity_block(universe, currency: str) -> None:
    marks = data.marks_for(currency)
    if not marks:
        return

    with st.expander("Put-call parity on live marks"):
        report = parity_report(universe, marks, currency=currency)
        st.caption(report.summary())
        if not report.rows:
            return
        st.dataframe(
            [
                {
                    "Expiry": row.expiry.strftime("%Y-%m-%d"),
                    "Strike": row.strike,
                    "C - P": round(row.call_price - row.put_price, 6),
                    "Expected": round(row.call_price - row.put_price - row.gap, 6),
                    "Gap (bps of fwd)": round(row.gap_bps_of_forward, 2),
                    "Breach": row.breached,
                }
                for row in sorted(report.rows, key=lambda r: -abs(r.gap))[:15]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Coin-settled parity is C - P = 1 - K/F. Gaps inside the combined "
            "spread are not evidence of mispricing."
        )


def _conventions_block(universe) -> None:
    with st.expander("Contract specification"):
        try:
            spec = conventions.load(VENUE_SPEC)
        except ConventionError as exc:
            st.error(str(exc))
            return

        st.caption(f"Source: {spec.source_url} · retrieved {spec.retrieved_at}")
        st.caption(
            f"Expiry {spec.expiry_time_utc:%H:%M} UTC · settlement index averaged over "
            f"{spec.settlement_window_minutes} minutes"
        )

        try:
            conventions.validate_expiry_convention(universe.instruments, spec)
        except ConventionError as exc:
            st.error(str(exc))
        else:
            st.success("Live expiries match the documented convention.")

        option_fees = spec.fee_schedule("option")
        st.caption(
            f"Option fee {option_fees.taker_rate:.4%} of underlying, capped at "
            f"{option_fees.cap_fraction_of_premium:.1%} of premium · delivery "
            f"{option_fees.delivery_rate:.4%}"
        )
