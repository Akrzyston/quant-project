"""Vol-carry position: a delta-hedged short straddle sized off the measured
variance risk premium, with capacity, risk, kill conditions, live P&L against
an entry marked in this session, and a historical view of when the signal
would have said to enter or stay out.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import plotly.graph_objects as go
import streamlit as st

from app import data, state
from app.registry import Slot, panel
from voltk import models
from voltk.forward import implied_forward_curve
from voltk.greeks import cash_greeks_from_coin
from voltk.instruments import OptionType
from voltk.market_making import SessionStep
from voltk.marketdata import MarketDataError
from voltk.models.base import CP
from voltk.models.bounds import ArbitrageError
from voltk.pnl import attribute_pnl
from voltk.portfolio import Position, aggregate_greeks
from voltk.realized_vol import close_to_close, infer_periods_per_year, reconcile, rolling_realized_vol
from voltk.strategy import (
    StrategyError,
    autocorrelation_adjusted_sharpe,
    backtest_premium_sharpe,
    capacity_from_open_interest,
    check_kill_conditions,
    expected_daily_edge,
    hedge_delta,
    simulate_short_straddle_path,
    size_for_vega_budget,
    skewness,
    with_hedge,
)
from voltk.surface import SurfaceError, calibrate_svi_slice, log_moneyness, smile_points
from voltk.universe import UniverseError
from voltk.variance_premium import (
    cumulative_captured_premium,
    favorable_windows,
    max_drawdown,
    summarise_premium,
    variance_risk_premium,
)

REALIZED_LOOKBACK_DAYS = 7
REALIZED_CANDLE_RESOLUTION = "60"
BACKTEST_LOOKBACK_DAYS = 60
SENSITIVITY_WINDOWS_DAYS = (7, 14, 30)
SCRATCH_KEY = "m8_straddle_entry"


@panel(
    key="position_view",
    title="Position",
    slot=Slot.FEATURED,
    order=10,
    milestone="M8",
)
def render() -> None:
    _narrative_block()

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
    spec = next((m for m in models.all_models() if m.settles_in_base and m.implemented), None)
    if spec is None:
        st.caption("No coin-settled pricing model is implemented yet.")
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
    _historical_backtest_block(universe, currency)
    _robustness_block(universe, currency)

    size, vega_budget = _sizing_block(call_greeks, put_greeks, currency)
    if size == 0:
        return

    _pnl_path_block(universe, currency, model, strike, implied_forward.expiry, rate, size)

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

    _risk_block(combined, hedge, underlying, realized_vol, currency)
    _capacity_block(size, call_oi, put_oi)
    kill_checks = _kill_block(realized_vol, entry_vol, tau, vega_budget)
    _entry_pnl_block(
        model, underlying, strike, tau, vol, rate, size, hedge,
        call_price, put_price, call_greeks, put_greeks, currency,
    )

    st.caption(
        "Loses if realized vol runs hot enough that gamma losses from "
        "delta-hedging outpace theta collected. A vol spike is the losing "
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


def _perpetual_name(universe, currency: str) -> str | None:
    futures = [f for f in universe.futures(currency) if f.expiry is None]
    return futures[0].name if futures else None


def _realized_vol(universe, currency: str) -> float | None:
    name = _perpetual_name(universe, currency)
    if name is None:
        return None
    end = datetime.now(UTC)
    start = end - timedelta(days=REALIZED_LOOKBACK_DAYS)
    candles = data.candles_for(name, start, end, REALIZED_CANDLE_RESOLUTION)
    if candles is None or len(candles.candles) < 2:
        return None
    periods = infer_periods_per_year(candles.candles)
    return close_to_close(candles.candles, periods_per_year=periods).value


def _narrative_block() -> None:
    st.markdown(
        "Sell volatility when it's priced rich against what's actually "
        "realizing, delta-hedge to isolate that view from direction. This "
        "position is short gamma: it collects theta in small, steady "
        "increments for as long as the underlying doesn't move much, and "
        "loses money in large, fast increments exactly when it does. A "
        "single vol spike can erase weeks of collected theta. The kill "
        "conditions below exist to cut the position before that happens, "
        "not after. Full memo: `docs/strategy_proposal.md`."
    )
    with st.expander("The math behind this", expanded=True):
        st.markdown(
            "Over one interval, a delta-hedged option position earns gamma "
            "P&L from whatever move happens, plus the theta it "
            "already collected:"
        )
        st.latex(r"\text{Edge} = \tfrac{1}{2}\,\Gamma\,(\Delta S)^2 + \Theta\,\Delta t")
        st.caption(
            "`voltk.strategy.expected_daily_edge` computes this from the "
            "position's own combined Greeks. Gamma is negative for a short "
            "straddle (a cost); theta is positive (a credit)."
        )
        st.markdown(
            "Those two terms come from different vols. Theta got locked in "
            "at whatever vol the straddle sold at. The gamma cost only shows "
            "up later, set by how much the underlying really moves. "
            "Sell rich enough relative to that and the "
            "credit outruns the cost. That gap is the entire trade:"
        )
        st.latex(r"\text{VRP} = \sigma_{\text{implied}} - \sigma_{\text{realized}}")
        st.caption(
            "Implied is Deribit's DVOL. Realized is computed here from raw "
            "candles, never taken from the venue's own number. The thesis "
            "metrics above are this exact gap, right now."
        )
        st.markdown(
            "A call and a put at the same strike cancel most of each "
            "other's delta but leave a residual, which is why the position "
            "gets hedged instead of trusted to sit flat on its own:"
        )
        st.latex(r"\Delta_{\text{hedge}} = -\left(\Delta_{\text{call}} + \Delta_{\text{put}}\right)")
        st.caption(
            "`hedge_delta` computes it; the combined delta in the risk table "
            "below is what applying this produces."
        )
        st.markdown(
            "How well this has gone historically shows up as a Sharpe ratio "
            "on the daily premium series: mean over spread, annualized:"
        )
        st.latex(r"\text{Sharpe} = \frac{\bar{p}}{s_p}\sqrt{365}")
        st.caption(
            "p is the daily premium (implied minus realized) charted below. "
            "The number itself is further down, next to why it runs optimistic."
        )
    st.divider()


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


def _historical_backtest_block(universe, currency: str) -> None:
    with st.expander("Historical entry/exit signal", expanded=True):
        name = _perpetual_name(universe, currency)
        if name is None:
            st.caption("No perpetual future listed for this currency; nothing to backtest against.")
            return

        end = datetime.now(UTC)
        start = end - timedelta(days=BACKTEST_LOOKBACK_DAYS)
        candles = data.candles_for(name, start, end, REALIZED_CANDLE_RESOLUTION)
        dvol_series = data.dvol_for(currency, lookback_hours=BACKTEST_LOOKBACK_DAYS * 24)
        if candles is None or len(candles.candles) < 2 or not dvol_series or not dvol_series.points:
            st.caption("Not enough history to backtest right now.")
            return

        # Deribit's own realized-vol history caps at ~16 days regardless of what's
        # requested (confirmed live) -- computed independently from candles instead,
        # so this reaches back the full BACKTEST_LOOKBACK_DAYS. Uses the same 7-day
        # rolling window the live thesis figure above uses, just stepped daily.
        realized_points = rolling_realized_vol(
            candles.candles, window=timedelta(days=REALIZED_LOOKBACK_DAYS), step=timedelta(days=1),
        )
        if not realized_points:
            st.caption("Not enough candle history yet to compute a rolling realized-vol series.")
            return

        points = variance_risk_premium(dvol_series.points, realized_points)
        summary = summarise_premium(points)
        if summary is None:
            st.caption("No aligned implied/realized observations within the matching tolerance.")
            return
        windows = favorable_windows(points)

        st.markdown("**Vol level: implied vs. realized**")
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=[p.timestamp for p in points], y=[p.implied for p in points],
            name="Implied (DVOL)", mode="lines",
        ))
        figure.add_trace(go.Scatter(
            x=[p.timestamp for p in points], y=[p.realized for p in points],
            name=f"Realized ({REALIZED_LOOKBACK_DAYS}d rolling)", mode="lines",
        ))
        for w in windows:
            figure.add_vrect(x0=w.start, x1=w.end, fillcolor="green", opacity=0.12, line_width=0)
        figure.update_layout(
            xaxis_title="Time", yaxis_title="Annualized vol", height=340,
            margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "The two lines are the raw inputs; green is where implied ran "
            "above realized, where a 'sell vol' rule would have said yes."
        )

        published = data.historical_vol_for(currency)
        reconciliation = reconcile(realized_points, published.points) if published else None
        if reconciliation is not None:
            st.caption(
                f"Cross-checked against Deribit's own published realized-vol series "
                f"over the ~16 days it covers: mean gap of {reconciliation.mean_abs_diff:.2%} "
                f"({reconciliation.n} points). Deribit doesn't document its own window "
                "or estimator, so a perfect match isn't the bar. This is the same "
                "reconciliation this project already runs against DVOL: checking the "
                "two land in the same neighborhood, not that they're identical."
            )

        st.markdown("**The signal itself: daily premium**")
        premium_figure = go.Figure(go.Bar(
            x=[p.timestamp for p in points], y=[p.premium for p in points],
            marker_color=["seagreen" if p.premium > 0 else "indianred" for p in points],
            name="Premium",
        ))
        premium_figure.add_hline(y=0, line_width=1, line_color="gray")
        premium_figure.update_layout(
            xaxis_title="Time", yaxis_title="Implied - realized",
            yaxis_tickformat=".0%", height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        )
        st.plotly_chart(premium_figure, width="stretch")
        st.caption(
            "This is the number the trade runs on. Positive (green) "
            "means selling was worth it that day; negative (red) means it wasn't. "
            "The vol-level chart above is what that's built from; this is the "
            "one line that matters for a go/no-go decision."
        )

        result = backtest_premium_sharpe([p.premium for p in points])
        adjusted = autocorrelation_adjusted_sharpe([p.premium for p in points])
        weighted_n = sum(w.n for w in windows)

        cols = st.columns(5)
        cols[0].metric("Days favorable to sell vol", f"{summary.fraction_positive:.0%}")
        cols[1].metric("Distinct entry/exit windows", f"{len(windows)}")
        cols[2].metric(
            "Mean premium while favorable",
            f"{sum(w.mean_premium * w.n for w in windows) / weighted_n:+.2%}" if weighted_n else "--",
        )
        cols[3].metric(
            f"Backtested Sharpe ({BACKTEST_LOOKBACK_DAYS}d, raw)",
            f"{result.annualized_sharpe:+.2f}" if result is not None else "--",
            help=f"n={result.n} daily observations, assumed independent. See the adjusted figure next to it." if result else None,
        )
        cols[4].metric(
            "Autocorrelation-adjusted",
            f"{adjusted.adjusted_sharpe:+.2f}" if adjusted is not None else "--",
            delta=f"{adjusted.adjusted_sharpe - adjusted.raw_sharpe:+.2f} vs. raw" if adjusted is not None else None,
            help=(
                f"Lag-1 autocorrelation {adjusted.lag1_autocorrelation:.2f} leaves "
                f"roughly {adjusted.effective_annual_observations:.1f} independent "
                "observations per year, not 365. This is the more honest number."
            ) if adjusted is not None else None,
        )

        st.caption(
            "The raw Sharpe assumes 365 independent daily draws. This series "
            "barely moves day to day, so that assumption is badly wrong. The "
            "autocorrelation-adjusted figure corrects the annualization for the "
            "actual lag-1 autocorrelation instead of pretending it's zero. Even "
            "the adjusted number is still optimistic for three further reasons: "
            "it's a signal-timing proxy, never a simulated dollar P&L against a "
            "real repriced position; the window happened not to contain a vol "
            "spike; and n is small enough that the standard error on either "
            "Sharpe here is itself large."
        )


def _robustness_block(universe, currency: str) -> None:
    with st.expander("Robustness and distribution", expanded=True):
        name = _perpetual_name(universe, currency)
        if name is None:
            st.caption("No perpetual future listed for this currency; nothing to check.")
            return

        end = datetime.now(UTC)
        start = end - timedelta(days=BACKTEST_LOOKBACK_DAYS)
        candles = data.candles_for(name, start, end, REALIZED_CANDLE_RESOLUTION)
        dvol_series = data.dvol_for(currency, lookback_hours=BACKTEST_LOOKBACK_DAYS * 24)
        if candles is None or len(candles.candles) < 2 or not dvol_series or not dvol_series.points:
            st.caption("Not enough history to check right now.")
            return

        realized_points = rolling_realized_vol(
            candles.candles, window=timedelta(days=REALIZED_LOOKBACK_DAYS), step=timedelta(days=1),
        )
        points = variance_risk_premium(dvol_series.points, realized_points)
        if not points:
            st.caption("No aligned observations to check right now.")
            return

        st.markdown("**Distribution of the daily premium**")
        premiums = [p.premium for p in points]
        mean_premium = sum(premiums) / len(premiums)
        hist_figure = go.Figure(go.Histogram(x=premiums, nbinsx=20, marker_color="steelblue"))
        hist_figure.add_vline(x=0.0, line_color="gray", line_width=1)
        hist_figure.add_vline(x=mean_premium, line_color="orange", line_dash="dash", annotation_text="mean")
        hist_figure.update_layout(
            xaxis_title="Daily premium", xaxis_tickformat=".0%", yaxis_title="Days",
            height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        )
        st.plotly_chart(hist_figure, width="stretch")
        skew = skewness(premiums)
        st.metric(
            "Skew",
            f"{skew:+.2f}" if skew is not None else "--",
            help="A short-gamma P&L is textbook negatively skewed: small frequent gains, rare large losses. This is the premium signal, not realized P&L, so it need not (and here mostly doesn't) show that shape yet.",
        )
        st.caption(
            "This is what the Sharpe above is actually computed from. A tight, "
            "symmetric bell curve is what would make sqrt(365) a defensible "
            "annualization. A lumpy, fat-tailed one, typical of a vol series, "
            "is exactly why it isn't."
        )

        st.markdown("**Cumulative signal: timed entry vs. always in**")
        timed = cumulative_captured_premium(points)
        always_in = cumulative_captured_premium(points, threshold=float("-inf"))
        cum_figure = go.Figure()
        cum_figure.add_trace(go.Scatter(
            x=[p.timestamp for p in points], y=list(timed),
            mode="lines", line=dict(color="seagreen"), name="Timed (favorable days only)",
        ))
        cum_figure.add_trace(go.Scatter(
            x=[p.timestamp for p in points], y=list(always_in),
            mode="lines", line=dict(color="gray", dash="dot"), name="Always in (unconditional)",
        ))
        cum_figure.update_layout(
            xaxis_title="Time", yaxis_title="Cumulative premium (vol-point-days)", yaxis_tickformat=".0%",
            height=280, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(cum_figure, width="stretch")
        st.metric(
            "Max drawdown, always-in line",
            f"{max_drawdown(always_in):.2%}",
            help="Timing only ever adds, so its drawdown is trivially zero by construction, not a real finding. This is the drawdown of just holding the raw signal unconditionally, which can and does go negative; the gap between the two lines above is what the timing rule is worth.",
        )
        st.caption(
            "The axis sums a daily vol-point gap across ~90 days, so it isn't capped "
            "near 100% the way a real return would be. A large-looking number here "
            "reflects many days added together, not leverage or a loss of principal. "
            "Neither line is a P&L curve: no sizing, hedging cost, or repricing, just "
            "the raw signal accumulated two ways. The green line only adds on "
            "favorable days (flat otherwise), so it can never draw down; the gray "
            "line adds every day, including negative ones, and is what actually "
            "shows risk. A real hedged position also pays theta and gamma on "
            "unfavorable days, so its real drawdown would be worse than either line here."
        )

        st.markdown("**Does this survive a different realized-vol window?**")
        rows = []
        for window_days in SENSITIVITY_WINDOWS_DAYS:
            window_realized = rolling_realized_vol(
                candles.candles, window=timedelta(days=window_days), step=timedelta(days=1),
            )
            window_points = variance_risk_premium(dvol_series.points, window_realized)
            window_summary = summarise_premium(window_points)
            if window_summary is None:
                continue
            window_sharpe = backtest_premium_sharpe([p.premium for p in window_points])
            rows.append({
                "Realized-vol window": f"{window_days}d",
                "n": window_summary.n,
                "Days favorable": f"{window_summary.fraction_positive:.0%}",
                "Mean premium": f"{window_summary.mean_premium:+.2%}",
                "Sharpe": f"{window_sharpe.annualized_sharpe:+.2f}" if window_sharpe else "--",
            })
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            f"Same {BACKTEST_LOOKBACK_DAYS}-day window, three different realized-vol "
            "lookbacks (the panel above uses 7d). Watch the Sharpe column specifically: "
            "a longer realized-vol window smooths the series, which mechanically lowers "
            "its day-to-day variance and inflates the annualized Sharpe even when days "
            "favorable and mean premium look similar. Same sqrt(365)-on-smooth-data "
            "problem as above, just worse with a longer window. Favorable-day fraction "
            "and mean premium are the more stable numbers here; Sharpe is the least."
        )


def _sizing_block(call_greeks, put_greeks, currency: str) -> tuple[float, float]:
    vega_per_straddle = abs(call_greeks.vega) + abs(put_greeks.vega)
    vega_budget = st.number_input(
        f"Vega risk budget ({currency}, coin)", min_value=0.0001, value=1.0, step=0.1, key="position_vega_budget"
    )
    try:
        size = size_for_vega_budget(vega_per_straddle, vega_budget)
    except StrategyError:
        st.caption("This strike has no vega left to size against (too deep or too close to expiry).")
        return 0.0, vega_budget
    st.metric("Straddles (short)", f"{size:g}", help=f"vega/straddle {vega_per_straddle:.6f} coin")
    return size, vega_budget


def _pnl_path_block(universe, currency: str, model, strike: float, expiry: datetime, rate: float, size: float) -> None:
    with st.expander("Historical P&L: if you'd held this exact position", expanded=True):
        name = _perpetual_name(universe, currency)
        if name is None:
            st.caption("No perpetual future listed for this currency; nothing to walk.")
            return

        end = datetime.now(UTC)
        start = end - timedelta(days=BACKTEST_LOOKBACK_DAYS)
        candles = data.candles_for(name, start, end, REALIZED_CANDLE_RESOLUTION)
        dvol_series = data.dvol_for(currency, lookback_hours=BACKTEST_LOOKBACK_DAYS * 24)
        if candles is None or len(candles.candles) < 2 or not dvol_series or not dvol_series.points:
            st.caption("Not enough history to walk right now.")
            return

        year_seconds = 365.0 * 24 * 3600
        candle_sorted = sorted(candles.candles, key=lambda c: c.timestamp)
        dvol_sorted = sorted(dvol_series.points, key=lambda p: p.timestamp)

        steps: list[SessionStep] = []
        t = candle_sorted[0].timestamp
        while t <= candle_sorted[-1].timestamp:
            tau = (expiry - t).total_seconds() / year_seconds
            if tau <= 0.0:
                break  # this expiry had already happened at this point in the window
            nearest_candle = min(candle_sorted, key=lambda c: abs((c.timestamp - t).total_seconds()))
            nearest_dvol = min(dvol_sorted, key=lambda p: abs((p.timestamp - t).total_seconds()))
            forward_t = nearest_candle.close * math.exp(rate * tau)
            steps.append(SessionStep(timestamp=t, underlying=forward_t, tau=tau, vol=nearest_dvol.close))
            t += timedelta(days=1)

        try:
            path = simulate_short_straddle_path(model, steps, strike=strike, rate=rate, size=size)
        except ArbitrageError:
            st.caption("Couldn't walk the full window: this strike hit an arbitrage bound somewhere in it (typically very close to expiry).")
            return
        if not path:
            st.caption("Not enough history in this expiry's window to walk yet.")
            return

        figure = go.Figure(go.Scatter(
            x=[p.timestamp for p in path], y=[p.cumulative_pnl for p in path],
            mode="lines", line=dict(color="steelblue"), fill="tozeroy",
        ))
        figure.add_hline(y=0.0, line_width=1, line_color="gray")
        figure.update_layout(
            xaxis_title="Time", yaxis_title=f"Cumulative P&L ({currency}, coin)",
            height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        )
        st.plotly_chart(figure, width="stretch")

        final = path[-1].cumulative_pnl
        drawdown = max_drawdown([p.cumulative_pnl for p in path])
        cols = st.columns(2)
        cols[0].metric(f"P&L over the window ({currency}, coin)", f"{final:+.4f}")
        cols[1].metric(f"Max drawdown ({currency}, coin)", f"{drawdown:.4f}")
        st.caption(
            f"Reprices the exact {size:g}-straddle position ({strike:g} strike) at every "
            "day in the window and rehedges to flat delta daily, the same construction "
            "the math above describes. Not a real historical backtest: vol at each step "
            "is DVOL, a single index-wide number, not this strike's own historical smile, "
            "so the further this strike sits from the money at a given point, the less "
            "that number actually applies to it."
        )


def _risk_block(combined, hedge: float, underlying: float, realized_vol: float | None, currency: str) -> None:
    with st.expander("Combined risk (options + hedge)", expanded=True):
        rows = [{"Greek": f, f"Value ({currency}, coin)": round(getattr(combined, f), 8)} for f in
                ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            f"Every Greek here is coin-denominated (this position settles in "
            f"{currency}, not cash). Hedge: {hedge:+.6f} units of the "
            "underlying to bring delta to zero."
        )

        if realized_vol is not None:
            edge = expected_daily_edge(combined.gamma, combined.theta, underlying, realized_vol)
            cols = st.columns(3)
            cols[0].metric(f"Expected gamma P&L (1d, {currency})", f"{edge.gamma_pnl:+.6f}")
            cols[1].metric(f"Expected theta P&L (1d, {currency})", f"{edge.theta_pnl:+.6f}")
            cols[2].metric(f"Expected edge (1d, {currency})", f"{edge.total:+.6f}")


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
    call_price, put_price, call_greeks, put_greeks, currency: str,
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
            st.caption("Not entered yet. Everything above is a live preview.")
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
        cols[0].metric(f"Option P&L ({currency}, coin)", f"{option_pnl:+.6f}")
        cols[1].metric(f"Hedge P&L ({currency}, coin)", f"{hedge_pnl:+.6f}")
        cols[2].metric(f"Total P&L ({currency}, coin)", f"{total_pnl:+.6f}")
