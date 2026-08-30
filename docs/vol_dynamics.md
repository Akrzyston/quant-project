# Volatility dynamics: implied vs realized, the variance premium, and sticky regime

## In one sentence

Deribit's own DVOL and realized-vol figures are opaque numbers that this
project reconciles against independently-computed ones rather than trusting,
their spread over time is the variance risk premium, and whether the smile
moves with the strike or with the forward is measured from one day of
intraday option candles rather than assumed.

## Realized volatility: two estimators, and why they won't match Deribit's

`voltk.realized_vol` computes two independent numbers from raw perpetual
candles (`get_tradingview_chart_data`):

- **close-to-close**: sample standard deviation of log returns between
  consecutive candle closes, scaled by `sqrt(periods_per_year)`.
- **Parkinson (range)**: uses each candle's own high/low, not just its
  close, so it captures intra-candle movement close-to-close discards
  entirely. About 5x more statistically efficient than close-to-close for
  the same sample size (Parkinson, 1980), at the cost of assuming pure
  diffusion (no jumps, no drift) -- an assumption close-to-close doesn't
  need.

`periods_per_year` is inferred from the **actual median spacing between
candle timestamps**, not the requested resolution -- the venue can return a
coarser or gappier grid than asked (thin trading, an outage), and trusting
the request parameter over the data would silently mis-annualize the
result.

Deribit's `get_historical_volatility` is a black box: the venue does not
document its sampling frequency, window length, or annualisation
convention. Reconciled live (see `app/panels/vol_history.py`), the two
own-computed estimators and Deribit's own figure sit within roughly 100-200
bps of each other on a two-week BTC window -- close, but never exact, and
that gap is the expected signature of a different estimator on a different
window, not a bug in either number. A candidate who reports an exact match
either got lucky or is comparing numbers built the same way by coincidence.

## The variance risk premium

`voltk.variance_premium.variance_risk_premium` pairs each DVOL close to the
nearest realized-vol observation within a stated gap tolerance (default one
hour) -- the two series come from independently-sampled endpoints on
different timestamp grids, so a naive zip of two same-length lists would
silently misalign them. A DVOL point with nothing close enough in time is
dropped, not matched to a stale observation or interpolated across the gap:
a real data outage should show up as a shorter series, not a smoothed-over
one.

`premium = implied - realized` at each aligned timestamp. `summarise_premium`
reports the mean, the fraction of the window it ran positive, and the exact
timestamps it inverted (`premium < 0`). A persistently positive premium is
the textbook variance risk premium: sellers of volatility are compensated
on average for bearing realized-vol risk, because implied systematically
overstates what actually realizes. Inversions are not an error condition --
they are the periods that compensation ran the other way, and a real
trading decision would need to know exactly when.

## Sticky-strike vs sticky-delta, measured

`voltk.surface.Surface` already computes two THEORETICAL predictions for
`d(vol)/d(spot)` at a fixed strike: `dvol_sticky_strike` (identically zero,
by definition of the regime -- the smile is pinned to absolute strikes) and
`dvol_dspot_sticky_delta` (nonzero -- the fitted curve is held fixed in
relative log-moneyness, so the smile "follows the underlying"). Which one
actually describes the market is an empirical question M4 leaves open for
M6 to answer.

### Inverting a historical option candle needs the forward at that timestamp

The candle is coin-denominated (confirmed against the live endpoint,
matching every other Deribit option quote this project handles), and tau
genuinely shrinks through a session -- neither is true of a single
snapshot. The forward at each timestamp comes from the same put-call parity
identity `voltk.forward.implied_forward_curve` already uses for a full
chain snapshot (`voltk.forward.forward_from_parity`, extracted so both call
sites share one formula), applied per timestamp to a call/put candle pair
rather than once to a book summary:

```
Coin-settled: C - P = 1 - K/F  =>  F_K = K / (1 - (C-P))
```

Reading the forward off the perpetual directly, the way Deribit's own chain
display reads spot, would reintroduce the exact basis trap M2 exists to
avoid -- so the perpetual is used only as the regression's spot leg, never
as the forward for inversion.

`voltk.sticky_regime.observations_from_candles` aligns call, put and
perpetual candles by **exact timestamp match**, not position: Deribit does
not guarantee identical tick grids across three separate instruments when
one trades thinly, and a missing tick in one series must drop that
timestamp everywhere rather than silently misaligning the rest. Points the
solver can't identify (vega below floor -- deep in/out-of-the-money, no
information left about vol) are dropped before they can add noise to the
regression.

### The regression is on changes, not levels

`voltk.sticky_regime.regress_vol_on_spot` runs OLS on **first differences**
(`d(vol)` against `d(spot)` between consecutive observations), not on the
raw levels. A level regression would pick up whatever shared trend both
series have over the session -- vol and spot can easily share a trend for
reasons that have nothing to do with the sticky-strike question -- rather
than the local sensitivity `d(vol)/d(spot)` that
`Delta_eff = Delta_BS + Vega * d(sigma)/dS` actually needs.

The empirical slope, `sticky_strike_prediction` (always 0.0) and
`sticky_delta_prediction` (`Surface.dvol_dspot_sticky_delta` at that
strike, forward and tau, from a surface fit in the same session) are
reported together. The function does not itself declare a winner: which
theoretical prediction the empirical slope sits closer to is the actual
answer, and it can legitimately differ by strike, by expiry, or by how
volatile the session was -- a single day is one draw, not a proof.

## A note on calendar time

Crypto trades continuously -- there is no overnight gap, no weekend, no
exchange holiday. Calendar time is the right time measure for every tau in
this library (`(expiry - as_of) / 365 days`, `ACT/365` throughout), unlike
equities, where calendar time and trading time diverge and a model that
ignores the difference misprices weekend theta. This project never needed a
trading-time correction for that reason, not because the distinction
doesn't exist.

## What is checked

| Property | Test |
| --- | --- |
| DVOL wire values are a percentage, not a decimal fraction | `tests/test_dvol_parse.py::test_dvol_series_wire_values_are_a_percentage_not_a_decimal_fraction` |
| Realized-vol wire values are a percentage | `tests/test_vol_history_parse.py::test_historical_volatility_series_converts_percentage_to_decimal` |
| Annualisation factor comes from actual candle spacing, not the request | `tests/test_realized_vol.py::test_infer_periods_per_year_ignores_the_requested_resolution_and_uses_actual_gaps` |
| Close-to-close recovers a known constant-step vol exactly | `tests/test_realized_vol.py::test_close_to_close_recovers_a_known_constant_step_vol` |
| VRP alignment matches nearest-in-time, drops points beyond the gap tolerance | `tests/test_variance_premium.py::test_variance_risk_premium_drops_points_beyond_the_max_gap` |
| Candle inversion recovers the forward and vol used to construct prices | `tests/test_sticky_regime.py::test_observations_from_candles_recovers_the_forward_and_vol_used_to_construct_prices` |
| A timestamp missing from any one candle series is dropped everywhere | `tests/test_sticky_regime.py::test_observations_from_candles_drops_timestamps_missing_from_any_series` |
| Regression recovers an exact known slope on synthetic data | `tests/test_sticky_regime.py::test_regress_vol_on_spot_recovers_an_exact_known_slope` |
