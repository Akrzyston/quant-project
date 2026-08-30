# voltk — volatility toolkit + dashboard

A reusable pricing/risk library with a Streamlit view on the outside. The
dashboard accumulates one panel per milestone.

```
uv run pytest -q                        # 898 cases
uv run streamlit run streamlit_app.py
make all                                # regenerate every report figure
```

## The boundary

`src/voltk/` holds instruments, market data, capture, and pricing. It imports no
Streamlit, Plotly, or `app`. Two tests enforce this: a static AST scan of every
module, and a runtime import with the view packages poisoned in `sys.meta_path`.

`app/` renders. No pricing, fitting, or risk logic. `scripts/figures/` does the
same for the static report figures, one file per figure, none of them pricing
or fitting anything themselves either.

## Report figures

`make all` runs every script in `scripts/figures/` and writes PNGs to
`reports/figures/` (gitignored — regenerated, not committed). Every figure
loads off a stored snapshot (`scripts/figures/common.py`), capturing one
first if `data/snapshots.db` doesn't have one yet, so the same run works on
a fresh checkout and reproduces identically on repeated runs against the
same snapshot. Two figures — implied vs realized vol, and the simulated
market-making session — are necessarily live: DVOL, realized-vol history,
and intraday candles have no snapshot/replay concept anywhere else in this
project either, so those two scripts pull live data the same way the
dashboard panels that need the same history already do.

## M0 — instrument dossier and snapshot pipeline

### Synchronous capture

A chain read at a different moment from its underlying gives a wrong surface and
nothing downstream can detect it. REST cannot deliver an instant, so
`voltk.capture.capture` brackets instead:

1. Instrument definitions for every currency and kind (slowest-moving, first).
2. Index price per currency, opening the bracket.
3. Book summaries for options and futures (fastest-moving).
4. Index price per currency again, closing the bracket.

Elapsed window and index drift across the bracket are recorded. Either breaching
its limit marks the capture degraded with the reason attached; nothing is
silently discarded. `as_of` is the bracket midpoint, so tau is centred on the
observation rather than biased to whichever end the clock was read at, and index
prices are the mean of the two readings.

Defaults are 6000 ms and 15 bps, both arguments.

### Snapshots

SQLite at `data/snapshots.db`. Two tables: `snapshots` for metadata, `payloads`
for raw bodies.

Response bodies are stored verbatim and SHA-256'd. Nothing derived is persisted.
Reload re-parses those bytes through the same functions the live path uses, and
a hash mismatch on load raises rather than returning suspect data.

Replay determinism rests on one rule: `as_of` comes from the snapshot, never from
the clock. `Instrument.tau()` takes a required `as_of` argument, so there is no
`datetime.now()` default to leak. `ReplaySource` returns the index bracket
readings in capture order, so recorded drift survives a reload.

`tests/test_replay_determinism.py` proves it by reloading the same snapshot four
times with sleeps between and asserting a single dossier digest.

`ReplaySource` satisfies the same `MarketDataSource` protocol as the live client,
so `capture()` runs unchanged against a snapshot.

### Contract specification

Per-instrument facts — settlement currency, contract size, tick size, minimum
trade amount, commissions, inverse or linear — are read from venue metadata. A
missing field raises `MetadataError` rather than defaulting.

Facts the API does not express live in `src/voltk/specs/deribit.json` with a
source URL and retrieval date: the 08:00 UTC expiry convention, the settlement
averaging window, and the premium cap on option fees.

The expiry convention is checked against live metadata rather than trusted.
`validate_expiry_convention` raises if any instrument expires off-convention.

### Universe

The universe is a set of currencies, not one. `UniverseSpec` carries currencies
and kinds; capture reads them all in a single bracket. This is what makes
cross-currency work possible later: a relative-value trade needs both chains
observed at the same instant, and that cannot be reconstructed by stitching two
separate captures together. Selecting a second currency in the dashboard costs
nothing structurally.

The forward curve is built from dated futures marks per currency. Option expiries
rarely coincide with future expiries, so `forward_for` interpolates linearly in
time between bracketing futures and extrapolates flat outside. With no dated
futures it falls back to the index.

### No hardcoded identifiers

`tests/test_no_hardcoded_identifiers.py` scans every string constant in the
library for currency tickers, instrument names, index names, and perpetual names.
Docstrings are exempt; anything assignable or comparable is not.

Index names come from the `price_index` field on instrument metadata. Currencies
come from discovery. The spec file describes the venue rather than any asset, and
is scanned too.

## Slots

| Slot | Region | Rendering |
| --- | --- | --- |
| `LEFT_RAIL` | top left | stacked cards |
| `MAIN` | top right | tabs once more than one panel |
| `QUOTER_RAIL` | bottom left | stacked cards |
| `QUOTER_MAIN` | bottom right | tabs once more than one panel |

Add a panel with one file under `app/panels/`:

```python
@panel(key="smile", title="Smile", slot=Slot.MAIN, order=30, milestone="M3")
def render() -> None:
    ...
```

Then add `("smile", "main", 30)` to `EXPECTED` in `tests/test_registry.py`. That
one-line diff is what makes a panel that stops registering fail CI rather than
vanish from the screen.

## M1 — pricing engine

Five models: Black-Scholes (spot), Black-76 (forward), Bachelier (normal vol),
a CRR binomial tree with early exercise and discrete dividends, and the inverse
adjustment.

Every model exposes `price`, `implied_vol`, `greeks`, `bounds`, and
`vol_bracket`. Registering a model in `voltk/models/__init__.py` is what makes it
selectable in the UI; no model name is written in the view layer.

### Validation

- Round trip `price -> implied_vol -> price` under 1e-10 across every model,
  strike, expiry, vol, and right.
- Put-call parity per model. The inverse relation is `C - P = 1 - K/F`, not
  `D(F - K)`, and is checked separately.
- No-arbitrage bounds on every price; a quote outside them raises
  `ArbitrageError` rather than returning a fitted number.
- Analytic greeks against Richardson-extrapolated finite differences, with a
  per-greek tolerance and a conditioning-derived floor. See the module docstring
  in `tests/test_greeks_finite_difference.py` for why a fixed tolerance is not
  enough.
- Put-call parity against observed chain marks in `voltk/validation.py`, which
  reports gaps against the combined spread rather than asserting equality. The
  coin-settled relation is `C - P = 1 - K/F`.
- Binomial convergence to Black-Scholes at first order, zero early-exercise
  premium on a dividend-free American call, positive premium once dividends are
  added.

`implied_vol_detailed` reports whether the answer is identified: where vega falls
below a floor the price carries almost no information about vol, and a solver
will happily return a number that means nothing. The round trip still closes on
price; the vol does not.

### The inverse adjustment

Derived two independent ways — a replicating portfolio via change of numeraire,
and direct conversion of the quote-currency value — which agree to about 1e-17.
Full argument in `docs/inverse_replication.md`, including the quanto comparison
and the delta-versus-strike shape.

## M2 — forward curve, chain forensics

Deribit's own chain display converts USD bid/ask off the index price, not the
forward. Read those numbers at face value and every strike is quietly skewed
by whatever the basis happens to be that day. `voltk/forward.py` catches this
by deriving a forward independently: for every expiry, invert put-call parity
at each two-sided strike — coin-settled `F_K = K / (1 - (C-P))`, quote-settled
`F_K = K + (C-P)/D` — and take the median across strikes rather than fitting a
regression. One bad quote does not move the answer, and there is no extra
machinery to validate. The Forward Curve panel plots this implied forward
against the traded future and the index; a chain priced off the index instead
of the forward recovers the index almost exactly, with a basis running to
hundreds of bps against the future, which is the trap's signature.

`voltk/chain.py` runs a hygiene funnel over the raw chain before anything
downstream sees it: missing quote, non-positive mark, crossed market, in that
order. Each instrument is attributed to the first rule that drops it, and
every rule's dropped fraction is reported against the original chain size, not
the shrinking survivor pool, so the rules read independently. The Chain View
panel shows the funnel and every dropped instrument name — nothing is
discarded without being counted and shown. This is deliberately scoped to raw
data hygiene; parity and no-arbitrage bound checks are a model-dependent kind
of judgment and stay in `voltk/validation.py`, with their own display in the
dossier's parity expander.

`Universe.forward_for` extrapolates flat outside the bracketing futures rather
than, say, linearly extending the last segment's slope — flat is the least
surprising choice when there is no information past the last traded future,
and is confirmed here as the M2 decision rather than revisited.

## M3 — surface construction, DVOL cross-check

Fit in total variance against log-moneyness, not raw implied vol against
strike. `w = vol^2 * tau` against `k = ln(K/F)` is the quantity that is linear
under the time interpolation this milestone does, and whose monotonicity in
maturity *is* the calendar no-arbitrage condition — neither statement holds
for vol against strike, and strikes are not comparable across expiries the
way moneyness is.

**The structural prior is raw SVI** (Gatheral): `w(k) = a + b{ρ(k-m) + √((k-m)²+σ²)}`.
Three reasons, stated and defended rather than assumed:

- It is linear in `(a, bρ, b)` for a fixed `(m, σ)`, so most of the fit is
  well-conditioned; `voltk/surface.py` calibrates the full nonlinear problem
  with `scipy.optimize.least_squares`, bounded to stay a valid slice
  (`b≥0, |ρ|<1, σ>0`), with the joint non-negative-variance constraint
  `a + bσ√(1-ρ²) ≥ 0` checked post-fit across a few re-seeded attempts.
- Its wings grow **linearly in total variance** by construction, so implied
  vol grows like `√|k|` rather than exploding — this is what keeps deep-wing
  extrapolation from producing the "400% vol at the tails" failure mode
  without any ad hoc capping.
- Gatheral's own `g(k)` function gives an exact, checkable butterfly test —
  `butterfly_check` evaluates it analytically (SVI's own derivatives, no
  finite differences) on a grid padded 20% past the fitted strikes, so a
  problem just outside the observed range doesn't go unchecked.

`Surface` stitches independently-calibrated slices into one continuous
strike/time object: total variance interpolates linearly in `tau` between the
two bracketing expiries at fixed `k` (algebraically the same thing as CBOE's
constant-maturity formula), and extrapolates outside the fitted maturity
range by holding the instantaneous variance rate constant at the nearest
slice — the same flat-rate idea SVI's wings already apply in the strike
direction, applied once more in time, so neither axis of extrapolation can
blow up. `calendar_check` verifies non-decreasing total variance in maturity
between every adjacent calibrated pair. `Surface.dvol_dforward_sticky_delta`
gives the model's own sticky-delta vol sensitivity to a forward move —
corrected in M4 from an earlier mislabeling as sticky-strike; genuine
sticky-strike is identically zero by definition (see the M4 section below).

**Model-free variance cross-check.** `voltk/variance.py` replicates CBOE's
variance-swap-style discretized sum over the OTM strike ladder, using the
forward derived from parity (`voltk/forward.py`, M2) rather than the traded
future — the same convention CBOE's own methodology uses, and the reason M2
and M3 are the same project. `constant_maturity_variance` interpolates this
to a target tenor (30 days) the same way `Surface` does. The Smile panel
reports this model-free index against both Deribit's published DVOL and the
fitted surface's own ATM level side by side: a surface can be right
pointwise and wrong in aggregate, and this is what catches that.

`smile_points` and `model_free_variance` both convert a coin-denominated mark
to quote-currency terms before using it — corrected in M5 from a real bug
where a raw coin-scale mark was fed unconverted into the quote-currency
Black76 reference/CBOE sum; see the M5 section below for the fix and why it
went undetected until then.

`dvol_series`'s own parsing had a second, independent bug of the same shape:
DVOL closes arrive on the wire as a percentage number, not the decimal
fraction the parser assumed and a regression test asserted without ever
checking a live response — corrected in M6, see that section for the fix.

**Numerics**: `numpy`/`scipy` are new dependencies as of this milestone, the
first ones added to `src/voltk/`. Every earlier module (`solver.py`'s
Newton/bisection, `forward.py`'s median-of-strikes) is hand-rolled on `math`;
SVI calibration is exactly the class of problem — real nonlinear least
squares, array/grid construction for the 3D surface — where a battle-tested
library is the professional choice, not a shortcut, and it's a scoped
exception rather than a retroactive rewrite of what came before.

## M4 — greeks in both units, bucketed vega, shock ladder

### A correction to M3, found while building this milestone

`Surface.dvol_dspot` was mislabeled. It bumped the forward and held the
fitted SVI curve fixed in relative log-moneyness `k=ln(K/F)` — the smile
"follows the underlying," which is the textbook definition of **sticky-delta**
(confirmed against an independent options-theory source), not sticky-strike
as the old docstring claimed. Genuine sticky-strike means the smile is pinned
to absolute strikes, so vol at a fixed strike does not move at all — it is
identically zero, which has a clean financial reading used below. Split into
`Surface.dvol_dforward_sticky_delta`, `Surface.dvol_dspot_sticky_delta`
(chain-ruled through `F=S·exp(rate·τ)`), and `Surface.dvol_sticky_strike`
(always `0.0`).

### Coin and cash are not a units conversion

`voltk/greeks.cash_greeks_from_coin` derives cash Greeks from
`InverseOption`'s coin Greeks via the self-quanto chain rule:
`V_cash(S) = V_coin(F(S))·S`, `F(S)=S·exp(rate·τ)`. Differentiating:

```
Δ_cash = V_coin + F·Δ_coin
Γ_cash = exp(rate·τ)·(F·Γ_coin + 2·Δ_coin)
Vega_cash = S·Vega_coin   (and theta, rho, vanna, volga the same way)
```

`InverseOption`'s Greeks are *forward* Greeks (`Δ_coin = ∂V_coin/∂F`), and
cash value is a genuinely *spot* quantity — delta comes out right even from a
naive `F`-for-`S` substitution (a coincidence of the algebra), but gamma is
off by a real, material amount (~4% at a realistic rate) without the
`exp(rate·τ)` factor, verified against a direct finite difference of
`V_cash(S)`. Vega/theta/rho/vanna/volga stay simple spot multiples, since a
vol/tau/rate bump holds both `F` and `S` fixed — no product-rule term. Tests
regression-lock the trap: cash delta and gamma are asserted to *differ* from
the naive forms, not just to match the correct ones.

### Vanna and volga

Closed forms for Black-Scholes/Black-76 (`-Vega·d2/(underlying·vol·√τ)`,
`Vega·d1·d2/vol`), Bachelier (same shape with its own unsigned `d`), and
`InverseOption` (derived from the same `F·pdf(d1)=K·pdf(d2)` cancellation
vega/theta already use, cp-independent like they are) — all four verified
against Richardson-extrapolated finite differences of the analytic vega
itself, both call and put, to 9+ significant figures. `Binomial` reprices on
rebuilt trees, the same central-difference approach it already uses for
vega/rho.

### Skew-adjusted delta, both regimes

`Δ_eff = Δ_flat + Vega × ∂σ/∂(underlying)`. Under sticky-strike this is
exactly `Δ_flat` — the fitted smile doesn't move as spot moves, by
definition, so the flat delta is already right. Under sticky-delta the smile
follows the underlying, so the adjustment is real. M6 later determines
empirically which regime actually holds against intraday data; M4 only
computes both, routed to the model's own spot- or forward-flavoured
sensitivity so the adjustment stays dimensionally consistent with that
model's delta.

### Bucketed vega by surface control point

SVI has no literal spline knots — five global parameters, not a piecewise
curve — so a "control point" is one of the market smile points the expiry
was actually calibrated against. Buckets are formed by rank in log-moneyness,
never a hardcoded strike or delta cutoff. Each bucket's vega comes from
bumping only that bucket's points' total variance by a fixed additive amount,
refitting SVI, and repricing. The reconciliation (bucketed vegas should sum
to the parallel vega) rests on an exact identity: `w=a+b(...)` is linear in
`a`, so a uniform bump across *every* point is absorbed by `a` alone with
`b/ρ/m/σ` unchanged (verified to 1e-9) — what makes the sum meaningful when
only a subset is bumped, rather than a loose first-order claim.

### Structured shock ladder

Three named, reproducible transformations of the *current* fitted SVI slice
— level bumps `a`, skew bumps `ρ`, curvature bumps `σ` — each provably exact
at the slice's own vertex `k=m`: `w(m)=a+bσ`, `∂w/∂k(m)=bρ`,
`∂²w/∂k²(m)=b/σ`. A level shock leaves both derivatives untouched; a skew
shock leaves the level and curvature untouched, changing only the slope; a
curvature shock leaves the slope untouched, changing only the curvature.
Genuinely not a parallel vol shift — none of the three, nor any pair, moves
vol by a proportional amount across strikes.

**Honest scope**: this is not a true historical level/skew/curvature PCA.
Deribit's public API has no bulk historical-chain endpoint
(`get_book_summary_by_currency`/`get_instruments` are current-state only),
DVOL history is a single scalar time series with no per-strike information,
and the local snapshot store has no automated SVI-history harvesting yet — so
a real empirical decomposition isn't buildable right now. One partial, real
use of history: the level shock's *magnitude* (not its shape) is sized from
the realized vol-of-vol of Deribit's own published DVOL closes when
available, falling back to a documented default otherwise.

### Portfolio aggregation

Deliberately minimal and session-local — `voltk/portfolio.py` sums cash
Greeks across a handful of user-picked positions (cash aggregation is always
valid once each position's own cash Greek is correctly derived) and refuses
native-unit aggregation across mixed settlement currencies or conventions
rather than silently summing incompatible units. Not persisted: real
position tracking is M8's job.

## M5 — snapshot browser, A-vs-B compare, P&L attribution

### A real bug in M3, found while smoke-testing this milestone

Every M3/M4 test built its synthetic chain by pricing through `Black76`
(quote-currency scale). Deribit actually quotes options in the settlement
(coin) currency — confirmed against Deribit's own docs and support articles,
"Bitcoin options are priced in Bitcoin" — and `smile_points`/
`model_free_variance` fed those coin-scale marks straight into a quote-scale
Black76 reference/CBOE sum with no conversion. Nothing crashed (unsolvable
points were quietly dropped, or the variance number was just wrong), and
every existing test happened to already be quote-scale, so nothing caught it
until this milestone's first synthetic fixture priced through `InverseOption`
instead. `tests/synthetic_chain.py` now prices via `InverseOption` (matching
what a real chain actually looks like), and both functions convert
`price_coin -> price_coin * forward` before use — not a new assumption, but
`InverseOption`'s own internal identity (`price_coin * forward` equals
`Black76(forward, ..., rate=0, ...)` exactly, "the discount factor cancels
against the forward") and Deribit's own stated future-value quoting
convention. `model_free_variance` needed a second, related fix: that same
conversion already performs the CBOE formula's own `e^{rT}` undiscounting
(`Black76(rate=0) = Black76(rate=R)*e^{R*tau}` for any `R`, another exact
identity), so applying `e^{rT}` again on top double-counts it — caught only
because `tests/synthetic_chain.py` gained an optional nonzero `rate`
parameter for this milestone; every prior fixture had rate pinned to exactly
zero by construction, which can't tell "no extra discount" from "extra
discount of `e^0=1`."

### Schema and round-trip (Core)

No schema change: the snapshot store already persists only raw response
bytes (`src/voltk/snapshots.py`), matching the M0 design note that a snapshot
should be re-parseable "by code that did not exist until M5." The brief's
"fit parameters" deliverable means re-deriving them at load time via M3's
`calibrate_surface`, not persisting a new blob — storing fitted vols instead
of the raw chain is a named failure mode, not a shortcut to take here.
Bit-for-bit reproduction composes two already-separately-proven halves:
reload produces an identical universe (already covered by
`tests/test_snapshots.py`/`tests/test_replay_determinism.py`), and
`calibrate_surface` is a pure function of its inputs — no RNG, no threading
in its seed schedule — so calling it twice on identical inputs now has a
dedicated test proving exact `SVISlice` equality.

### Snapshot browser and A-vs-B compare

`app/data.py` gained `universe_for_snapshot`/`marks_for_snapshot`, explicit-
snapshot-id variants of `active_universe`/`marks_for` — those two are coupled
to the session's single current replay selection and can't address two
arbitrary, explicitly-chosen snapshots at once. The new panel fits a
`Surface` at each of two chosen snapshots and diffs them on a **unioned**
k/tau grid (both surfaces' fitted ranges combined before sampling) — sampling
each on its own range and only then comparing would silently compare a
fitted point in one against an extrapolated point in the other as if they
were equally trustworthy. `voltk/surface_diff.py`'s `compare_surfaces`
answers "what moved" in ATM vol per expiry rather than as a raw SVI-parameter
diff — SVI is not identifiable, so two calibrations can differ wildly in
`(a,b,rho,m,sigma)` while implying nearly the same smile; the vol the surface
actually implies is the only safe comparison unit.

### P&L attribution

`voltk/pnl.py`'s `attribute_pnl` implements the identity in
`docs/pnl_attribution.md`, which is worth reading in full for the theta sign
convention (verified against the FD test harness's own sign flip) and a
second, subtler coin/cash unit trap distinct from M4's: `cash_greeks_from_coin`'s
delta and gamma are **spot** derivatives, but `InverseOption`'s own Greeks
are **forward** derivatives — so the attribution's `dS` must be the spot
difference between the two snapshots for a coin-settled position, not the
forward difference the model's native Greeks would suggest. Residual is
always its own field, shown in the panel, never subtracted away.

## M6 — implied vs realized, variance premium, sticky regime

### A second, independent unit bug — DVOL, found the same way M5's was

Confirmed directly against the live endpoint (BTC and ETH both): DVOL closes
arrive on the wire as a percentage number (`37.95` meaning `37.95%`), not the
decimal fraction `dvol_series`'s own docstring claimed and a regression test
asserted without ever checking a real response. Every synthetic test
elsewhere in the suite constructs `DvolPoint` directly with already-decimal
values, so none of them exercised the actual parsing path. This silently
broke every absolute-level DVOL comparison on live data since M3 —
`compare_to_dvol` on the Smile panel diffed a model-free vol of `~0.35`
against a DVOL "close" of `~38`. `shocks.py`'s M4 level-shock sizing was
unaffected: it uses log-returns of DVOL closes, scale-invariant to a constant
factor, so the bug never reached a shipped number there. Fixed by dividing by
100 in `dvol_series`, the same convention `historical_volatility_series`
already applies to its own percentage-scale endpoint.

### Implied vs realized (Core)

Two new Deribit endpoints, both live-only and best-effort like `dvol_for`
(`app/data.py`: `historical_vol_for`, `candles_for`): `get_historical_volatility`
(Deribit's own realized figure, whole trailing history, no window control) and
`get_tradingview_chart_data` (OHLCV candles for any instrument, spot or
option — confirmed live that option candles are coin-denominated, matching
every other Deribit option quote this project handles). `voltk/realized_vol.py`
computes an independent number from raw perpetual candles — close-to-close
and Parkinson range estimators, annualised from the candles' own actual
median timestamp spacing rather than the requested resolution, so a coarser
or gappier response than asked doesn't silently mis-annualize. The two never
match Deribit's published figure exactly (~100-200bps apart on a live BTC
run) — expected, since Deribit's own window/sampling/estimator choice is
undisclosed, not a discrepancy either number needs to explain away.

### Variance risk premium and sticky regime (Extended)

`voltk/variance_premium.py` pairs DVOL closes to the nearest realized-vol
observation within a stated gap tolerance (the two are independently sampled
on different grids) and reports the mean premium, the fraction of the window
it ran positive, and the exact inversion timestamps.

`voltk/sticky_regime.py` answers the question M4's own notes leave open:
does the market behave like sticky-strike or sticky-delta? One day of
intraday call/put candles across a strike ladder plus the perpetual are
aligned by exact timestamp, each pair inverted to implied vol using the
forward *at that timestamp* (put-call parity, `voltk/forward.py`'s
`forward_from_parity`, extracted so the per-snapshot and per-timestamp call
sites share one formula — reading the perpetual directly for this would
reintroduce the exact basis trap M2 exists to avoid). The vol change at each
fixed strike is regressed on the spot change (first differences, not levels,
so a shared session trend in both series doesn't get mistaken for the local
sensitivity `Delta_eff`'s formula needs) and reported alongside both
theoretical predictions `Surface` already computes:
`dvol_sticky_strike` (0, by definition) and a same-session
`dvol_dspot_sticky_delta`. The function reports the comparison; it does not
declare a winner — full derivation in `docs/vol_dynamics.md`.

### Crypto trades continuously

Every `tau` in this library is already `(expiry - as_of) / 365 days`,
calendar time throughout — this milestone is the one that explicitly calls
out *why* that's correct rather than an oversight: crypto has no weekend, no
exchange holiday, nothing for a trading-time correction to correct for.

## M7 — live quoting simulator

Mock quotes only, nothing ever submitted. `voltk/quoting.py` derives a
two-sided width from three named terms — fit residual against the surface,
gamma rehedge cost over a stated requoting interval, and a fraction of the
live market's own spread — rather than a hand-set number, and reports each
term so the panel can show why the quote is as wide as it is. Inventory
shades the quote via the standard Avellaneda-Stoikov reservation price, not
a bespoke formula.

`voltk/market_making.py` simulates a session against a real intraday
perpetual path: a fill happens whenever the next step's theo would have
moved through our quote, and each fill's markout is reattributed through
M5's own `attribute_pnl` — edge captured, adverse selection (the directional
move against the resulting position), vega P&L, and a residual that's
reported, never absorbed. The fill rule is deliberately a worst case
(maximally-informed counterparty), documented as such in
`docs/market_making.md` rather than presented as an average session.

The Mock Orderbook panel only renders against a coin-settled model — the
live market it compares against is itself coin-denominated, and pricing a
Deribit option through a quote-settled reference would need a conversion
this milestone didn't need to build. Selecting a quote-settled model shows
a caption explaining why instead of a wrong number.

## M8 — strategy proposal, position, and P&L

Full memo in `docs/strategy_proposal.md`, which names the losing scenario
before anything else per the milestone's own accept criteria. One tradeable
relationship: a delta-hedged short straddle sized off M6's own measured VRP,
using `voltk/strategy.py`. Hedging and expected-edge sizing both reuse
constructions this project already built rather than inventing new ones —
`hedge_delta`/`with_hedge` zero out the combined position's delta the same
way M7's reservation price already treats inventory, and
`expected_daily_edge`'s representative move is the identical sizing
`voltk.quoting.derive_width` uses for gamma risk, applied to expected P&L
instead of a spread.

Capacity is a stated participation rate of real open interest (`open_interest`
newly parsed onto `Quote`, alongside the bid/ask/mark M2 already extracted).
Kill conditions are three independent, always-evaluated checks, not a single
short-circuited one, so a caller sees every breach at once.

The Position panel also computes a backtested Sharpe from the trailing daily
VRP and reports it with three explicit caveats (no historical chain to
reprice against, autocorrelation inflating the `sqrt(365)` annualization, and
survivorship of a window that happened not to contain a vol spike) — directly
answering the milestone's own "any Sharpe claim" requirement rather than
avoiding the topic by not computing one.

Entry and live P&L are session-scoped (`state.scratch`, the same mechanism
every other panel's ad hoc UI state already uses), not persisted — matching
`voltk/portfolio.py`'s standing note that real position tracking wasn't in
scope until this milestone, and even here stays a session-local preview, not
a database.

## Known gaps

- Deribit options are European with no dividends, so the binomial model has no
  early-exercise boundary to find and Bachelier has no negative-price market to
  validate against. Both need synthetic validation or a second data source.
- `Universe.implied_rate` takes the rate from the futures basis via
  `F = S*exp(r*tau)`. There is no rates curve to look up, so this is the only
  source; it is exact for the inverse model, where rho is zero regardless.
- The chain parity check, the implied-forward derivation, and the chain filter
  funnel have only been exercised against synthetic fixtures, not live quotes.
  Live Deribit marks will show whether the 5bps parity floor and the filter
  rules need tuning.
- Chain View and Forward Curve each pull book-summary marks and quotes
  independently (`data.marks_for` / `data.quotes_for`), so one render issues
  two live calls to the same endpoint. Matches `marks_for`'s existing
  uncached-per-render cost; not worth a caching layer for two panels.
- SVI calibration and the model-free variance index have only been verified
  against a synthetic chain priced exactly off a known SVI curve (round-trips
  to 1e-6). Live Deribit strikes are unevenly spaced and far more of them,
  so the calibration bounds/seed grid and the variance sum's discretization
  error need a live run to confirm the tolerances still hold.
- Deribit's DVOL construction is not published in detail, so the model-free
  cross-check is validated on shape and order of magnitude, not against an
  independently reproducible reference number.
- The shock ladder's skew and curvature magnitudes are documented defaults,
  not empirically sized the way the level shock is from DVOL. A genuine
  historical level/skew/curvature PCA needs either a bulk historical-chain
  endpoint Deribit doesn't publish, or enough accumulated local snapshots
  (with their SVI fits harvested, which nothing does yet) to decompose —
  neither exists at this point in the project.
- The portfolio section in Greeks Details is session-local scaffolding for
  M4's aggregation requirement, not M8's real position/P&L tracking; it holds
  nothing between reruns and persists nothing.
- M5 targets Core + Extended, not Advanced: no bucketed-vega P&L attribution
  to individual surface control points, no dedicated residual-diagnostics
  section beyond the panel's own explained/residual split, and no dedicated
  regression test that a refactored fitter still reproduces every stored
  snapshot (the bit-for-bit determinism test covers the fitter in isolation,
  not a sweep across accumulated real snapshot history).
- `data.dvol_for` stays live-only under snapshot replay (a deliberate M3
  choice, not new here) — DVOL is a cross-check against an external published
  index with no bracketed-capture/replay concept, so an A-vs-B comparison's
  DVOL context is always "now," not "as of either snapshot."
- The coin/quote conversion fix has only been verified against synthetic
  fixtures (including a deliberately nonzero-rate one), not live Deribit
  marks — the same "needs a live run to confirm" caveat as the rest of the
  surface-fitting stack above.
- M6 targets Core + Extended, not Advanced: no forecasting model (the brief
  frames this as "how do I improve the model by appropriate changes to loss,
  sampling, weighting, or features," not a score chase, and it needs a real
  backtest harness this project doesn't have yet), and no reconciliation of
  M3's own model-free variance index against DVOL *through time* — the
  existing cross-check (`compare_to_dvol`) is still a single-snapshot
  comparison, not a time series.
- The sticky-regime regression is one session's worth of intraday candles on
  whichever strike ladder happens to be listed and liquid enough to have
  candle history at the moment it's run — a single day is one draw, not a
  robust estimate, and the brief's own framing treats it that way ("one pull,
  many observations," not "the definitive answer").
- `get_historical_volatility` does not accept a start/end window — its
  trailing-history length (currently about two weeks) is whatever Deribit
  chooses to return, not something this project controls or can widen for a
  longer reconciliation.
- `historical_vol_for` and `candles_for` are live-only, the same established
  choice as `dvol_for` — history has no meaning to replay against a single
  snapshot, so the Vol History panel is unavailable in snapshot-replay mode.
- M7 targets the brief's Extended tier plus the P&L attribution/adverse-
  selection measurement its own accept criteria require, not the rest of
  Advanced: no comparison across different width policies, no fill-rate
  tradeoff analysis, no vol path in the simulated session (vega P&L reports
  correctly but is zero until one is wired in).
- The simulated session's fill rule is a documented worst case, not
  calibrated against any real fill-probability data — Deribit's public API
  has no historical order-book depth to calibrate one from.
- Quote width coefficients are exposed as UI sliders for inspection, the
  same way M4's shock magnitudes are — not fit to historical P&L, since
  nothing in this project tracks realized quoting P&L over time yet.
- M8 targets one tradeable relationship (vol carry) rather than all three the
  brief names as options (rich wings, calendar structure, vol carry) —
  chosen because it reuses the most already-built machinery. The two
  explicitly optional extras (a hedged structured/barrier product, a BTC vs
  ETH cross-currency portfolio) weren't attempted; either would need real
  new pricing-model or cross-asset work, not an extension of what M8 built.
- The backtested Sharpe is a vol-premium proxy over M6's own trailing DVOL
  history (about two weeks), not a simulated dollar P&L — there's no
  historical option chain to reprice a real position against day by day.
- Position entry/P&L is session-scoped only: it resets on a page reload and
  was never meant to survive one, matching the scope note already in
  `voltk/portfolio.py`.
