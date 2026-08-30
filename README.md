# voltk — volatility toolkit + dashboard

A reusable pricing/risk library with a Streamlit view on the outside. The
dashboard accumulates one panel per milestone.

```
uv run pytest -q                        # 585 cases
uv run streamlit run streamlit_app.py
```

## The boundary

`src/voltk/` holds instruments, market data, capture, and pricing. It imports no
Streamlit, Plotly, or `app`. Two tests enforce this: a static AST scan of every
module, and a runtime import with the view packages poisoned in `sys.meta_path`.

`app/` renders. No pricing, fitting, or risk logic.

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
