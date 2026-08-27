# voltk — volatility toolkit + dashboard

A reusable pricing/risk library with a Streamlit view on the outside. The
dashboard accumulates one panel per milestone.

```
uv run pytest -q                        # 90 tests
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

## Known gaps

- Deribit options are European with no dividends, so the binomial model has no
  early-exercise boundary to find and Bachelier has no negative-price market to
  validate against. Both need synthetic validation or a second data source.
- There is no rates curve. The `rate` argument will have to come from futures
  basis, decided at M2.
