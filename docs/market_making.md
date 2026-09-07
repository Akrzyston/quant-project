# Quoting and the simulated market-making session

## In one sentence

Quote width is a sum of named, stated risk terms rather than a hand-set
number, inventory shades the quote via the standard Avellaneda-Stoikov
reservation price, and a simulated session's P&L is attributed into edge
captured versus a Taylor-decomposed markout move, reusing the same
`attribute_pnl` identity M5 already built.

## Width

`voltk.quoting.derive_width` sums three terms, each a real cost a market
maker is compensated for:

- **Fit term** -- `vega * |fit_residual_vol|`. If the raw market mark at this
  strike sits away from the fitted curve, that's model uncertainty, and
  vega converts a vol-space uncertainty into a dollar one.
- **Gamma term** -- `0.5 * gamma * (expected move)^2`, the textbook cost of
  not being able to rehedge continuously. The expected move is
  `underlying * vol * sqrt(rehedge_interval)`, sized off this option's own
  implied vol rather than a guessed constant. `rehedge_interval` is a
  stated one-hour assumption -- how often a maker actually requotes.
- **Liquidity term** -- a fraction of the live market's own half-spread. A
  thin market is itself evidence that unwinding a position will cost more.

Coefficients on each term are stated parameters in the Mock Quoter panel,
not per-instrument tuning -- the function mapping them to a width never
changes.

## Inventory skew

`reservation_price` shifts the quote's center away from theoretical value
by `inventory * risk_aversion * vol^2 * tau` (Avellaneda & Stoikov, 2008).
Positive inventory pulls the reservation price down, making the ask more
attractive to sell down the position and the bid less attractive to add to
it; negative inventory does the reverse. This is the standard formula, not
a derived one -- the width formula above is what's actually specific to
this project.

## The simulated session

`voltk.market_making.simulate_session` walks a real path of the underlying
(perpetual candles, basis-adjusted to a forward the same way every other
forward in this project is) and vol held constant for the session. A fill
happens whenever the *next* step's theoretical value would have moved
through the currently quoted side.

This is a maximally-informed-counterparty rule: every fill is, by
construction, one that was about to go against us. It gives an upper bound
on adverse selection, not an average trading session -- stated explicitly
in the panel, not left implicit. A real book sees plenty of fills that
never get run over.

Each fill's markout is reattributed via `voltk.pnl.attribute_pnl` -- the
same identity M5 built for snapshot-to-snapshot P&L, here applied
fill-to-markout instead. `edge_captured` is the width actually realized at
the fill; `delta_term + gamma_term` is what this project calls adverse
selection (the directional cost of the market moving against the resulting
position); `vega_term` is reported on its own, and is zero in the current
implementation since vol is held constant per session -- the machinery
supports a real per-step vol path, it just isn't wired to one yet. The
residual is never folded into another term.

## What is checked

| Property | Test |
| --- | --- |
| Width sums its named terms | `test_derive_width_sums_its_named_terms` |
| Gamma term matches the Taylor rehedge cost | `test_derive_width_gamma_term_matches_the_taylor_rehedge_cost` |
| Reservation price shifts away from inventory | `test_reservation_price_shifts_away_from_positive_inventory` |
| A fill only happens when the path crosses the quote | `test_no_fill_when_the_path_stays_inside_the_quote` |
| Edge captured matches the quoted half-width absent skew | `test_edge_captured_is_the_quoted_half_width_with_no_inventory_skew` |
| Fill P&L terms sum to the total, exactly | `test_fill_pnl_terms_sum_to_total_pnl` |
| Inventory skew changes which side fills | `test_inventory_skew_can_trigger_a_second_fill_that_covers_the_position` |
