# Strategy proposal: delta-hedged short straddle, sized off the measured VRP

## The losing scenario, first

This position is short gamma. It makes money in small, steady increments
(theta) for as long as the underlying doesn't move much, and loses money in
large, fast increments exactly when it does. A single vol spike -- a real
move, a liquidation cascade, a macro surprise -- can erase weeks or months of
collected theta in the time it takes to notice and react. Gamma losses on a
short-vol position are not linear in the size of the move; a move twice as
large costs roughly four times as much (0.5 · Γ · ΔS²), so the tail is where
this trade actually loses, not the middle of the distribution. The historical
backtest below cannot see this risk if the window it's computed over didn't
contain one -- see "What the Sharpe number doesn't tell you."

This is priced in, not hidden: the kill conditions below exist specifically
to cut the position before a spike turns into the loss described above, and
the position is deliberately sized to a stated risk budget rather than to
whatever the market will absorb.

## Thesis

`voltk/variance_premium.py` (M6) measures DVOL against realized volatility
computed independently from perpetual candles. When implied sits above
realized, selling volatility and delta-hedging captures the difference
(Natenberg): a delta-hedged short-gamma position's expected P&L over an
interval is `0.5·Γ·(move)² + Θ·dt`, and Θ for a short position is positive
(collecting time decay) while the Γ term is a loss whose expected size is set
by *realized* vol, not the *implied* vol the position was struck at. If
realized comes in below implied, theta collected exceeds the expected gamma
loss.

This premium is not always positive -- the dashboard's Position panel reads
it live and states plainly when it's negative that this is a reason not to
put the trade on, not a number to explain away.

## Structure

A short straddle (short call + short put, same strike and expiry, struck
near the forward) is the maximum-vega, minimum-initial-delta way to express
a pure volatility view: an ATM straddle carries the most gamma and vega per
contract of any two-leg combination, and its residual delta after both legs
are combined is already small. `voltk/strategy.py::hedge_delta` computes the
exact residual and the panel hedges it with the underlying, so the position
starts genuinely delta-neutral rather than approximately so.

## Sizing

`size_for_vega_budget` picks the largest whole number of straddles whose
total vega stays within a stated risk budget (vega, not premium or notional
-- vega is the honest unit of risk for a position whose entire thesis is
about volatility). The budget is a number the trader states up front, not
backed out from what feels right after seeing the position.

## Capacity

`capacity_from_open_interest` caps the position at a stated participation
rate (10% by default) of the *binding* leg's open interest -- whichever of
the call or put has less depth. A position sized off vega budget alone can
come out larger than the market can actually absorb; the panel shows both
numbers side by side so that mismatch is visible, not discovered at the
worst time.

## Risk

The panel's combined-risk table reports delta (≈0, by construction), gamma,
vega, theta, rho, vanna, and volga for the hedged position, plus the
expected 1-day gamma and theta P&L split out separately using the currently
measured realized vol -- so the sign of the expected edge is visible before
the position is sized, not after.

## Kill conditions

Three independent, always-evaluated checks (`check_kill_conditions`), so a
breach in one doesn't hide a breach in another:

1. **Realized vol thesis** -- stop if trailing realized vol runs past 1.5x
   the vol the position was struck at. The thesis was "implied is rich
   against realized"; if realized catches up and passes implied, the thesis
   is falsified, not just under pressure.
2. **Loss budget** -- stop if mark-to-market loss exceeds half the vega risk
   budget. A budget that only bounds initial sizing and never triggers an
   exit isn't a risk limit.
3. **Time to expiry** -- stop (or refuse to enter) inside 3 days to expiry.
   Gamma accelerates as expiry approaches; the risk this position is being
   compensated for gets materially worse right when there's the least time
   left to react to it.

## What the Sharpe number doesn't tell you

The Position panel computes a backtested Sharpe from the trailing daily
premium series and reports it -- the milestone asks for this to be caveated,
not omitted. Three separate reasons the number overstates what running this
strategy would actually deliver:

- **No historical option chain to reprice against.** The Sharpe is computed
  on the vol premium itself as a directional proxy for the delta-hedged
  edge, not on a simulated dollar P&L from actually holding and hedging a
  position day by day.
- **Autocorrelation inflates the annualization.** Implied and realized vol
  both move slowly day to day, so consecutive daily premium observations
  are far from independent. The `sqrt(365)` annualization assumes
  independent daily draws; applying it to an autocorrelated series produces
  a Sharpe magnitude well beyond what the same average edge would support
  under genuinely independent daily outcomes.
- **Survivorship of the window.** A trailing window that happened not to
  contain a large vol spike will show a clean, positive-looking track
  record for a strategy whose entire risk is concentrated in vol spikes.
  This is the general form of the point above, restated for this specific
  strategy: the backtest can only be as good as the tail events it happened
  to sample, and short-vol strategies are exactly the ones where the
  relevant tail event is rare and expensive.

A real Sharpe from actually running this strategy is lower than the
backtested number, not higher, for all three reasons together.

## What is checked

| Property | Test |
| --- | --- |
| Hedge exactly zeroes combined delta | `test_hedge_delta_zeroes_out_the_combined_position` |
| Hedge changes only delta, not the other Greeks | `test_with_hedge_only_changes_delta` |
| Expected edge is positive when realized stays below entry vol | `test_expected_daily_edge_short_position_profits_when_realized_stays_below_entry_vol` |
| Expected edge is negative when realized exceeds entry vol | `test_expected_daily_edge_short_position_loses_when_realized_exceeds_entry_vol` |
| Sizing floors to whole contracts within budget | `test_size_for_vega_budget_floors_to_whole_contracts` |
| Capacity scales with the stated participation rate | `test_capacity_scales_with_participation` |
| Every kill condition is reported, not just the first breach | `test_kill_conditions_report_every_breach_not_just_the_first` |
| Backtested Sharpe matches a hand-computed case | `test_backtest_premium_sharpe_matches_a_hand_computed_case` |
