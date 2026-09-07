# P&L attribution, derived

## In one sentence

The change in an option's value between two points in time can be decomposed
into a delta, a gamma, a vega and a theta term taken from its Greeks at the
*start* of the period, plus a residual that is reported, not absorbed -- a
large residual is evidence the model is wrong, not noise to smooth over.

## Setup and notation

- `V(S, sigma, tau)` is the option's value, in whatever currency its Greeks
  are reported in (coin or cash -- see "Which Greeks" below).
- Snapshot `A` and snapshot `B` are two points in time, `A` earlier.
- `dS = S_B - S_A`, `dsigma = sigma_B - sigma_A`, `dt = t_B - t_A` (elapsed
  calendar time, positive -- `tau` itself *shrinks* by `dt` as time passes).
- Greeks are all taken **at snapshot A** -- this is a forward-looking Taylor
  expansion from the start of the period, not a symmetric or backward one.

## The identity

A second-order Taylor expansion of `V` around snapshot A, in `S` and `sigma`,
plus a first-order term in elapsed time:

```
V_B - V_A  ≈  Delta_A * dS  +  (1/2) Gamma_A * dS^2  +  Vega_A * dsigma  +  Theta_A * dt  +  residual
```

`residual` is *defined* as whatever is left over -- `attribute_pnl` computes
the actual repriced difference `V_B - V_A` independently (by calling
`model.price` at both snapshots, not by trusting the Taylor sum), and the
residual is the gap between that actual number and the four explained terms.
It is never subtracted away or folded into another term.

## The theta sign convention

This library defines `theta = -dV/dtau` (confirmed against
`tests/test_greeks_finite_difference.py::test_theta`'s own finite-difference
harness, which flips the sign: `numeric = -first_derivative(price, tau, ...)`).
Since `tau` decreases as calendar time passes (`dtau = -dt`), the chain rule
gives `dV = -theta * dtau = theta * dt` -- so `Theta_A * dt` with `dt` as
*positive elapsed time* is exactly right, not `-Theta_A * dt`. Verified
numerically, not just derived: with nothing else moving, `theta * dt` for a
10-day roll matches the actual repriced difference to within 2%.

## Units: read them off the Greeks, don't assume them

`Greeks.vega_bump` and `Greeks.theta_period` state what one unit of the
reported sensitivity corresponds to. Every model in this library currently
sets both to `1.0` -- a raw analytic partial derivative, "per one full unit of
vol" and "per one full year," not the "per 1% vol, per 1 day" convention the
`Greeks` dataclass's own *defaults* (`vega_bump=0.01`, `theta_period=1/365`)
would suggest if you didn't check. `attribute_pnl` divides by these fields
explicitly (`vega_term = vega * (dsigma / vega_bump)`, similarly for theta)
rather than hardcoding today's `1.0` convention, so the formula stays correct
if that ever changes -- `test_vega_and_theta_divide_by_their_own_bump_period_metadata`
constructs `Greeks` with non-`1.0` values specifically to prove this.

## Which Greeks: coin or cash, and a second, subtler unit trap

For a quote-settled model, coin and cash coincide and there is nothing further
to decide. For a coin-settled position (`InverseOption`), the natural choice
is the **cash** Greeks from `voltk.greeks.cash_greeks_from_coin` -- the
attribution should answer "how much did my USD P&L change," not "how much did
my BTC-denominated book value change."

This is where a second, subtler trap sits, distinct from the "coin isn't a
units conversion" trap `cash_greeks_from_coin` itself documents:
`InverseOption`'s own Greeks are **forward** Greeks (`delta = dV_coin/dF`),
but `cash_greeks_from_coin`'s delta and gamma are **spot** derivatives
(`dV_cash/dS`) -- confirmed by construction, since `V_cash(S) = V_coin(F(S))*S`
is differentiated with respect to `S`, not `F`. So `dS` in the attribution
identity must be the **spot** difference between the two snapshots when using
cash Greeks, not the forward difference -- using the forward difference here
(the model's own native convention) would silently reintroduce a real error,
the same shape of mistake `cash_greeks_from_coin`'s docstring already warns
about for gamma. `app/panels/snapshot_browser.py`'s `_attribute_one` branches
on `spec.settles_in_base` for exactly this reason.

## Why the residual is small but not zero

A 2nd-order expansion in `(S, sigma)` omits the cross terms -- vanna
(`d^2V/dS dsigma`) and volga (`d^2V/dsigma^2`), both already computed
elsewhere in this library (`voltk/greeks.py`, M4) but not folded into this
5-term identity, matching the brief's stated formula exactly. Verified
numerically: when spot, vol and time all move by a small, *common* factor
together, the residual shrinks as `O(h^2)` (clean second-order Taylor
convergence -- confirmed down to a 0.01x-scale move). At a realistic combined
move (2.5% spot, 5% vol, 10 days), the residual settles at about 1% of the
actual P&L -- the expected size of the omitted vanna/volga terms, not a sign
of a bug. A residual much larger than that on real data would be exactly the
kind of thing worth investigating: a large unexplained term is evidence
the model is wrong, not noise.

## What is checked

| Property | Test |
| --- | --- |
| Small proportional move closes the identity tightly | `test_small_proportional_move_closes_the_identity_tightly` |
| Larger move has a bounded, explainable residual | `test_larger_combined_move_has_a_bounded_not_zero_residual` |
| Theta sign convention, pure time roll | `test_theta_sign_convention_pure_time_roll` |
| Residual is reported, never absorbed | `test_residual_is_reported_not_absorbed_on_a_curved_case` |
| vega_bump/theta_period are read, not assumed | `test_vega_and_theta_divide_by_their_own_bump_period_metadata` |
| Coin-settled attribution uses spot, not forward, for dS | `app/panels/snapshot_browser.py::_attribute_one` (settles_in_base branch) |
