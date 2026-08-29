# The inverse option, derived

## In one sentence

A coin-settled call is not a Black-Scholes call because it pays out in the same
asset whose price determines the payoff, so its value measured in that asset is
`max(S-K,0)/S` — bounded above by one coin and concave in `S` — rather than the
unbounded, linear-above-strike payoff Black-Scholes prices.

## Setup and notation

- `S` is the index: units of quote currency per coin.
- `K` is the strike, quoted in the quote currency.
- `F` is the forward to expiry, taken from the futures curve.
- `tau` is the year fraction to expiry, `sigma` the lognormal vol of `S`.

The venue quotes premium, margin, and settlement in the coin. A call delivers
`max(S_T - K, 0) / S_T` coins at expiry.

## Route 1: the replicating portfolio

Take the settlement-currency payoff and rewrite it without any probabilistic
argument at all:

```
max(S - K, 0) / S  =  (1 - K/S)^+  =  K * (1/K - 1/S)^+
```

Write `X = 1/S`. The right-hand side is `K` puts on `X` struck at `1/K`. This is
an identity, not an approximation, and it holds path by path.

That leaves one question: what is the forward of `X`?

Change numeraire from the quote-currency money market to the coin itself. The
Radon-Nikodym derivative between the coin measure `Q^S` and the `T`-forward
measure `Q^T` is proportional to `S_T / F`. So

```
E^S[1/S_T]  =  E^T[(S_T / F) * (1/S_T)]  =  1/F
```

`X` is a martingale under the coin measure with forward `1/F`. If `S` is
lognormal with vol `sigma`, so is `X`, with the same vol — inverting a lognormal
negates the log but does not change its dispersion.

Under the coin numeraire the coin money market is the numeraire, so there is no
discounting left to apply. The value in coins is:

```
V = K * BlackPut(forward = 1/F, strike = 1/K, vol = sigma, tau)
```

This is `InverseOption.price_via_replication`.

## Route 2: value in the quote currency, then convert

The coins delivered at expiry are worth exactly `max(S_T - K, 0)` in the quote
currency, because that is what `(max(S_T - K, 0)/S_T) * S_T` equals. So the
quote-currency value of the contract is the ordinary Black-76 value. Divide by
spot to get the premium the venue actually quotes:

```
V = Black76(F, K, tau, sigma, r) / S
```

With `F = S * exp(r * tau)` the discount factor cancels against the spot, leaving

```
V = [F N(d1) - K N(d2)] / F
```

This is `InverseOption.price`, and it shows something the first route hides: the
coin premium does not depend on the interest rate at all once the forward is
given. `test_inverse_rho_is_exactly_zero` asserts equality, not closeness.

## The two routes agree

Expanding route 1 with `d1' = -d2` and `d2' = -d1`:

```
K * [(1/K) N(-d2') - (1/F) N(-d1')]  =  N(d1) - (K/F) N(d2)
```

which is route 2. `test_both_derivations_agree` checks this across 90 parameter
combinations; the two implementations share no code beyond Black-76 itself and
agree to roughly 1e-17.

## Greeks in the settlement currency

Differentiating `V = N(d1) - (K/F) N(d2)` with respect to `F`, the `N(d1)` terms
cancel against each other and what survives is:

```
delta = K N(d2) / F^2
```

Vega and theta collapse to a single term each via `F phi(d1) = K phi(d2)`:

```
vega  =  (K/F) phi(d2) sqrt(tau)
theta = -(K/F) phi(d2) sigma / (2 sqrt(tau))
rho   =  0
```

Put greeks are not derived separately. Inverse parity is `C - P = 1 - K/F`, so
differentiating that relation gives the put from the call exactly:

```
delta_put = delta_call - K/F^2
gamma_put = gamma_call + 2K/F^3
vega_put  = vega_call          (parity carries no vol)
```

Deriving them this way rather than independently means the two can never drift
apart under later edits.

## Delta against strike, and why the shape inverts

`delta = K N(d2) / F^2` is zero at both ends of the strike axis and positive in
between.

At low strikes `N(d2)` tends to one but `K` tends to zero, so the product
vanishes. The economic statement: a deep in-the-money coin-settled call is worth
`1 - K/F` coins, which is nearly one coin no matter what the underlying does. You
hold approximately one coin, and one coin is worth one coin. There is no
remaining exposure to measure.

At high strikes `N(d2)` decays faster than `K` grows, so the product vanishes
again — the ordinary reason an out-of-the-money option has no delta.

The peak sits between. In the quote currency the same contract has the familiar
monotone delta falling from one to zero; `test_quote_currency_delta_is_monotone_by_contrast`
pins that difference.

This is the practical trap. Someone hedging a coin-settled book with
quote-currency deltas is most wrong exactly where they feel safest: deep in the
money, where the quote-currency delta reads near one and the true coin delta is
near zero.

## If the underlying doubles or halves

Hold `K` fixed and move `F`.

Doubling. The call value approaches `1 - K/F` from below, so the upside still
unrealised is about `K/F`. Every doubling of `F` halves that remainder. The
position converges to being long one coin and stops responding to the underlying;
`test_doubling_the_underlying_halves_remaining_upside` checks the halving to 2%.

Halving. The option moves out of the money and its coin value collapses toward
zero, with the additional feature that the coins it would deliver are themselves
worth less. Both effects push the same way, which is why coin-settled downside is
sharper than the quote-currency view suggests.

The asymmetry is the point: capped on the way up, compounding on the way down.
Delta alone does not capture it, which is what makes gamma reporting in the
settlement currency matter at M4.

## How quanto greeks differ

A quanto pays a foreign-asset payoff in domestic currency at a **fixed**
conversion rate. Its forward carries a correlation adjustment:

```
F_quanto = F * exp(-rho * sigma_S * sigma_FX * tau)
```

Three differences from anything in the traditional model:

1. **Vega in a rate you do not trade.** A quanto has sensitivity to FX vol even
   though no FX option is in the contract. The traditional model has no such term.
2. **Correlation is a risk factor.** There is a derivative with respect to `rho`
   with no analogue in a single-asset model, and it cannot be hedged with
   vanillas on either underlying alone.
3. **Delta is scaled, not reshaped.** The fixed conversion rate is a constant
   multiplier, so quanto delta keeps the ordinary monotone shape.

The inverse option is the opposite case on every count. The conversion asset *is*
the underlying, so correlation is exactly one and the vol of the conversion rate
is exactly the vol of the underlying — nothing to estimate and nothing to hedge
separately. The adjustment is therefore an exact change of numeraire rather than
a correlation input, which is why the result is a clean closed form instead of an
approximation. And because the conversion rate is stochastic and identical to the
underlying, delta is genuinely reshaped rather than merely scaled.

Put briefly: a quanto adds a risk factor and keeps the shape; an inverse adds no
risk factor and changes the shape.

## What is checked

| Property | Test |
| --- | --- |
| Two derivations agree | `test_both_derivations_agree` |
| Parity `C - P = 1 - K/F` | `test_inverse_put_call_parity` |
| Value capped at one coin | `test_call_value_never_exceeds_one_coin` |
| Saturation at `1 - K/F` | `test_deep_in_the_money_call_saturates` |
| Delta vanishes at both extremes | `test_settlement_currency_delta_vanishes_at_both_extremes` |
| Rate independence, exactly | `test_inverse_rho_is_exactly_zero` |
| Quote-currency value recovers Black-76 | `test_quote_currency_value_recovers_black76` |
| Greeks against finite difference | `test_greeks_finite_difference.py` |
