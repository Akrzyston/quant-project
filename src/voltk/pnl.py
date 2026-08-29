"""P&L attribution: the standard Greeks-based Taylor decomposition.

ΔPnL ≈ Δ·ΔS + ½Γ·ΔS² + Vega·Δσ + Θ·Δt + residual

Terms are taken from the START-of-period Greeks (snapshot A), against the
ACTUAL repriced difference -- the residual is whatever this 2nd-order
expansion misses (cross terms like vanna·ΔS·Δσ, higher-order moves, and any
genuine model misspecification). It is always reported as its own field,
never subtracted away or folded into another term: a large residual is
evidence the model is wrong, not noise to be absorbed.

Vega and theta are divided by vega_bump/theta_period rather than assumed to
be "per unit vol" and "per year" -- every model in this library currently
sets both to 1.0 (raw analytic derivatives), but the Greeks dataclass itself
declares them as real per-instance metadata, not a global constant, so the
formula reads them off the actual Greeks object rather than hardcoding
today's convention.
"""

from __future__ import annotations

from dataclasses import dataclass

from voltk.greeks import Greeks


@dataclass(frozen=True, slots=True)
class PnlAttribution:
    price_a: float
    price_b: float
    actual_pnl: float
    delta_term: float
    gamma_term: float
    vega_term: float
    theta_term: float
    residual: float

    @property
    def explained(self) -> float:
        return self.actual_pnl - self.residual


def attribute_pnl(
    greeks_a: Greeks,
    price_a: float,
    price_b: float,
    d_underlying: float,
    d_vol: float,
    d_t_years: float,
) -> PnlAttribution:
    """Per-contract (unit size). Callers scale by position size themselves,
    the same way voltk.portfolio.aggregate_greeks scales at the panel layer
    rather than baking sizing into the Greeks object.
    """
    delta_term = greeks_a.delta * d_underlying
    gamma_term = 0.5 * greeks_a.gamma * d_underlying * d_underlying
    vega_term = greeks_a.vega * (d_vol / greeks_a.vega_bump)
    theta_term = greeks_a.theta * (d_t_years / greeks_a.theta_period)

    actual_pnl = price_b - price_a
    residual = actual_pnl - (delta_term + gamma_term + vega_term + theta_term)

    return PnlAttribution(
        price_a=price_a,
        price_b=price_b,
        actual_pnl=actual_pnl,
        delta_term=delta_term,
        gamma_term=gamma_term,
        vega_term=vega_term,
        theta_term=theta_term,
        residual=residual,
    )
