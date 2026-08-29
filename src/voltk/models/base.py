"""Shared numerics for the pricing models."""

from __future__ import annotations

import math
from enum import StrEnum

SQRT_2PI = math.sqrt(2.0 * math.pi)

# Below this, time value is numerically indistinguishable from zero and the
# lognormal parameterisation degenerates.
MIN_TAU = 1e-12
MIN_VOL = 1e-12


class CP(StrEnum):
    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> float:
        return 1.0 if self is CP.CALL else -1.0


class PricingError(ValueError):
    """Inputs outside the domain a model is defined on."""


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def check_inputs(underlying: float, strike: float, tau: float, rate: float) -> None:
    if underlying <= 0:
        raise PricingError(f"Underlying must be positive, got {underlying}.")
    if strike <= 0:
        raise PricingError(f"Strike must be positive, got {strike}.")
    if tau < 0:
        raise PricingError(f"Time to expiry must not be negative, got {tau}.")
    if not math.isfinite(rate):
        raise PricingError(f"Rate must be finite, got {rate}.")


def discount(rate: float, tau: float) -> float:
    return math.exp(-rate * tau)


def intrinsic(forward: float, strike: float, cp: CP) -> float:
    """Undiscounted intrinsic value in forward terms."""
    return max(cp.sign * (forward - strike), 0.0)
