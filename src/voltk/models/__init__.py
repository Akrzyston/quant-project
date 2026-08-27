"""Pricing model registry.

The UI reads this registry, so registering a model here is what makes it
selectable. Order is declared, not alphabetical: the first entry is the default.

Implementations land at M1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from voltk.greeks import Greeks


class Underlying(StrEnum):
    SPOT = "spot"
    FORWARD = "forward"


class CP(StrEnum):
    CALL = "call"
    PUT = "put"


@runtime_checkable
class PricingModel(Protocol):
    def price(
        self, forward: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> float: ...

    def implied_vol(
        self, price: float, forward: float, strike: float, tau: float, rate: float, cp: CP
    ) -> float: ...

    def greeks(
        self, forward: float, strike: float, tau: float, vol: float, rate: float, cp: CP
    ) -> Greeks: ...


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    display_name: str
    order: int
    underlying: Underlying
    vol_convention: str
    milestone: str
    rationale: str
    implemented: bool = False
    factory: type | None = None

    def build(self) -> PricingModel:
        if self.factory is None:
            raise NotImplementedError(
                f"{self.display_name} is scheduled for {self.milestone}."
            )
        return self.factory()


_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.key in _REGISTRY:
        raise ValueError(f"Model key {spec.key!r} is already registered.")
    _REGISTRY[spec.key] = spec
    return spec


def all_models() -> list[ModelSpec]:
    return sorted(_REGISTRY.values(), key=lambda s: (s.order, s.key))


def get(key: str) -> ModelSpec:
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"No model registered under {key!r}. Registered: {known}.") from None


register(
    ModelSpec(
        key="black_scholes",
        display_name="Black-Scholes",
        order=10,
        underlying=Underlying.SPOT,
        vol_convention="lognormal",
        milestone="M1",
        rationale="Spot-based lognormal. The reference case for every other model.",
    )
)
register(
    ModelSpec(
        key="black_76",
        display_name="Black-76",
        order=20,
        underlying=Underlying.FORWARD,
        vol_convention="lognormal",
        milestone="M1",
        rationale="Forward-based. What options on futures require.",
    )
)
register(
    ModelSpec(
        key="bachelier",
        display_name="Bachelier",
        order=30,
        underlying=Underlying.FORWARD,
        vol_convention="normal",
        milestone="M1",
        rationale="Normal vol, for markets where prices can go negative or vol is quoted normally.",
    )
)
register(
    ModelSpec(
        key="binomial_american",
        display_name="Binomial (American)",
        order=40,
        underlying=Underlying.SPOT,
        vol_convention="lognormal",
        milestone="M1",
        rationale="Early exercise with discrete dividends. No closed form.",
    )
)
register(
    ModelSpec(
        key="inverse_deribit",
        display_name="Inverse",
        order=50,
        underlying=Underlying.FORWARD,
        vol_convention="lognormal",
        milestone="M1",
        rationale="Coin-settled, so the quote-currency payoff is non-linear.",
    )
)
