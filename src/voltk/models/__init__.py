"""Pricing model registry.

The UI reads this registry, so registering a model here is what makes it
selectable. Order is declared, not alphabetical: the first entry is the default.

Implementations land at M1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Protocol, runtime_checkable

from voltk.greeks import Greeks
from voltk.models.bachelier import Bachelier
from voltk.models.base import CP, PricingError
from voltk.models.binomial import Binomial, Dividend
from voltk.models.black76 import Black76
from voltk.models.black_scholes import BlackScholes
from voltk.models.bounds import ArbitrageError, Bounds
from voltk.models.inverse import InverseOption
from voltk.models.solver import ImpliedVolResult, SolverError, implied_vol_detailed


class Underlying(StrEnum):
    SPOT = "spot"
    FORWARD = "forward"


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
    factory: Callable[[], PricingModel] | None = None
    settles_in_base: bool = False

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
        implemented=True,
        factory=BlackScholes,
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
        implemented=True,
        factory=Black76,
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
        implemented=True,
        factory=Bachelier,
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
        implemented=True,
        factory=Binomial,
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
        rationale=(
            "Settled in the same asset that determines the payoff, so its value in that "
            "asset is bounded and concave rather than linear above the strike."
        ),
        implemented=True,
        factory=InverseOption,
        settles_in_base=True,
    )
)
