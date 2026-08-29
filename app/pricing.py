"""Pricing helpers for the view layer.

Assembles the arguments a model needs from a captured universe. Contains no
maths: every number here comes from voltk.
"""

from __future__ import annotations

from dataclasses import dataclass

from voltk.instruments import Instrument, OptionType
from voltk.models import ModelSpec
from voltk.models.base import CP
from voltk.models.bounds import ArbitrageError
from voltk.models.solver import ImpliedVolResult, SolverError, implied_vol_detailed
from voltk.universe import Universe


@dataclass(frozen=True, slots=True)
class PricingInputs:
    underlying: float
    strike: float
    tau: float
    rate: float
    cp: CP
    settles_in_base: bool


def inputs_for(
    universe: Universe, instrument: Instrument, spec: ModelSpec, rate: float | None = None
) -> PricingInputs:
    """Assemble model arguments. rate defaults to the basis-implied rate."""
    if instrument.strike is None or instrument.expiry is None:
        raise ValueError(f"{instrument.name} is not a dated option.")

    if rate is None:
        rate = universe.implied_rate(instrument.base_currency, instrument.expiry)
    forward = universe.forward_for(instrument.base_currency, instrument.expiry)
    tau = instrument.tau(universe.as_of) or 0.0
    # Spot-parameterised models take spot; forward-parameterised take the forward.
    underlying = forward if spec.underlying.value == "forward" else universe.index(
        instrument.base_currency
    )
    return PricingInputs(
        underlying=underlying,
        strike=instrument.strike,
        tau=max(tau, 0.0),
        rate=rate,
        cp=CP.CALL if instrument.option_type is OptionType.CALL else CP.PUT,
        settles_in_base=spec.settles_in_base,
    )


def solve_implied(model, price: float, args: PricingInputs) -> ImpliedVolResult | None:
    """Returns None when the quote admits no implied vol."""
    try:
        return implied_vol_detailed(
            model, price, args.underlying, args.strike, args.tau, args.rate, args.cp
        )
    except (ArbitrageError, SolverError):
        return None
