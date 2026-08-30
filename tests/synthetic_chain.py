"""A synthetic option chain priced off a known raw-SVI curve.

Built directly, bypassing FakeVenue/capture entirely, with a wide strike
ladder -- FakeVenue's fixed 3 strikes are enough for pricing/parity tests but
not for fitting a 5-parameter SVI slice: with exactly 3 points the calibrated
curve exactly interpolates them regardless of the fit, so a test built on it
cannot tell a correct calibration from a wrong one.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from tests.conftest import option_payload
from voltk.forward import ImpliedForward
from voltk.instruments import Instrument
from voltk.models.base import CP
from voltk.models.inverse import InverseOption
from voltk.universe import ForwardPoint, Universe, UniverseSpec

CURRENCY = "XBT"
AS_OF = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)

DEFAULT_SVI = dict(a=0.02, b=0.10, rho=-0.3, m=0.0, sigma=0.2)


def svi_total_variance(
    k: float, *, a: float, b: float, rho: float, m: float, sigma: float
) -> float:
    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))


def build_synthetic_chain(
    *,
    forward: float = 60_000.0,
    tau: float = 30.0 / 365.0,
    svi: dict[str, float] | None = None,
    n_strikes: int = 15,
    k_span: float = 1.3,
    rate: float = 0.0,
) -> tuple[Universe, dict[str, float], ImpliedForward]:
    """A one-expiry chain priced exactly off a known raw-SVI curve.

    Returns (universe, marks, implied_forward). implied_forward is built by
    hand with the known forward rather than derived via
    implied_forward_curve, isolating variance.py/surface.py tests from
    forward.py's own already-tested parity derivation. By default
    index_prices is set equal to the forward (empty forward_curve, so
    Universe.forward_for falls back to the index) so implied_rate resolves
    to exactly 0.0, keeping the discount factor trivial. Passing a nonzero
    `rate` instead sets index = forward*exp(-rate*tau) and populates
    forward_curve with the given forward, so implied_rate resolves to
    exactly that rate -- for tests that specifically need a real discount
    factor to exercise (e.g. checking a formula's rate-handling, which a
    permanently-zero rate can never distinguish from a bug).

    Priced via InverseOption, not Black76: real Deribit option marks are
    coin-denominated (confirmed against Deribit's own docs -- "Bitcoin
    options are priced in Bitcoin"), not quote-currency, and a fixture that
    used quote-scale prices masked a real coin/quote conversion bug in
    smile_points for a long time. This fixture must match what a real chain
    actually looks like.
    """
    svi = svi or DEFAULT_SVI
    expiry = AS_OF + timedelta(seconds=tau * 365.0 * 24 * 3600)
    model = InverseOption()

    instruments: list[Instrument] = []
    marks: dict[str, float] = {}
    for i in range(n_strikes):
        k = -k_span + 2 * k_span * i / (n_strikes - 1)
        strike = forward * math.exp(k)
        w = svi_total_variance(k, **svi)
        vol = math.sqrt(w / tau)

        for call in (True, False):
            cp = CP.CALL if call else CP.PUT
            price = model.price(forward, strike, tau, vol, 0.0, cp)
            inst = Instrument.from_deribit(option_payload(CURRENCY, expiry, strike, call))
            instruments.append(inst)
            marks[inst.name] = price

    if rate == 0.0:
        index = forward
        forward_curve: tuple[ForwardPoint, ...] = ()
    else:
        index = forward * math.exp(-rate * tau)
        forward_curve = (
            ForwardPoint(currency=CURRENCY, expiry=expiry, forward=forward, instrument_name="synthetic-future"),
        )

    universe = Universe(
        spec=UniverseSpec.of([CURRENCY]),
        as_of=AS_OF,
        instruments=tuple(instruments),
        index_prices={CURRENCY: index},
        forward_curve=forward_curve,
    )
    implied_forward = ImpliedForward(
        currency=CURRENCY,
        expiry=expiry,
        forward=forward,
        traded_future=forward,
        basis=0.0,
        basis_bps=0.0,
        pairs_used=n_strikes,
        per_strike=(),
    )
    return universe, marks, implied_forward
