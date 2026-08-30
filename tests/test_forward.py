"""Implied forward from put-call parity.

Parity holds exactly in the models, so a chain priced off forward_for must
have its implied forward recover forward_for exactly. A chain priced off the
index instead (Deribit's own display mistake) must have its implied forward
recover the index instead, with a large basis against the traded future --
that gap is the trap's signature.
"""

from __future__ import annotations

import pytest

from tests.conftest import BASE, FakeVenue
from tests.test_validation import consistent_marks, universe  # noqa: F401
from voltk.capture import capture
from voltk.forward import forward_from_parity, implied_forward_curve
from voltk.instruments import OptionType
from voltk.models.base import CP
from voltk.models.black76 import Black76
from voltk.models.inverse import InverseOption
from voltk.universe import UniverseSpec

VOL = 0.6


def quote_settled_marks(universe, vol: float = VOL) -> dict[str, float]:
    """A chain priced off Black-76, so the quote-settled relation holds exactly."""
    model = Black76()
    marks: dict[str, float] = {}
    for inst in universe.options(BASE):
        if inst.strike is None or inst.expiry is None:
            continue
        forward = universe.forward_for(BASE, inst.expiry)
        tau = max(inst.tau(universe.as_of) or 0.0, 0.0)
        rate = universe.implied_rate(BASE, inst.expiry)
        cp = CP.CALL if inst.option_type is OptionType.CALL else CP.PUT
        marks[inst.name] = model.price(forward, inst.strike, tau, vol, rate, cp)
    return marks


def index_priced_marks(universe, vol: float = VOL) -> dict[str, float]:
    """The trap: every option priced off the index instead of the forward."""
    model = InverseOption()
    index = universe.index(BASE)
    marks: dict[str, float] = {}
    for inst in universe.options(BASE):
        if inst.strike is None or inst.expiry is None:
            continue
        tau = max(inst.tau(universe.as_of) or 0.0, 0.0)
        cp = CP.CALL if inst.option_type is OptionType.CALL else CP.PUT
        marks[inst.name] = model.price(index, inst.strike, tau, vol, 0.0, cp)
    return marks


def test_forward_from_parity_matches_the_forward_used_to_construct_prices() -> None:
    forward, strike, tau, vol = 65000.0, 68000.0, 0.25, 0.6
    model = Black76()
    call = model.price(forward, strike, tau, vol, 0.0, CP.CALL) / forward
    put = model.price(forward, strike, tau, vol, 0.0, CP.PUT) / forward

    recovered = forward_from_parity(strike, call, put, settles_in_base=True)

    assert recovered == pytest.approx(forward, rel=1e-6)


def test_forward_from_parity_returns_none_on_a_degenerate_strike() -> None:
    assert forward_from_parity(70000.0, 1.5, 0.0, settles_in_base=True) is None


def test_forward_from_parity_quote_settled_uses_the_discount_factor() -> None:
    forward, strike, discount = 65000.0, 68000.0, 0.98
    diff = (forward - strike) * discount
    recovered = forward_from_parity(strike, diff, 0.0, settles_in_base=False, discount=discount)
    assert recovered == pytest.approx(forward, rel=1e-9)


def test_consistent_marks_recover_the_traded_future(universe) -> None:
    curve = implied_forward_curve(universe, consistent_marks(universe), currency=BASE)
    assert curve
    for point in curve:
        assert point.forward == pytest.approx(point.traded_future, rel=1e-6)
        assert abs(point.basis_bps) < 1.0


def test_quote_settled_relation_also_recovers_the_forward(universe) -> None:
    curve = implied_forward_curve(
        universe, quote_settled_marks(universe), currency=BASE, settles_in_base=False
    )
    assert curve
    for point in curve:
        assert point.forward == pytest.approx(point.traded_future, rel=1e-6)
        assert abs(point.basis_bps) < 1.0


def test_the_index_trap_produces_a_material_basis(universe) -> None:
    index = universe.index(BASE)
    curve = implied_forward_curve(universe, index_priced_marks(universe), currency=BASE)
    assert curve
    for point in curve:
        assert point.forward == pytest.approx(index, rel=1e-6)
        assert abs(point.basis_bps) > 50
        expected_basis = index - point.traded_future
        assert point.basis == pytest.approx(expected_basis, rel=1e-6)


def test_expiries_without_two_sided_marks_are_absent(universe) -> None:
    marks = consistent_marks(universe)
    for name in [n for n in marks if n.endswith("-P")]:
        del marks[name]
    assert implied_forward_curve(universe, marks, currency=BASE) == ()


def test_pairs_used_counts_strikes_actually_used(universe) -> None:
    marks = consistent_marks(universe)
    expiry = universe.expiries(BASE)[0]
    strike = universe.strikes(BASE, expiry)[0]
    dropped = [
        inst.name
        for inst in universe.options(BASE)
        if inst.expiry == expiry and inst.strike == strike
    ]
    for name in dropped:
        del marks[name]

    curve = implied_forward_curve(universe, marks, currency=BASE)
    point = next(p for p in curve if p.expiry == expiry)
    assert point.pairs_used == len(universe.strikes(BASE, expiry)) - 1
    assert strike not in [s for s, _ in point.per_strike]


def test_a_degenerate_strike_is_skipped_not_raised(universe) -> None:
    marks = consistent_marks(universe)
    expiry = universe.expiries(BASE)[0]
    strike = universe.strikes(BASE, expiry)[0]
    call = next(
        inst for inst in universe.options(BASE)
        if inst.expiry == expiry and inst.strike == strike and inst.option_type == OptionType.CALL
    )
    put = next(
        inst for inst in universe.options(BASE)
        if inst.expiry == expiry and inst.strike == strike and inst.option_type == OptionType.PUT
    )
    marks[call.name] = marks[put.name] + 1.2  # C - P = 1.2 makes 1 - (C-P) negative

    curve = implied_forward_curve(universe, marks, currency=BASE)
    point = next(p for p in curve if p.expiry == expiry)
    assert point.pairs_used == len(universe.strikes(BASE, expiry)) - 1
    assert strike not in [s for s, _ in point.per_strike]
