"""Parity and bounds checks against observed quotes.

The parity relation is exact in the models, so the interesting behaviour is what
happens on quotes that are stale, wide, or wrong. These tests build a chain from
model prices, perturb it, and check the report says so.
"""

from __future__ import annotations

import pytest

from tests.conftest import BASE, FakeVenue
from voltk.capture import capture
from voltk.instruments import OptionType
from voltk.models.base import CP
from voltk.models.inverse import InverseOption
from voltk.universe import UniverseSpec
from voltk.validation import bounds_violations, parity_report

VOL = 0.6


@pytest.fixture
def universe(venue: FakeVenue):
    return capture(venue, UniverseSpec.of([BASE])).universe


def consistent_marks(universe, vol: float = VOL) -> dict[str, float]:
    """A chain priced off one model, so parity holds by construction."""
    model = InverseOption()
    marks: dict[str, float] = {}
    for inst in universe.options(BASE):
        if inst.strike is None or inst.expiry is None:
            continue
        forward = universe.forward_for(BASE, inst.expiry)
        tau = max(inst.tau(universe.as_of) or 0.0, 0.0)
        cp = CP.CALL if inst.option_type is OptionType.CALL else CP.PUT
        marks[inst.name] = model.price(forward, inst.strike, tau, vol, 0.0, cp)
    return marks


def test_consistent_chain_shows_no_breaches(universe) -> None:
    report = parity_report(universe, consistent_marks(universe), currency=BASE)
    assert report.checked > 0
    assert not report.breaches


def test_a_mispriced_quote_is_flagged(universe) -> None:
    marks = consistent_marks(universe)
    victim = next(name for name in marks if name.endswith("-C"))
    marks[victim] += 0.05

    report = parity_report(universe, consistent_marks(universe) | {victim: marks[victim]},
                           currency=BASE)
    assert len(report.breaches) == 1
    assert report.worst.gap == pytest.approx(0.05, abs=1e-9)


def test_a_wide_spread_excuses_a_small_gap(universe) -> None:
    """A gap inside the combined spread is not evidence of anything."""
    marks = consistent_marks(universe)
    victim = next(name for name in marks if name.endswith("-C"))
    marks[victim] += 0.02

    tight = parity_report(universe, marks, currency=BASE)
    wide = parity_report(universe, marks, currency=BASE, spreads={victim: 0.05})
    assert tight.breaches
    assert not wide.breaches


def test_pairs_without_two_sided_marks_are_skipped(universe) -> None:
    marks = consistent_marks(universe)
    for name in [n for n in marks if n.endswith("-P")]:
        del marks[name]
    report = parity_report(universe, marks, currency=BASE)
    assert report.checked == 0
    assert "No call-put pairs" in report.summary()


def test_quote_settled_relation_differs_from_coin_settled(universe) -> None:
    """The same marks cannot satisfy both parity relations."""
    marks = consistent_marks(universe)
    coin = parity_report(universe, marks, currency=BASE, settles_in_base=True)
    quote = parity_report(universe, marks, currency=BASE, settles_in_base=False)
    assert not coin.breaches
    assert quote.breaches


def test_a_quote_below_intrinsic_is_reported(universe) -> None:
    marks = consistent_marks(universe)
    victim = next(
        inst
        for inst in universe.options(BASE)
        if inst.option_type is OptionType.CALL and inst.strike == min(
            i.strike for i in universe.options(BASE) if i.strike
        )
    )
    marks[victim.name] = 0.0

    violations = bounds_violations(universe, marks, InverseOption(), currency=BASE)
    assert any(inst.name == victim.name and side == "below intrinsic"
               for inst, _, side in violations)


def test_a_quote_above_one_coin_is_reported(universe) -> None:
    marks = consistent_marks(universe)
    victim = next(name for name in marks if name.endswith("-C"))
    marks[victim] = 1.5

    violations = bounds_violations(universe, marks, InverseOption(), currency=BASE)
    assert any(inst.name == victim and side == "above the cap"
               for inst, _, side in violations)


def test_clean_chain_has_no_bounds_violations(universe) -> None:
    marks = consistent_marks(universe)
    assert bounds_violations(universe, marks, InverseOption(), currency=BASE) == []
