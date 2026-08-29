"""Chain hygiene filtering: what gets dropped, why, and how much."""

from __future__ import annotations

from tests.conftest import BASE, FakeVenue
from voltk.capture import capture
from voltk.chain import filter_chain
from voltk.marketdata.parse import Quote
from voltk.universe import UniverseSpec


def _instruments(venue: FakeVenue):
    return capture(venue, UniverseSpec.of([BASE])).universe.options(BASE)


def _full_marks(instruments) -> dict[str, float]:
    return {inst.name: 0.05 for inst in instruments}


def test_missing_quote_rule_fires_correctly(venue: FakeVenue) -> None:
    instruments = _instruments(venue)
    marks = _full_marks(instruments)
    missing = instruments[0]
    del marks[missing.name]

    report = filter_chain(instruments, marks)
    outcome = next(o for o in report.outcomes if o.name == "missing quote")

    assert outcome.dropped == (missing.name,)
    assert outcome.dropped_fraction == outcome.dropped_count / outcome.total


def test_non_positive_mark_rule_fires_correctly(venue: FakeVenue) -> None:
    instruments = _instruments(venue)
    marks = _full_marks(instruments)
    zero, negative = instruments[0], instruments[1]
    marks[zero.name] = 0.0
    marks[negative.name] = -1.0

    report = filter_chain(instruments, marks)
    non_positive = next(o for o in report.outcomes if o.name == "non-positive mark")
    missing = next(o for o in report.outcomes if o.name == "missing quote")

    assert set(non_positive.dropped) == {zero.name, negative.name}
    assert missing.dropped == ()


def test_an_instrument_is_never_double_counted(venue: FakeVenue) -> None:
    instruments = _instruments(venue)
    marks = _full_marks(instruments)
    del marks[instruments[0].name]

    report = filter_chain(instruments, marks)
    all_dropped = [name for o in report.outcomes for name in o.dropped]

    assert all_dropped.count(instruments[0].name) == 1
    assert len(all_dropped) == len(set(all_dropped))


def test_kept_and_dropped_reconstruct_the_original_set(venue: FakeVenue) -> None:
    instruments = _instruments(venue)
    marks = _full_marks(instruments)
    marks[instruments[0].name] = -1.0
    del marks[instruments[1].name]

    report = filter_chain(instruments, marks)
    reconstructed = {i.name for i in report.kept} | {
        name for o in report.outcomes for name in o.dropped
    }

    assert reconstructed == {i.name for i in instruments}
    assert len(report.kept) + sum(o.dropped_count for o in report.outcomes) == len(instruments)


def test_crossed_market_rule_only_evaluated_with_quotes(venue: FakeVenue) -> None:
    instruments = _instruments(venue)
    marks = _full_marks(instruments)
    crossed_inst = instruments[0]
    quotes = {crossed_inst.name: Quote(bid=0.06, ask=0.05, mark=0.05)}

    with_quotes = filter_chain(instruments, marks, quotes)
    crossed = next(o for o in with_quotes.outcomes if o.name == "crossed market")
    assert crossed.dropped == (crossed_inst.name,)

    without_quotes = filter_chain(instruments, marks)
    crossed_absent = next(o for o in without_quotes.outcomes if o.name == "crossed market")
    assert crossed_absent.dropped_count == 0


def test_crossed_market_rule_ignores_one_sided_quotes(venue: FakeVenue) -> None:
    instruments = _instruments(venue)
    marks = _full_marks(instruments)
    quotes = {instruments[0].name: Quote(bid=None, ask=0.05, mark=0.05)}

    report = filter_chain(instruments, marks, quotes)
    crossed = next(o for o in report.outcomes if o.name == "crossed market")
    assert crossed.dropped == ()


def test_dropped_fraction_is_against_the_original_total(venue: FakeVenue) -> None:
    instruments = _instruments(venue)
    marks = _full_marks(instruments)
    total = len(instruments)
    del marks[instruments[0].name]
    marks[instruments[1].name] = -1.0

    report = filter_chain(instruments, marks)
    non_positive = next(o for o in report.outcomes if o.name == "non-positive mark")

    assert non_positive.total == total
    assert non_positive.dropped_fraction == 1 / total
