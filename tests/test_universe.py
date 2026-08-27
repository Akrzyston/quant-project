"""Universe specification tests."""

from __future__ import annotations

import pytest

from tests.conftest import ALT, BASE
from voltk.instruments import Kind
from voltk.universe import UniverseError, UniverseSpec


def test_single_currency_is_not_cross_currency() -> None:
    assert not UniverseSpec.of([BASE]).is_cross_currency


def test_multiple_currencies_are_cross_currency() -> None:
    assert UniverseSpec.of([BASE, ALT]).is_cross_currency


def test_duplicates_are_collapsed_by_of() -> None:
    assert UniverseSpec.of([BASE, BASE]).currencies == (BASE,)


def test_duplicates_are_rejected_by_the_constructor() -> None:
    with pytest.raises(UniverseError, match="Duplicate"):
        UniverseSpec(currencies=(BASE, BASE))


def test_empty_universe_is_rejected() -> None:
    with pytest.raises(UniverseError, match="at least one currency"):
        UniverseSpec(currencies=())


def test_key_is_stable_across_kind_ordering() -> None:
    a = UniverseSpec.of([BASE], kinds=[Kind.OPTION, Kind.FUTURE])
    b = UniverseSpec.of([BASE], kinds=[Kind.FUTURE, Kind.OPTION])
    assert a.key() == b.key()
