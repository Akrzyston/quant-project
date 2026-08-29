"""Universe specification."""

from __future__ import annotations

import pytest

from tests.conftest import ALT, BASE
from voltk.instruments import Kind
from voltk.universe import UniverseError, UniverseSpec


def test_cross_currency_is_detected() -> None:
    assert not UniverseSpec.of([BASE]).is_cross_currency
    assert UniverseSpec.of([BASE, ALT]).is_cross_currency


def test_duplicate_currencies_are_rejected() -> None:
    assert UniverseSpec.of([BASE, BASE]).currencies == (BASE,)
    with pytest.raises(UniverseError, match="Duplicate"):
        UniverseSpec(currencies=(BASE, BASE))


def test_key_is_stable_across_kind_ordering() -> None:
    """The key lands in snapshot ids, so it must not depend on argument order."""
    a = UniverseSpec.of([BASE], kinds=[Kind.OPTION, Kind.FUTURE])
    b = UniverseSpec.of([BASE], kinds=[Kind.FUTURE, Kind.OPTION])
    assert a.key() == b.key()
