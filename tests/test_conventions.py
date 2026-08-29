"""Venue convention tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.conftest import BASE, option_payload
from voltk import conventions
from voltk.conventions import ConventionError
from voltk.instruments import Instrument

VENUE = "deribit"


def test_spec_loads_with_provenance() -> None:
    spec = conventions.load(VENUE)
    assert spec.source_url.startswith("https://")
    assert spec.retrieved_at
    assert spec.schema_version >= 1


def test_expiry_time_comes_from_the_spec_file() -> None:
    spec = conventions.load(VENUE)
    assert (spec.expiry_time_utc.hour, spec.expiry_time_utc.minute) == (8, 0)


def test_conforming_expiries_validate() -> None:
    spec = conventions.load(VENUE)
    expiry = datetime(2026, 12, 25, 8, 0, tzinfo=UTC)
    instruments = [Instrument.from_deribit(option_payload(BASE, expiry, 60000.0, True))]
    conventions.validate_expiry_convention(instruments, spec)


def test_off_convention_expiry_is_caught() -> None:
    spec = conventions.load(VENUE)
    expiry = datetime(2026, 12, 25, 14, 30, tzinfo=UTC)
    instruments = [Instrument.from_deribit(option_payload(BASE, expiry, 60000.0, True))]
    with pytest.raises(ConventionError, match="off the documented"):
        conventions.validate_expiry_convention(instruments, spec)


def test_option_fee_switches_between_rate_and_cap() -> None:
    """Charged on the underlying, but capped as a fraction of a small premium."""
    schedule = conventions.load(VENUE).fee_schedule("option")
    underlying = 60000.0

    cheap = schedule.fee(underlying, 10.0, maker=False)
    assert cheap == pytest.approx(schedule.cap_fraction_of_premium * 10.0)
    assert cheap < schedule.taker_rate * underlying

    expensive = schedule.fee(underlying, 5000.0, maker=False)
    assert expensive == pytest.approx(schedule.taker_rate * underlying)


def test_future_fee_has_no_premium_cap() -> None:
    schedule = conventions.load(VENUE).fee_schedule("future")
    assert schedule.cap_fraction_of_premium is None
    assert schedule.fee(60000.0, 1.0, maker=False) == pytest.approx(
        schedule.taker_rate * 60000.0
    )


def test_unknown_venue_is_refused() -> None:
    with pytest.raises(ConventionError, match="No spec file"):
        conventions.load("nonexistent_venue")
