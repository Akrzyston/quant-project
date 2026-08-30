"""Instrument metadata tests."""

from __future__ import annotations

import pickle
from datetime import UTC, datetime, timedelta

import pytest

from tests.conftest import AS_OF, BASE, option_payload
from voltk.instruments import Instrument, Kind, MetadataError, Settlement

EXPIRY = datetime(2026, 12, 25, 8, 0, tzinfo=UTC)


def _payload(**overrides):
    return option_payload(BASE, EXPIRY, 60000.0, True) | overrides


def test_survives_pickle_round_trip() -> None:
    inst = Instrument.from_deribit(_payload())
    assert pickle.loads(pickle.dumps(inst)) == inst


def test_inverse_settlement_from_instrument_type() -> None:
    assert Instrument.from_deribit(_payload()).settlement is Settlement.INVERSE


def test_linear_settlement_from_instrument_type() -> None:
    inst = Instrument.from_deribit(_payload(instrument_type="linear", settlement_currency="USDC"))
    assert inst.settlement is Settlement.LINEAR
    assert not inst.is_inverse


def test_settlement_falls_back_to_currency_comparison() -> None:
    payload = _payload()
    payload.pop("instrument_type")
    assert Instrument.from_deribit(payload).is_inverse


@pytest.mark.parametrize("field", ["contract_size", "instrument_name", "settlement_currency"])
def test_missing_required_field_raises_rather_than_defaulting(field) -> None:
    payload = _payload()
    payload.pop(field)
    with pytest.raises(MetadataError, match=field):
        Instrument.from_deribit(payload)


def test_tau_requires_an_explicit_observation_time() -> None:
    inst = Instrument.from_deribit(_payload())
    with pytest.raises(TypeError):
        inst.tau()


def test_tau_is_measured_from_the_given_time() -> None:
    inst = Instrument.from_deribit(_payload())
    a = inst.tau(AS_OF)
    b = inst.tau(AS_OF + timedelta(days=1))
    assert a > b


def test_perpetual_has_no_expiry() -> None:
    inst = Instrument.from_deribit(
        _payload(kind=str(Kind.FUTURE), expiration_timestamp=32503708800000, strike=None,
                 option_type=None)
    )
    assert inst.expiry is None
    assert inst.tau(AS_OF) is None
    assert inst.expiry_label == "PERPETUAL"
