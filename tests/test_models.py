"""Model registry tests."""

from __future__ import annotations

import pytest

from voltk import models


def test_default_selection_is_the_reference_case() -> None:
    assert models.all_models()[0].key == "black_scholes"


def test_ordering_is_declared_not_alphabetical() -> None:
    orders = [spec.order for spec in models.all_models()]
    assert orders == sorted(orders)
    assert len(orders) == len(set(orders))


def test_every_required_model_is_registered() -> None:
    required = {"black_scholes", "black_76", "bachelier", "binomial_american", "inverse_deribit"}
    assert required <= {spec.key for spec in models.all_models()}


def test_unimplemented_model_refuses_to_build() -> None:
    spec = models.get("inverse_deribit")
    if not spec.implemented:
        with pytest.raises(NotImplementedError, match="M1"):
            spec.build()

