from __future__ import annotations

import pytest
import torch

from fedllm_tracing.federated.types import ClientUpdate
from fedllm_tracing.secure_agg.subset_query import SecureAggregateOracle


def _updates() -> tuple[ClientUpdate, ...]:
    return tuple(
        ClientUpdate(
            client_id=client_id,
            round_id=1,
            state_dict={"lora_A": torch.tensor([float(client_id + 1)])},
        )
        for client_id in range(4)
    )


def test_subset_sum_is_exact() -> None:
    oracle = SecureAggregateOracle(_updates())
    result = oracle.query([0, 2, 3])
    torch.testing.assert_close(result["lora_A"], torch.tensor([8.0]))
    assert oracle.query_count == 1


@pytest.mark.parametrize("subset", [[], [0, 0]])
def test_invalid_subset_is_rejected(subset: list[int]) -> None:
    oracle = SecureAggregateOracle(_updates())
    with pytest.raises(ValueError):
        oracle.query(subset)


def test_unknown_client_is_rejected() -> None:
    with pytest.raises(KeyError, match="99"):
        SecureAggregateOracle(_updates()).query([99])

