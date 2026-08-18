from __future__ import annotations

import pytest
import torch

from fedllm_tracing.adapters.update_ops import (
    add_states,
    relative_state_error,
    state_l2_norm,
)
from fedllm_tracing.federated.fedavg import fedavg
from fedllm_tracing.federated.types import ClientUpdate


def _update(client_id: int, value: float, examples: int | None = 1) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        round_id=0,
        state_dict={"layer.lora_A": torch.tensor([value, 2 * value])},
        num_examples=examples,
    )


def test_unweighted_fedavg_matches_hand_calculation() -> None:
    result = fedavg([_update(0, 1.0), _update(1, 3.0)], weighted=False)
    torch.testing.assert_close(result["layer.lora_A"], torch.tensor([2.0, 4.0]))


def test_weighted_fedavg_matches_hand_calculation() -> None:
    result = fedavg(
        [_update(0, 1.0, examples=1), _update(1, 3.0, examples=3)],
        weighted=True,
    )
    torch.testing.assert_close(result["layer.lora_A"], torch.tensor([2.5, 5.0]))


def test_weighted_fedavg_requires_example_counts() -> None:
    with pytest.raises(ValueError, match="num_examples"):
        fedavg([_update(0, 1.0, examples=None)], weighted=True)


def test_incompatible_updates_are_rejected() -> None:
    first = _update(0, 1.0)
    second = ClientUpdate(1, 0, {"other.lora_A": torch.ones(2)}, 1)
    with pytest.raises(ValueError, match="keys differ"):
        fedavg([first, second], weighted=False)


def test_update_operations_do_not_mutate_inputs() -> None:
    first = _update(0, 1.0)
    second = _update(1, 2.0)
    snapshot = first.state_dict["layer.lora_A"].clone()
    result = add_states(first.state_dict, second.state_dict)
    torch.testing.assert_close(first.state_dict["layer.lora_A"], snapshot)
    assert result["layer.lora_A"].data_ptr() != first.state_dict["layer.lora_A"].data_ptr()
    assert state_l2_norm(first.state_dict) == pytest.approx(5**0.5)
    assert relative_state_error(first.state_dict, first.state_dict) == 0.0


def test_client_update_validates_metadata_and_tensors() -> None:
    with pytest.raises(ValueError, match="round_id"):
        ClientUpdate(0, -1, {"layer.lora_A": torch.ones(1)})
    with pytest.raises(TypeError, match="floating point"):
        ClientUpdate(0, 0, {"layer.lora_A": torch.ones(1, dtype=torch.int64)})

