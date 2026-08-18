from __future__ import annotations

import pytest
import torch

from fedllm_tracing.adapters.update_ops import relative_state_error
from fedllm_tracing.evaluation.noise import perturb_update
from fedllm_tracing.federated.types import ClientUpdate


@pytest.mark.parametrize("level", [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
def test_noise_has_requested_relative_norm(level: float) -> None:
    update = ClientUpdate(
        2,
        1,
        {
            "a.lora_A": torch.linspace(-1, 1, 50),
            "a.lora_B": torch.linspace(1, 2, 30),
        },
        10,
    )
    perturbed = perturb_update(update, level, seed=123)
    assert relative_state_error(perturbed.state_dict, update.state_dict) == pytest.approx(
        level, rel=2e-6, abs=1e-7
    )
    assert perturbed.client_id == update.client_id
    assert perturbed.num_examples == update.num_examples


def test_noise_is_deterministic_and_does_not_mutate_input() -> None:
    update = ClientUpdate(0, 0, {"lora_A": torch.ones(10)})
    snapshot = update.state_dict["lora_A"].clone()
    first = perturb_update(update, 0.2, seed=8)
    second = perturb_update(update, 0.2, seed=8)
    torch.testing.assert_close(first.state_dict["lora_A"], second.state_dict["lora_A"])
    torch.testing.assert_close(update.state_dict["lora_A"], snapshot)


def test_nonzero_relative_noise_rejects_zero_update() -> None:
    update = ClientUpdate(0, 0, {"lora_A": torch.zeros(4)})
    with pytest.raises(ValueError, match="zero-norm"):
        perturb_update(update, 0.1)

