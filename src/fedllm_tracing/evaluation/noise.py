"""Controlled update perturbations for attribution-sensitivity experiments."""

from __future__ import annotations

import torch

from fedllm_tracing.adapters.update_ops import state_l2_norm
from fedllm_tracing.federated.types import ClientUpdate


def perturb_update(
    update: ClientUpdate, relative_noise: float, *, seed: int = 42
) -> ClientUpdate:
    """Add Gaussian noise normalized to a requested global relative L2 norm."""
    if relative_noise < 0:
        raise ValueError("relative_noise cannot be negative")
    target_norm = state_l2_norm(update.state_dict)
    if target_norm == 0.0 and relative_noise > 0.0:
        raise ValueError("Cannot define relative noise for a zero-norm update")
    if relative_noise == 0.0:
        state = {
            key: tensor.detach().clone() for key, tensor in update.state_dict.items()
        }
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        raw_noise = {
            key: torch.randn(
                tensor.shape,
                generator=generator,
                dtype=tensor.dtype,
                device="cpu",
            ).to(tensor.device)
            for key, tensor in update.state_dict.items()
        }
        noise_norm = state_l2_norm(raw_noise)
        factor = relative_noise * target_norm / noise_norm
        state = {
            key: tensor + raw_noise[key] * factor
            for key, tensor in update.state_dict.items()
        }
    return ClientUpdate(
        client_id=update.client_id,
        round_id=update.round_id,
        state_dict=state,
        num_examples=update.num_examples,
    )

