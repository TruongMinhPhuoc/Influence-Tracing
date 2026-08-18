"""CPU-only mock federated learning utilities."""

from __future__ import annotations

import torch

from fedllm_tracing.federated.fedavg import fedavg
from fedllm_tracing.federated.types import ClientUpdate, FederatedRound


MOCK_LORA_KEY = "model.layers.0.self_attn.q_proj.lora_A"


def generate_mock_updates(
    *,
    num_clients: int,
    dimension: int,
    round_id: int = 0,
    seed: int = 42,
    update_scale: float = 1.0,
) -> tuple[ClientUpdate, ...]:
    """Generate reproducible vector updates inside a LoRA-like state dict."""
    if num_clients < 2:
        raise ValueError("At least two clients are required")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if update_scale < 0:
        raise ValueError("update_scale cannot be negative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(
        ClientUpdate(
            client_id=client_id,
            round_id=round_id,
            state_dict={
                MOCK_LORA_KEY: torch.randn(
                    dimension, generator=generator, dtype=torch.float32
                )
                * update_scale
            },
            num_examples=100,
        )
        for client_id in range(num_clients)
    )


def simulate_round(
    updates: tuple[ClientUpdate, ...], *, weighted: bool = False
) -> FederatedRound:
    if not updates:
        raise ValueError("Cannot simulate an empty round")
    aggregate = fedavg(updates, weighted=weighted)
    return FederatedRound(
        round_id=updates[0].round_id,
        client_updates=updates,
        aggregate=aggregate,
    )

