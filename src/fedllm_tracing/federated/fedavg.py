"""Federated averaging over adapter-only client updates."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from fedllm_tracing.adapters.update_ops import weighted_sum_states
from fedllm_tracing.federated.types import ClientUpdate


def fedavg(
    updates: Sequence[ClientUpdate], *, weighted: bool = True
) -> dict[str, torch.Tensor]:
    """Aggregate compatible updates with example-weighted or uniform averaging."""
    if not updates:
        raise ValueError("FedAvg requires at least one client update")
    round_ids = {update.round_id for update in updates}
    if len(round_ids) != 1:
        raise ValueError("FedAvg updates must belong to the same round")

    if weighted:
        if any(update.num_examples is None for update in updates):
            raise ValueError("Weighted FedAvg requires num_examples on every update")
        counts = [int(update.num_examples) for update in updates]  # type: ignore[arg-type]
        total = sum(counts)
        weights = [count / total for count in counts]
    else:
        weights = [1.0 / len(updates)] * len(updates)

    return weighted_sum_states(
        (update.state_dict for update in updates), weights
    )
