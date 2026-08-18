"""Small explicit local adapter training loop usable on CPU and GPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class LocalTrainingResult:
    losses: tuple[float, ...]

    @property
    def initial_loss(self) -> float:
        return self.losses[0]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]


def train_local_adapter(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    learning_rate: float = 5e-4,
    epochs: int = 1,
    max_steps: int | None = None,
    device: str | torch.device = "cpu",
) -> LocalTrainingResult:
    """Train only parameters already marked trainable (normally LoRA weights)."""
    if not batches:
        raise ValueError("At least one local batch is required")
    if learning_rate <= 0 or epochs <= 0:
        raise ValueError("learning_rate and epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Model has no trainable parameters")

    resolved_device = torch.device(device)
    model.to(resolved_device)
    was_training = model.training
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    losses: list[float] = []
    step = 0
    try:
        for _ in range(epochs):
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                moved = {key: value.to(resolved_device) for key, value in batch.items()}
                output = model(**moved, use_cache=False)
                if output.loss is None or not torch.isfinite(output.loss):
                    raise RuntimeError("Local training produced a non-finite or missing loss")
                output.loss.backward()
                optimizer.step()
                losses.append(float(output.loss.detach().cpu()))
                step += 1
                if max_steps is not None and step >= max_steps:
                    return LocalTrainingResult(tuple(losses))
    finally:
        model.train(was_training)
    return LocalTrainingResult(tuple(losses))

