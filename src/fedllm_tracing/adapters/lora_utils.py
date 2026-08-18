"""PEFT LoRA attachment, extraction, loading, and temporary swapping."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from fedllm_tracing.federated.types import ClientUpdate


@dataclass(frozen=True)
class LoraSpec:
    rank: int = 4
    alpha: int = 8
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("At least one LoRA target module is required")

    def to_peft_config(self) -> LoraConfig:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.target_modules),
            bias="none",
        )


def attach_lora(model: torch.nn.Module, spec: LoraSpec) -> torch.nn.Module:
    """Attach one default causal-LM LoRA adapter to a base model."""
    return get_peft_model(model, spec.to_peft_config())


def _validate_lora_state(state: dict[str, torch.Tensor] | Any) -> None:
    if not state:
        raise ValueError("LoRA state cannot be empty")
    invalid = [
        key
        for key in state
        if ".lora_A." not in key
        and ".lora_B." not in key
        and ".lora_embedding_A" not in key
        and ".lora_embedding_B" not in key
    ]
    if invalid:
        raise ValueError(f"State contains non-LoRA parameters: {invalid[:3]}")


def extract_lora_update(
    model: torch.nn.Module,
    *,
    client_id: int,
    round_id: int,
    num_examples: int | None = None,
    adapter_name: str = "default",
) -> ClientUpdate:
    state = get_peft_model_state_dict(
        model, adapter_name=adapter_name, save_embedding_layers=False
    )
    _validate_lora_state(state)
    cloned = {key: value.detach().cpu().clone() for key, value in state.items()}
    return ClientUpdate(client_id, round_id, cloned, num_examples)


def load_lora_update(
    model: torch.nn.Module,
    update: ClientUpdate,
    *,
    adapter_name: str = "default",
) -> None:
    state = dict(update.state_dict)
    _validate_lora_state(state)
    result = set_peft_model_state_dict(model, state, adapter_name=adapter_name)
    unexpected = tuple(getattr(result, "unexpected_keys", ()))
    mismatched = tuple(getattr(result, "mismatched_keys", ()))
    if unexpected or mismatched:
        raise ValueError(
            f"Could not load LoRA update; unexpected={unexpected}, mismatched={mismatched}"
        )


@contextmanager
def use_lora_update(
    model: torch.nn.Module,
    update: ClientUpdate,
    *,
    adapter_name: str = "default",
) -> Iterator[None]:
    """Temporarily replace an adapter and restore it even after an exception."""
    original = {
        key: tensor.detach().cpu().clone()
        for key, tensor in get_peft_model_state_dict(
            model, adapter_name=adapter_name, save_embedding_layers=False
        ).items()
    }
    load_lora_update(model, update, adapter_name=adapter_name)
    try:
        yield
    finally:
        set_peft_model_state_dict(model, original, adapter_name=adapter_name)


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def lora_spec_from_config(config: dict[str, Any]) -> LoraSpec:
    return LoraSpec(
        rank=int(config["rank"]),
        alpha=int(config["alpha"]),
        dropout=float(config.get("dropout", 0.0)),
        target_modules=tuple(config["target_modules"]),
    )
