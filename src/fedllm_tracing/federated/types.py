"""Shared update types for aggregation, estimation, and tracing."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch


TensorState = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class ClientUpdate:
    """A client's trainable adapter update for one federated round.

    The mapping is read-only and must contain only floating-point tensors. Tensor
    contents should be treated as immutable by callers; all toolkit operations
    allocate new tensors.
    """

    client_id: int
    round_id: int
    state_dict: TensorState
    num_examples: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.client_id, int):
            raise TypeError("client_id must be an int")
        if not isinstance(self.round_id, int) or self.round_id < 0:
            raise ValueError("round_id must be a non-negative int")
        if self.num_examples is not None and self.num_examples <= 0:
            raise ValueError("num_examples must be positive when provided")
        if not self.state_dict:
            raise ValueError("state_dict must contain at least one tensor")

        copied: dict[str, torch.Tensor] = {}
        for key, tensor in self.state_dict.items():
            if not isinstance(key, str) or not key:
                raise ValueError("state_dict keys must be non-empty strings")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor")
            if not tensor.is_floating_point():
                raise TypeError(f"state_dict[{key!r}] must be floating point")
            copied[key] = tensor
        object.__setattr__(self, "state_dict", MappingProxyType(copied))


@dataclass(frozen=True)
class FederatedRound:
    """The inputs and aggregate produced by one mock FL round."""

    round_id: int
    client_updates: tuple[ClientUpdate, ...]
    aggregate: TensorState

