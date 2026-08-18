"""Architecture-neutral contract for a future ProToken implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from fedllm_tracing.federated.types import ClientUpdate


@dataclass(frozen=True)
class TokenAttribution:
    """Per-client attribution for one token in its autoregressive context."""

    client_id: int
    token_id: int
    layer_scores: Mapping[str, float]

    @property
    def score(self) -> float:
        return float(sum(self.layer_scores.values()))


class TokenTracer(ABC):
    """ProToken-style tracing contract.

    Implementations must capture ``h_G^(l-1)`` from the global forward pass and
    feed that exact tensor through the client-updated layer. A separate full
    client forward pass followed by hidden-state similarity is not valid.
    """

    @abstractmethod
    def capture_global_states(
        self, global_model: Any, input_ids: torch.Tensor, layers: Sequence[str]
    ) -> Mapping[str, torch.Tensor]:
        """Capture each selected global layer's exact input tensor."""

    @abstractmethod
    def compute_client_layer_activation(
        self,
        global_model: Any,
        client_update: ClientUpdate,
        layer: str,
        global_layer_input: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a client-updated layer to the captured global layer input."""

    @abstractmethod
    def compute_token_gradient(
        self,
        global_model: Any,
        target_token_id: int,
        layer_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the target logit's gradient with respect to layer output."""

    @abstractmethod
    def compute_attribution_score(
        self, client_activation: torch.Tensor, token_gradient: torch.Tensor
    ) -> float:
        """Compute the ProToken inner product for one client/layer/token."""

    @abstractmethod
    def trace(
        self,
        global_model: Any,
        client_updates: Sequence[ClientUpdate],
        prompt: str,
        target_tokens: Sequence[int],
        layers: Sequence[str],
        *,
        teacher_forcing: bool = True,
    ) -> Sequence[TokenAttribution]:
        """Trace every target token under its proper autoregressive context."""

