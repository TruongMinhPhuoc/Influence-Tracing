"""Architecture-neutral contract for a future ProToken implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from fedllm_tracing.federated.types import ClientUpdate
from fedllm_tracing.tracing.hooks import GlobalForwardTrace, LayerCallContext


@dataclass(frozen=True)
class TokenAttribution:
    """Per-client attribution for one token in its autoregressive context."""

    client_id: int
    token_id: int
    layer_scores: Mapping[str, float]
    token_index: int = 0
    sequence_position: int = 0

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
        self,
        global_model: Any,
        model_inputs: Mapping[str, torch.Tensor],
        layers: Sequence[str],
    ) -> GlobalForwardTrace:
        """Run globally and capture exact selected layer calls and outputs."""

    @abstractmethod
    def compute_client_layer_activation(
        self,
        global_model: Any,
        client_update: ClientUpdate,
        layer: str,
        global_layer_call: LayerCallContext,
    ) -> torch.Tensor:
        """Apply a client-updated layer to the captured global layer input."""

    @abstractmethod
    def compute_token_gradient(
        self,
        target_logit: torch.Tensor,
        layer_output: torch.Tensor,
        *,
        retain_graph: bool = True,
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
