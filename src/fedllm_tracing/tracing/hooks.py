"""Minimal reusable layer-input capture hook for future model integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class LayerCallContext:
    """The exact positional/keyword arguments and tensor output of one layer call."""

    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    output: torch.Tensor

    @property
    def hidden_states(self) -> torch.Tensor:
        if not self.args or not isinstance(self.args[0], torch.Tensor):
            raise RuntimeError("Captured layer call has no tensor hidden-state input")
        return self.args[0]


@dataclass(frozen=True)
class GlobalForwardTrace:
    """Global model output paired with selected layer call contexts."""

    model_output: Any
    layers: Mapping[str, LayerCallContext]


@dataclass
class LayerInputCapture:
    """Capture the first positional input received by a PyTorch module."""

    detach: bool = False
    value: torch.Tensor | None = field(default=None, init=False)
    _handle: Any = field(default=None, init=False, repr=False)

    def _hook(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("Selected layer did not receive a tensor first input")
        self.value = inputs[0].detach() if self.detach else inputs[0]

    def attach(self, module: torch.nn.Module) -> "LayerInputCapture":
        if self._handle is not None:
            raise RuntimeError("Capture hook is already attached")
        self._handle = module.register_forward_pre_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "LayerInputCapture":
        if self._handle is None:
            raise RuntimeError("Call attach(module) before entering the context")
        return self

    def __exit__(self, *_args: object) -> None:
        self.remove()
