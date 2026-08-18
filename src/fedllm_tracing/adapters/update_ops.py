"""Pure tensor-state operations shared by the research pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from fedllm_tracing.federated.types import TensorState


def assert_compatible(states: Iterable[TensorState]) -> tuple[TensorState, ...]:
    """Validate identical keys, shapes, dtypes, and devices."""
    materialized = tuple(states)
    if not materialized:
        raise ValueError("At least one tensor state is required")
    reference = materialized[0]
    reference_keys = set(reference)
    for state_index, state in enumerate(materialized[1:], start=1):
        if set(state) != reference_keys:
            raise ValueError(f"Tensor-state keys differ at index {state_index}")
        for key, reference_tensor in reference.items():
            tensor = state[key]
            if tensor.shape != reference_tensor.shape:
                raise ValueError(f"Shape mismatch for {key!r} at index {state_index}")
            if tensor.dtype != reference_tensor.dtype:
                raise ValueError(f"Dtype mismatch for {key!r} at index {state_index}")
            if tensor.device != reference_tensor.device:
                raise ValueError(f"Device mismatch for {key!r} at index {state_index}")
    return materialized


def clone_state(state: TensorState) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().clone() for key, tensor in state.items()}


def zeros_like_state(state: TensorState) -> dict[str, torch.Tensor]:
    return {key: torch.zeros_like(tensor) for key, tensor in state.items()}


def scale_state(state: TensorState, factor: float) -> dict[str, torch.Tensor]:
    return {key: tensor * factor for key, tensor in state.items()}


def add_states(left: TensorState, right: TensorState) -> dict[str, torch.Tensor]:
    assert_compatible((left, right))
    return {key: left[key] + right[key] for key in left}


def subtract_states(left: TensorState, right: TensorState) -> dict[str, torch.Tensor]:
    assert_compatible((left, right))
    return {key: left[key] - right[key] for key in left}


def weighted_sum_states(
    states: Iterable[TensorState], weights: Iterable[float]
) -> dict[str, torch.Tensor]:
    materialized = assert_compatible(states)
    materialized_weights = tuple(float(weight) for weight in weights)
    if len(materialized) != len(materialized_weights):
        raise ValueError("Number of weights must match number of states")
    result = zeros_like_state(materialized[0])
    for state, weight in zip(materialized, materialized_weights, strict=True):
        for key in result:
            result[key].add_(state[key], alpha=weight)
    return result


def state_l2_norm(state: TensorState) -> float:
    squared_norm = sum(
        torch.sum(tensor.detach().to(dtype=torch.float64) ** 2).item()
        for tensor in state.values()
    )
    return float(squared_norm**0.5)


def relative_state_error(estimate: TensorState, target: TensorState) -> float:
    difference = subtract_states(estimate, target)
    denominator = state_l2_norm(target)
    numerator = state_l2_norm(difference)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def state_to_cpu_contiguous(state: TensorState) -> dict[str, torch.Tensor]:
    return {
        key: tensor.detach().to(device="cpu").contiguous()
        for key, tensor in state.items()
    }


def validate_state_mapping(state: Mapping[str, torch.Tensor]) -> None:
    """Validate a standalone mapping by constructing no additional state."""
    if not state:
        raise ValueError("Tensor state cannot be empty")
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("Tensor state must map string keys to tensors")

