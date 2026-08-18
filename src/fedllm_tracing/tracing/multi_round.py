"""Explicit baseline aggregation modes for historical provenance scores."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum


class AggregationMode(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    LAST = "last"
    WEIGHTED_MEAN = "weighted_mean"


def aggregate_history(
    round_scores: Mapping[int, float],
    *,
    mode: AggregationMode | str,
    weights: Mapping[int, float] | None = None,
) -> float:
    """Aggregate per-round provenance without implying causal influence."""
    if not round_scores:
        raise ValueError("At least one round score is required")
    selected_mode = AggregationMode(mode)
    ordered = sorted(round_scores.items())
    values = [float(score) for _, score in ordered]
    if selected_mode is AggregationMode.SUM:
        return sum(values)
    if selected_mode is AggregationMode.MEAN:
        return sum(values) / len(values)
    if selected_mode is AggregationMode.LAST:
        return values[-1]
    if weights is None or set(weights) != set(round_scores):
        raise ValueError("weighted_mean requires one weight for every round")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Round weights cannot be negative")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Round weights must have a positive sum")
    return sum(score * weights[round_id] for round_id, score in ordered) / total_weight

