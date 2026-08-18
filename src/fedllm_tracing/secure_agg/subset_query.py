"""Simulation of the subset-sum view exposed by Secure Aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from fedllm_tracing.adapters.update_ops import weighted_sum_states
from fedllm_tracing.federated.types import ClientUpdate


class SecureAggregateOracle:
    """Expose only aggregate sums for requested subsets of client IDs."""

    def __init__(self, updates: Sequence[ClientUpdate]) -> None:
        if not updates:
            raise ValueError("Secure aggregation requires at least one update")
        indexed = {update.client_id: update for update in updates}
        if len(indexed) != len(updates):
            raise ValueError("Client IDs must be unique")
        if len({update.round_id for update in updates}) != 1:
            raise ValueError("All secure-aggregation updates must share a round")
        # Compatibility is validated once without exposing updates to estimators.
        weighted_sum_states(
            (update.state_dict for update in updates), [0.0] * len(updates)
        )
        self._updates = indexed
        self._query_count = 0

    @property
    def client_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._updates))

    @property
    def round_id(self) -> int:
        return next(iter(self._updates.values())).round_id

    @property
    def query_count(self) -> int:
        return self._query_count

    def query(self, subset: Iterable[int]) -> dict[str, torch.Tensor]:
        """Return the exact sum for a non-empty subset without revealing members."""
        members = tuple(subset)
        if not members:
            raise ValueError("Secure-aggregation subset cannot be empty")
        if len(set(members)) != len(members):
            raise ValueError("Secure-aggregation subset contains duplicate IDs")
        unknown = set(members) - set(self._updates)
        if unknown:
            raise KeyError(f"Unknown client IDs: {sorted(unknown)}")
        self._query_count += 1
        return weighted_sum_states(
            (self._updates[client_id].state_dict for client_id in members),
            [1.0] * len(members),
        )


def secure_aggregate(
    updates: Sequence[ClientUpdate], subset: Iterable[int]
) -> dict[str, torch.Tensor]:
    """Convenience wrapper for one simulated secure subset-sum query."""
    return SecureAggregateOracle(updates).query(subset)
