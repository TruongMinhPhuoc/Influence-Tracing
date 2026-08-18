"""FedAttr-style client update estimation from aggregate subset queries."""

from __future__ import annotations

from dataclasses import dataclass

from fedllm_tracing.adapters.update_ops import (
    scale_state,
    subtract_states,
    weighted_sum_states,
)
from fedllm_tracing.federated.types import ClientUpdate
from fedllm_tracing.secure_agg.sampling import PairedSubset, sample_paired_subsets
from fedllm_tracing.secure_agg.subset_query import SecureAggregateOracle


@dataclass(frozen=True)
class FedAttrEstimate:
    """An estimated client update plus auditable query metadata."""

    update: ClientUpdate
    query_pairs: tuple[PairedSubset, ...]

    @property
    def num_pairs(self) -> int:
        return len(self.query_pairs)

    @property
    def secure_query_count(self) -> int:
        return 2 * self.num_pairs


def estimate_client_update(
    *,
    target_id: int,
    oracle: SecureAggregateOracle,
    num_queries: int,
    subset_size: int,
    seed: int = 42,
) -> FedAttrEstimate:
    """Estimate one update by averaging paired aggregate differences.

    ``num_queries`` follows the research notation ``M`` and counts query pairs;
    the oracle is therefore called exactly ``2 * num_queries`` times.
    """
    pairs = sample_paired_subsets(
        client_ids=oracle.client_ids,
        target_id=target_id,
        num_pairs=num_queries,
        subset_size=subset_size,
        seed=seed,
    )
    differences = []
    for pair in pairs:
        target_present = oracle.query(pair.included)
        target_absent = oracle.query(pair.excluded)
        differences.append(subtract_states(target_present, target_absent))

    summed = weighted_sum_states(differences, [1.0] * len(differences))
    estimated_state = scale_state(summed, 1.0 / len(differences))
    return FedAttrEstimate(
        update=ClientUpdate(
            client_id=target_id,
            round_id=oracle.round_id,
            state_dict=estimated_state,
        ),
        query_pairs=pairs,
    )

