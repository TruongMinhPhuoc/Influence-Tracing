from __future__ import annotations

from collections import Counter

import torch

from fedllm_tracing.adapters.update_ops import relative_state_error
from fedllm_tracing.federated.simulator import generate_mock_updates
from fedllm_tracing.secure_agg.fedattr import estimate_client_update
from fedllm_tracing.secure_agg.sampling import sample_paired_subsets
from fedllm_tracing.secure_agg.subset_query import SecureAggregateOracle


def test_sampling_invariants_and_determinism() -> None:
    arguments = dict(
        client_ids=list(range(8)),
        target_id=3,
        num_pairs=50,
        subset_size=3,
        seed=7,
    )
    pairs = sample_paired_subsets(**arguments)
    assert pairs == sample_paired_subsets(**arguments)
    for pair in pairs:
        assert 3 in pair.included
        assert 3 not in pair.excluded
        assert len(pair.included) == 4
        assert len(pair.excluded) == 3
        assert len(set(pair.included)) == len(pair.included)
        assert len(set(pair.excluded)) == len(pair.excluded)


def test_sampling_is_symmetric_for_non_target_clients() -> None:
    pairs = sample_paired_subsets(
        client_ids=list(range(6)),
        target_id=0,
        num_pairs=10_000,
        subset_size=2,
        seed=11,
    )
    included_counts = Counter(
        client_id
        for pair in pairs
        for client_id in pair.included
        if client_id != 0
    )
    excluded_counts = Counter(client_id for pair in pairs for client_id in pair.excluded)
    for client_id in range(1, 6):
        difference = abs(included_counts[client_id] - excluded_counts[client_id])
        assert difference < 350


def test_fedattr_shape_query_count_and_determinism() -> None:
    updates = generate_mock_updates(num_clients=6, dimension=32, seed=3)
    first_oracle = SecureAggregateOracle(updates)
    first = estimate_client_update(
        target_id=2,
        oracle=first_oracle,
        num_queries=7,
        subset_size=2,
        seed=99,
    )
    second = estimate_client_update(
        target_id=2,
        oracle=SecureAggregateOracle(updates),
        num_queries=7,
        subset_size=2,
        seed=99,
    )
    assert first_oracle.query_count == 14
    assert first.secure_query_count == 14
    assert first.update.state_dict.keys() == updates[2].state_dict.keys()
    for key in first.update.state_dict:
        assert first.update.state_dict[key].shape == updates[2].state_dict[key].shape
        torch.testing.assert_close(first.update.state_dict[key], second.update.state_dict[key])


def test_empirical_fedattr_mean_approaches_target_update() -> None:
    updates = generate_mock_updates(num_clients=5, dimension=24, seed=5)
    target = updates[1]
    estimates = []
    for seed in range(600):
        result = estimate_client_update(
            target_id=1,
            oracle=SecureAggregateOracle(updates),
            num_queries=2,
            subset_size=2,
            seed=seed,
        )
        estimates.append(result.update.state_dict)
    key = next(iter(target.state_dict))
    empirical_mean = torch.stack([state[key] for state in estimates]).mean(dim=0)
    assert relative_state_error({key: empirical_mean}, target.state_dict) < 0.12


def test_more_queries_reduce_average_estimation_error() -> None:
    updates = generate_mock_updates(num_clients=7, dimension=128, seed=17)
    target = updates[0]

    def average_error(num_queries: int) -> float:
        values = []
        for seed in range(30):
            estimate = estimate_client_update(
                target_id=0,
                oracle=SecureAggregateOracle(updates),
                num_queries=num_queries,
                subset_size=3,
                seed=seed,
            )
            values.append(relative_state_error(estimate.update.state_dict, target.state_dict))
        return sum(values) / len(values)

    assert average_error(40) < average_error(2)

