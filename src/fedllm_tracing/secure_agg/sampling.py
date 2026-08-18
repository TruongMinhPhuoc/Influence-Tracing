"""Symmetric paired-subset sampling for FedAttr-style estimation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PairedSubset:
    """One target-present and one target-absent Secure Aggregation query."""

    included: tuple[int, ...]
    excluded: tuple[int, ...]


def sample_paired_subsets(
    *,
    client_ids: Sequence[int],
    target_id: int,
    num_pairs: int,
    subset_size: int,
    seed: int,
) -> tuple[PairedSubset, ...]:
    """Sample independent, distribution-matched masking sets for each arm.

    ``subset_size`` is the number of non-target masking clients in both arms.
    Consequently, the included aggregate has size ``subset_size + 1`` and the
    excluded aggregate has size ``subset_size``.
    """
    unique_ids = tuple(sorted(set(client_ids)))
    if len(unique_ids) != len(client_ids):
        raise ValueError("client_ids must be unique")
    if target_id not in unique_ids:
        raise KeyError(f"Unknown target client: {target_id}")
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    others = [client_id for client_id in unique_ids if client_id != target_id]
    if not 0 < subset_size <= len(others):
        raise ValueError(
            "subset_size must be positive and no larger than K - 1"
        )

    rng = random.Random(seed)
    pairs: list[PairedSubset] = []
    for _ in range(num_pairs):
        included_mask = tuple(sorted(rng.sample(others, subset_size)))
        excluded_mask = tuple(sorted(rng.sample(others, subset_size)))
        pairs.append(
            PairedSubset(
                included=tuple(sorted((target_id, *included_mask))),
                excluded=excluded_mask,
            )
        )
    return tuple(pairs)

