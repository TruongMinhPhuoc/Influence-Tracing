"""Reusable client-attribution fidelity and ranking metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Hashable

import numpy as np
from scipy.stats import rankdata, spearmanr


Scores = Mapping[Hashable, float] | Sequence[float] | np.ndarray


def _aligned_scores(
    oracle: Scores, estimated: Scores
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(oracle, Mapping) != isinstance(estimated, Mapping):
        raise TypeError("Both score inputs must use the same representation")
    if isinstance(oracle, Mapping):
        assert isinstance(estimated, Mapping)
        if set(oracle) != set(estimated):
            raise ValueError("Score mappings must contain identical client IDs")
        keys = sorted(oracle, key=lambda item: (type(item).__name__, repr(item)))
        oracle_array = np.asarray([oracle[key] for key in keys], dtype=float)
        estimated_array = np.asarray([estimated[key] for key in keys], dtype=float)
    else:
        oracle_array = np.asarray(oracle, dtype=float)
        estimated_array = np.asarray(estimated, dtype=float)
    if oracle_array.ndim != 1 or estimated_array.ndim != 1:
        raise ValueError("Scores must be one-dimensional")
    if oracle_array.size == 0 or oracle_array.size != estimated_array.size:
        raise ValueError("Scores must be non-empty and have equal length")
    if not np.all(np.isfinite(oracle_array)) or not np.all(np.isfinite(estimated_array)):
        raise ValueError("Scores must be finite")
    return oracle_array, estimated_array


def attribution_mae(oracle: Scores, estimated: Scores) -> float:
    oracle_array, estimated_array = _aligned_scores(oracle, estimated)
    return float(np.mean(np.abs(estimated_array - oracle_array)))


def spearman_correlation(oracle: Scores, estimated: Scores) -> float:
    """Return Spearman rho with deterministic handling of constant arrays."""
    oracle_array, estimated_array = _aligned_scores(oracle, estimated)
    oracle_constant = np.all(oracle_array == oracle_array[0])
    estimated_constant = np.all(estimated_array == estimated_array[0])
    if oracle_constant or estimated_constant:
        return 1.0 if np.array_equal(oracle_array, estimated_array) else 0.0
    return float(spearmanr(oracle_array, estimated_array).statistic)


def _descending_order(scores: np.ndarray) -> np.ndarray:
    # mergesort preserves canonical client order when scores tie.
    return np.argsort(-scores, kind="mergesort")


def top1_agreement(oracle: Scores, estimated: Scores) -> bool:
    oracle_array, estimated_array = _aligned_scores(oracle, estimated)
    return bool(_descending_order(oracle_array)[0] == _descending_order(estimated_array)[0])


def topk_overlap(oracle: Scores, estimated: Scores, *, k: int) -> float:
    oracle_array, estimated_array = _aligned_scores(oracle, estimated)
    if not 0 < k <= oracle_array.size:
        raise ValueError("k must be between 1 and the number of clients")
    oracle_top = set(_descending_order(oracle_array)[:k].tolist())
    estimated_top = set(_descending_order(estimated_array)[:k].tolist())
    return len(oracle_top & estimated_top) / k


def ndcg(oracle: Scores, estimated: Scores, *, k: int | None = None) -> float:
    """Compute NDCG@k using shifted non-negative oracle scores as relevance."""
    oracle_array, estimated_array = _aligned_scores(oracle, estimated)
    if k is None:
        k = int(oracle_array.size)
    if not 0 < k <= oracle_array.size:
        raise ValueError("k must be between 1 and the number of clients")

    relevance = oracle_array - np.min(oracle_array)
    if np.all(relevance == 0):
        return 1.0

    predicted_order = _descending_order(estimated_array)[:k]
    ideal_order = _descending_order(oracle_array)[:k]
    discounts = np.log2(np.arange(2, k + 2, dtype=float))
    dcg = float(np.sum(relevance[predicted_order] / discounts))
    ideal_dcg = float(np.sum(relevance[ideal_order] / discounts))
    return dcg / ideal_dcg


def client_ranks(scores: Scores) -> np.ndarray:
    """Return descending ranks with average ranks for ties (rank 1 is best)."""
    if isinstance(scores, Mapping):
        keys = sorted(scores, key=lambda item: (type(item).__name__, repr(item)))
        array = np.asarray([scores[key] for key in keys], dtype=float)
    else:
        array = np.asarray(scores, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Scores must be a non-empty one-dimensional array")
    return rankdata(-array, method="average")

