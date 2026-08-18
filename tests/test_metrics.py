from __future__ import annotations

import pytest

from fedllm_tracing.evaluation.metrics import (
    attribution_mae,
    ndcg,
    spearman_correlation,
    top1_agreement,
    topk_overlap,
)


def test_perfect_ranking_metrics() -> None:
    oracle = {2: 0.2, 0: 1.0, 1: 0.5}
    estimated = {1: 0.5, 2: 0.2, 0: 1.0}
    assert attribution_mae(oracle, estimated) == 0.0
    assert spearman_correlation(oracle, estimated) == pytest.approx(1.0)
    assert top1_agreement(oracle, estimated)
    assert topk_overlap(oracle, estimated, k=2) == 1.0
    assert ndcg(oracle, estimated) == pytest.approx(1.0)


def test_reversed_ranking_metrics() -> None:
    oracle = [3.0, 2.0, 1.0]
    estimated = [1.0, 2.0, 3.0]
    assert attribution_mae(oracle, estimated) == pytest.approx(4 / 3)
    assert spearman_correlation(oracle, estimated) == pytest.approx(-1.0)
    assert not top1_agreement(oracle, estimated)
    assert topk_overlap(oracle, estimated, k=1) == 0.0
    assert 0.0 <= ndcg(oracle, estimated) < 1.0


def test_constant_and_tie_behavior_is_deterministic() -> None:
    assert spearman_correlation([1, 1], [1, 1]) == 1.0
    assert spearman_correlation([1, 1], [2, 2]) == 0.0
    assert top1_agreement({1: 3, 0: 3}, {0: 3, 1: 2})
    assert ndcg([5, 5], [0, 1]) == 1.0


def test_mapping_keys_must_match() -> None:
    with pytest.raises(ValueError, match="identical client IDs"):
        attribution_mae({0: 1.0}, {1: 1.0})

