"""Attribution fidelity metrics and controlled perturbations."""

from fedllm_tracing.evaluation.metrics import (
    attribution_mae,
    ndcg,
    spearman_correlation,
    top1_agreement,
    topk_overlap,
)
from fedllm_tracing.evaluation.noise import perturb_update

__all__ = [
    "attribution_mae",
    "ndcg",
    "perturb_update",
    "spearman_correlation",
    "top1_agreement",
    "topk_overlap",
]

