from __future__ import annotations

import pytest
import torch

from fedllm_tracing.tracing.hooks import LayerInputCapture
from fedllm_tracing.tracing.interfaces import TokenAttribution
from fedllm_tracing.tracing.multi_round import aggregate_history


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("sum", 6.0), ("mean", 2.0), ("last", 3.0)],
)
def test_multi_round_baselines(mode: str, expected: float) -> None:
    assert aggregate_history({2: 3.0, 0: 1.0, 1: 2.0}, mode=mode) == expected


def test_weighted_multi_round_baseline() -> None:
    assert aggregate_history(
        {0: 1.0, 1: 3.0},
        mode="weighted_mean",
        weights={0: 1.0, 1: 3.0},
    ) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="one weight"):
        aggregate_history({0: 1.0, 1: 3.0}, mode="weighted_mean", weights={0: 1.0})


def test_token_attribution_sums_layers() -> None:
    attribution = TokenAttribution(1, 42, {"layer.0": 0.2, "layer.1": -0.1})
    assert attribution.score == pytest.approx(0.1)


def test_layer_input_capture_uses_exact_module_input() -> None:
    layer = torch.nn.Linear(3, 2)
    capture = LayerInputCapture().attach(layer)
    input_tensor = torch.randn(4, 3, requires_grad=True)
    with capture:
        layer(input_tensor)
    assert capture.value is input_tensor

