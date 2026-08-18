from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import torch

from fedllm_tracing.federated.types import ClientUpdate
from fedllm_tracing.utils.result_logger import ResultLogger, create_run_directory
from fedllm_tracing.utils.serialization import load_client_update, save_client_update


def test_client_update_round_trip_is_lossless(tmp_path) -> None:
    update = ClientUpdate(
        client_id=4,
        round_id=2,
        state_dict={
            "layer.lora_A": torch.randn(3, 4),
            "layer.lora_B": torch.randn(4, 2, dtype=torch.float64),
        },
        num_examples=17,
    )
    path = save_client_update(update, tmp_path / "client_update")
    restored = load_client_update(path)
    assert restored.client_id == update.client_id
    assert restored.round_id == update.round_id
    assert restored.num_examples == update.num_examples
    assert restored.state_dict.keys() == update.state_dict.keys()
    for key in update.state_dict:
        assert torch.equal(restored.state_dict[key], update.state_dict[key])
        assert restored.state_dict[key].dtype == update.state_dict[key].dtype
    with pytest.raises(FileExistsError):
        save_client_update(update, path)


def test_run_directories_never_overwrite(tmp_path) -> None:
    timestamp = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
    first = create_run_directory(
        tmp_path, experiment="noise sensitivity", seed=42, timestamp=timestamp
    )
    second = create_run_directory(
        tmp_path, experiment="noise sensitivity", seed=42, timestamp=timestamp
    )
    assert first != second
    assert first.exists() and second.exists()


def test_result_logger_enriches_jsonl_records(tmp_path) -> None:
    config = {"experiment": {"seed": 42}}
    logger = ResultLogger(
        base_dir=tmp_path, experiment="fedattr_mock", seed=42, config=config
    )
    logger.log({"client_id": 1, "update_error_norm": 0.2})
    logger.log({"client_id": 2, "update_error_norm": 0.3})
    lines = logger.results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["experiment"] == "fedattr_mock"
    assert first["seed"] == 42
    assert len(first["config_hash"]) == 12
    assert first["client_id"] == 1
    assert (logger.run_dir / "config.yaml").exists()

