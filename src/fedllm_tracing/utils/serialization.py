"""Safe, lossless ClientUpdate serialization using safetensors."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from fedllm_tracing.adapters.update_ops import state_to_cpu_contiguous
from fedllm_tracing.federated.types import ClientUpdate


def _paths(path: str | Path) -> tuple[Path, Path]:
    tensor_path = Path(path)
    if tensor_path.suffix != ".safetensors":
        tensor_path = tensor_path.with_suffix(".safetensors")
    return tensor_path, tensor_path.with_suffix(".json")


def save_client_update(
    update: ClientUpdate, path: str | Path, *, overwrite: bool = False
) -> Path:
    tensor_path, metadata_path = _paths(path)
    if not overwrite and (tensor_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"Refusing to overwrite update at {tensor_path}")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_to_cpu_contiguous(update.state_dict), str(tensor_path))
    metadata = {
        "client_id": update.client_id,
        "round_id": update.round_id,
        "num_examples": update.num_examples,
    }
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return tensor_path


def load_client_update(path: str | Path) -> ClientUpdate:
    tensor_path, metadata_path = _paths(path)
    if not tensor_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Incomplete serialized update at {tensor_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ClientUpdate(
        client_id=int(metadata["client_id"]),
        round_id=int(metadata["round_id"]),
        state_dict=load_file(str(tensor_path), device="cpu"),
        num_examples=metadata.get("num_examples"),
    )

