"""Explicit device/dtype loading for Hugging Face causal language models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class ModelPreflight:
    model_name: str
    model_type: str
    architectures: tuple[str, ...]
    num_hidden_layers: int | None
    hidden_size: int | None
    vocab_size: int | None
    tokenizer_class: str
    tokenizer_vocab_size: int


def resolve_device(requested: str = "auto") -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    normalized = requested.lower()
    if normalized == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    supported = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in supported:
        raise ValueError(f"Unsupported dtype: {requested!r}")
    dtype = supported[normalized]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 is not a supported CPU execution default")
    return dtype


def load_causal_lm(
    model_name: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> ModelBundle:
    """Load a tokenizer and causal LM without implicit multi-device dispatch."""
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(resolved_device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer defines neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=resolved_dtype,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model.to(resolved_device)
    return ModelBundle(model, tokenizer, resolved_device, resolved_dtype)


def inspect_model_assets(
    model_name: str,
    *,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> ModelPreflight:
    """Download/validate only config and tokenizer, never model weight files."""
    config = AutoConfig.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    return ModelPreflight(
        model_name=model_name,
        model_type=str(getattr(config, "model_type", "unknown")),
        architectures=tuple(getattr(config, "architectures", ()) or ()),
        num_hidden_layers=getattr(config, "num_hidden_layers", None),
        hidden_size=getattr(config, "hidden_size", None),
        vocab_size=getattr(config, "vocab_size", None),
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_vocab_size=len(tokenizer),
    )
