from __future__ import annotations

import copy
import math

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from fedllm_tracing.adapters.lora_utils import (
    LoraSpec,
    attach_lora,
    count_trainable_parameters,
    extract_lora_update,
    load_lora_update,
    use_lora_update,
)
from fedllm_tracing.data.causal_lm import (
    collate_teacher_forcing,
    make_teacher_forcing_example,
)
from fedllm_tracing.federated.client import train_local_adapter
from fedllm_tracing.federated.types import ClientUpdate
from fedllm_tracing.models.loader import resolve_device, resolve_dtype
from fedllm_tracing.tracing.protoken import PeftTokenTracer


def _config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )


def _model(seed: int = 5) -> torch.nn.Module:
    torch.manual_seed(seed)
    return attach_lora(
        Qwen2ForCausalLM(_config()),
        LoraSpec(rank=2, alpha=4, target_modules=("q_proj", "v_proj")),
    )


def test_device_and_dtype_resolution_on_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_dtype(torch.device("cpu"), "auto") == torch.float32
    with pytest.raises(ValueError, match="float16"):
        resolve_dtype(torch.device("cpu"), "float16")


def test_teacher_forcing_masks_prompt_and_pads() -> None:
    first = make_teacher_forcing_example([1, 4], [9, 10])
    second = make_teacher_forcing_example([1], [8])
    batch = collate_teacher_forcing([first, second], pad_token_id=0)
    assert batch["input_ids"].tolist() == [[1, 4, 9, 10], [1, 8, 0, 0]]
    assert batch["labels"].tolist() == [[-100, -100, 9, 10], [-100, 8, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]


def test_lora_extract_load_and_context_restore() -> None:
    model = _model()
    trainable, total = count_trainable_parameters(model)
    assert 0 < trainable < total
    original = extract_lora_update(model, client_id=0, round_id=0)
    changed_state = {key: tensor.clone() for key, tensor in original.state_dict.items()}
    first_key = next(iter(changed_state))
    changed_state[first_key].add_(0.25)
    changed = ClientUpdate(1, 0, changed_state)
    load_lora_update(model, changed)
    torch.testing.assert_close(
        extract_lora_update(model, client_id=1, round_id=0).state_dict[first_key],
        changed.state_dict[first_key],
    )
    load_lora_update(model, original)
    with use_lora_update(model, changed):
        inside = extract_lora_update(model, client_id=1, round_id=0)
        torch.testing.assert_close(inside.state_dict[first_key], changed.state_dict[first_key])
    restored = extract_lora_update(model, client_id=0, round_id=0)
    torch.testing.assert_close(restored.state_dict[first_key], original.state_dict[first_key])


def test_local_lora_training_changes_adapter_and_has_finite_loss() -> None:
    model = _model()
    before = extract_lora_update(model, client_id=0, round_id=0)
    example = make_teacher_forcing_example([1, 4], [9])
    batch = collate_teacher_forcing([example], pad_token_id=0)
    result = train_local_adapter(
        model, [batch, batch], learning_rate=0.02, device="cpu"
    )
    after = extract_lora_update(model, client_id=0, round_id=0)
    assert len(result.losses) == 2
    assert all(math.isfinite(loss) for loss in result.losses)
    assert any(
        not torch.equal(before.state_dict[key], after.state_dict[key])
        for key in before.state_dict
    )


def test_protoken_replays_exact_global_layer_input_and_restores_adapter() -> None:
    global_model = _model(seed=8)
    global_update = extract_lora_update(global_model, client_id=-1, round_id=0)
    state_one = {key: tensor.clone() for key, tensor in global_update.state_dict.items()}
    state_two = {key: tensor.clone() for key, tensor in global_update.state_dict.items()}
    for key in state_one:
        if ".lora_B." in key:
            state_one[key].add_(0.05)
            state_two[key].sub_(0.03)
    clients = [ClientUpdate(0, 0, state_one), ClientUpdate(1, 0, state_two)]
    layer_name = "base_model.model.model.layers.0"
    observed_inputs: list[int] = []

    def observe_input(_module, args, _kwargs):
        observed_inputs.append(args[0].data_ptr())

    handle = global_model.get_submodule(layer_name).register_forward_pre_hook(
        observe_input, with_kwargs=True
    )
    try:
        attributions = PeftTokenTracer().trace_token_ids(
            global_model,
            clients,
            prompt_ids=[1, 4],
            target_tokens=[9],
            layers=[layer_name],
        )
    finally:
        handle.remove()
    assert len(attributions) == 2
    assert all(math.isfinite(item.score) for item in attributions)
    assert len(observed_inputs) == 3  # one global call plus one replay per client
    assert len(set(observed_inputs)) == 1
    restored = extract_lora_update(global_model, client_id=-1, round_id=0)
    for key in global_update.state_dict:
        torch.testing.assert_close(restored.state_dict[key], global_update.state_dict[key])

