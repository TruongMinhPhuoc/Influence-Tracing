#!/usr/bin/env python
"""Run an offline tiny-Qwen LoRA -> ProToken -> noise smoke experiment on CPU."""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fedllm_tracing.adapters.lora_utils import (
    LoraSpec,
    attach_lora,
    extract_lora_update,
    load_lora_update,
)
from fedllm_tracing.data.causal_lm import (
    collate_teacher_forcing,
    make_teacher_forcing_example,
)
from fedllm_tracing.evaluation.metrics import (
    attribution_mae,
    ndcg,
    spearman_correlation,
    top1_agreement,
)
from fedllm_tracing.evaluation.noise import perturb_update
from fedllm_tracing.federated.client import train_local_adapter
from fedllm_tracing.federated.fedavg import fedavg
from fedllm_tracing.federated.types import ClientUpdate
from fedllm_tracing.tracing.protoken import PeftTokenTracer
from fedllm_tracing.utils.result_logger import ResultLogger
from fedllm_tracing.utils.seed import seed_everything


def _tiny_qwen_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )


def _scores_by_client(attributions: list[object]) -> dict[int, float]:
    scores: defaultdict[int, float] = defaultdict(float)
    for attribution in attributions:
        scores[attribution.client_id] += attribution.score
    return dict(scores)


def main() -> None:
    seed = 42
    seed_everything(seed)
    config = _tiny_qwen_config()
    base_model = Qwen2ForCausalLM(config)
    base_state = copy.deepcopy(base_model.state_dict())
    lora_spec = LoraSpec(rank=2, alpha=4, target_modules=("q_proj", "v_proj"))
    template_base = Qwen2ForCausalLM(config)
    template_base.load_state_dict(base_state)
    template_model = attach_lora(template_base, lora_spec)
    initial_adapter = extract_lora_update(
        template_model, client_id=-1, round_id=0
    )

    client_updates = []
    training_summaries = []
    facts = [([1, 10], [50]), ([1, 11], [51]), ([1, 12], [52])]
    for client_id, (prompt_ids, target_ids) in enumerate(facts):
        client_base = Qwen2ForCausalLM(config)
        client_base.load_state_dict(base_state)
        client_model = attach_lora(client_base, lora_spec)
        load_lora_update(client_model, initial_adapter)
        example = make_teacher_forcing_example(prompt_ids, target_ids)
        batch = collate_teacher_forcing([example], pad_token_id=0)
        training = train_local_adapter(
            client_model,
            [batch] * 6,
            learning_rate=0.03,
            epochs=1,
            device="cpu",
        )
        client_updates.append(
            extract_lora_update(
                client_model,
                client_id=client_id,
                round_id=0,
                num_examples=1,
            )
        )
        training_summaries.append((training.initial_loss, training.final_loss))

    global_base = Qwen2ForCausalLM(config)
    global_base.load_state_dict(base_state)
    global_model = attach_lora(global_base, lora_spec)
    load_lora_update(global_model, initial_adapter)
    averaged_state = fedavg(client_updates, weighted=True)
    load_lora_update(
        global_model,
        ClientUpdate(-1, 0, averaged_state, num_examples=3),
    )

    tracer = PeftTokenTracer()
    layers = [
        "base_model.model.model.layers.0",
        "base_model.model.model.layers.1",
    ]
    run_config = {
        "model": "offline_tiny_random_qwen2",
        "num_clients": 3,
        "round_id": 0,
        "lora_rank": 2,
        "training_steps": 6,
        "noise_levels": [0.0, 0.1, 0.5],
        "seed": seed,
    }
    logger = ResultLogger(
        base_dir=REPOSITORY_ROOT / "outputs",
        experiment="protoken_cpu_smoke",
        seed=seed,
        config=run_config,
    )

    print("Offline tiny-Qwen2 ProToken CPU smoke")
    print("clients = 3, layers = 2, LoRA rank = 2")
    for client_id, (initial, final) in enumerate(training_summaries):
        print(f"client={client_id} loss={initial:.6f}->{final:.6f}")

    for prompt_id, (prompt_ids, target_ids) in enumerate(facts):
        oracle_attributions = tracer.trace_token_ids(
            global_model, client_updates, prompt_ids, target_ids, layers
        )
        oracle_scores = _scores_by_client(oracle_attributions)
        print(f"prompt={prompt_id} oracle_scores={oracle_scores}")
        for noise_level in run_config["noise_levels"]:
            noisy_updates = [
                perturb_update(
                    update,
                    float(noise_level),
                    seed=seed + 1000 * prompt_id + update.client_id,
                )
                for update in client_updates
            ]
            noisy_attributions = tracer.trace_token_ids(
                global_model, noisy_updates, prompt_ids, target_ids, layers
            )
            noisy_scores = _scores_by_client(noisy_attributions)
            summary = {
                "mae": attribution_mae(oracle_scores, noisy_scores),
                "spearman": spearman_correlation(oracle_scores, noisy_scores),
                "top1": top1_agreement(oracle_scores, noisy_scores),
                "ndcg": ndcg(oracle_scores, noisy_scores),
            }
            print(f"  noise={noise_level:.1f} metrics={summary}")
            for client_id in sorted(oracle_scores):
                logger.log(
                    {
                        "model": run_config["model"],
                        "lora_rank": run_config["lora_rank"],
                        "num_clients": run_config["num_clients"],
                        "round_id": 0,
                        "client_id": client_id,
                        "prompt_id": f"tiny_p{prompt_id}",
                        "target_token": target_ids[0],
                        "attribution_layers": layers,
                        "noise_level": noise_level,
                        "update_error_norm": noise_level,
                        "oracle_score": oracle_scores[client_id],
                        "estimated_score": noisy_scores[client_id],
                        **summary,
                    }
                )
    print(f"Results: {logger.results_path}")
    print("SMOKE PASS (implementation validation only; not a scientific result)")


if __name__ == "__main__":
    main()
