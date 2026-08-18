# Federated LLM Influence Tracing

Research infrastructure for privacy-preserving, behavior-specific client
provenance tracing in federated language models under Secure Aggregation.

Current implementation status, validated results, and the GPU handoff checklist
are documented in [`docs/PROJECT_PROGRESS.md`](docs/PROJECT_PROGRESS.md).

The immediate question is deliberately narrow: **does a noisy FedAttr-style
estimate of a client update preserve enough information for ProToken client
attribution?** This repository first builds a CPU-only, model-free testbed for
update estimation and evaluation. It does not currently implement federated
unlearning or claim exact causal influence.

## Scientific roles

- **FedAttr** estimates a client update from paired Secure Aggregation subset
  sums. The estimate is intentionally noisy and the estimator only accesses an
  aggregate-query oracle.
- **ProToken** will attribute individual generated tokens by passing the exact
  captured global layer input through a client-updated layer and taking its
  inner product with the global target-token gradient. It is not hidden-state
  similarity between independent full-model forward passes.
- Multi-round aggregation is postponed until single-round attribution remains
  useful under controlled noise and FedAttr estimation. Future multi-round
  outputs are historical provenance scores, not counterfactual effects.

## CPU milestone

Create and activate an isolated environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,llm]"
```

Run all tests and the mock experiment:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\test_fedattr_mock.py
.\.venv\Scripts\python.exe scripts\run_protoken_cpu_smoke.py
```

The default experiment generates 10 random LoRA-like client updates of
dimension 10,000 and evaluates `M = 2, 3, 5, 10, 20` paired queries. Each run
creates a unique directory under `outputs/` containing `config.yaml` and
append-only `results.jsonl`.

Configuration files are merged left-to-right. Dotted command-line overrides
take highest precedence:

```powershell
.\.venv\Scripts\python.exe scripts\test_fedattr_mock.py `
  --config configs\base.yaml `
  --set federated.num_clients=6 `
  --set secure_agg.query_counts=[5,20]
```

Here, `subset_size` means the number of non-target masking clients sampled in
each arm of a pair. The target-present aggregate therefore contains one more
client than the target-absent aggregate. `num_queries`/`M` counts pairs, so each
estimate makes `2M` aggregate queries.

## LLM integration available before GPU access

The optional `llm` dependencies provide Transformers, PEFT, Datasets, and
Accelerate. The repository now includes:

- explicit CPU/CUDA device and dtype resolution;
- Hugging Face causal-LM/tokenizer loading;
- PEFT LoRA attachment, state extraction, loading, and exception-safe swapping;
- prompt-masked teacher-forcing batches;
- a minimal local LoRA training loop;
- a concrete PEFT ProToken tracer;
- an offline tiny-Qwen end-to-end CPU smoke experiment.

The tracer performs one global full-model forward, computes target-logit
gradients from that graph, and replays only the selected client-updated layers
with the exact positional arguments and keyword arguments captured from the
global layer calls. It never performs a separate client full-model forward.

Before using a GPU, config/tokenizer access for the intended checkpoint can be
checked without downloading model weights:

```powershell
.\.venv\Scripts\python.exe scripts\check_model_access.py `
  --model Qwen/Qwen2.5-0.5B
```

The offline smoke uses a randomly initialized, very small Qwen2 architecture.
It trains three synchronized LoRA clients on distinct token facts, applies
FedAvg, computes oracle ProToken scores, and evaluates controlled noise. Its
results validate implementation only and must not be reported as scientific
evidence. All federated clients start from the same base weights and the same
initial LoRA adapter state before local training.

## Next research gates

After GPU access, replace the offline random Qwen with Qwen2.5-0.5B and a small
controlled text dataset: 3 clients, one FL round, LoRA rank 4, and teacher-forced
prompts. Measure attribution MAE, Spearman correlation, Top-1 agreement, Top-k
overlap, and NDCG as controlled relative update noise increases. Replace
synthetic noise with FedAttr estimates only after the oracle/noise path is
verified; add multi-round tracing only after single-round ranking preservation
is established.
