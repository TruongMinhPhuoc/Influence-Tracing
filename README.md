# Federated LLM Influence Tracing

Research infrastructure for privacy-preserving, behavior-specific client
provenance tracing in federated language models under Secure Aggregation.

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
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run all tests and the mock experiment:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\test_fedattr_mock.py
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

## Next research gates

After GPU access, implement oracle ProToken with 3 clients, one FL round, a
small LoRA model, and teacher-forced prompts. Then measure attribution MAE,
Spearman correlation, Top-1 agreement, Top-k overlap, and NDCG as controlled
relative update noise increases. Replace synthetic noise with FedAttr estimates
only after the oracle/noise path is verified; add multi-round tracing only after
single-round ranking preservation is established.

