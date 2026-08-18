#!/usr/bin/env python
"""Run the first CPU-only FedAttr estimator sanity experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fedllm_tracing.adapters.update_ops import relative_state_error
from fedllm_tracing.config import load_config
from fedllm_tracing.federated.simulator import generate_mock_updates
from fedllm_tracing.secure_agg.fedattr import estimate_client_update
from fedllm_tracing.secure_agg.subset_query import SecureAggregateOracle
from fedllm_tracing.utils.result_logger import ResultLogger
from fedllm_tracing.utils.seed import seed_everything


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="YAML config path; repeat to merge from left to right",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key using a YAML value",
    )
    return parser.parse_args()


def run_experiment(config: dict[str, Any]) -> Path:
    experiment_config = config["experiment"]
    federated_config = config["federated"]
    secure_config = config["secure_agg"]
    seed = int(experiment_config["seed"])
    seed_everything(seed)

    updates = generate_mock_updates(
        num_clients=int(federated_config["num_clients"]),
        dimension=int(federated_config["dimension"]),
        round_id=int(federated_config["round_id"]),
        seed=seed,
        update_scale=float(federated_config["update_scale"]),
    )
    update_by_id = {update.client_id: update for update in updates}
    configured_target = secure_config.get("target_client_id")
    target_ids = (
        [int(configured_target)]
        if configured_target is not None
        else sorted(update_by_id)
    )

    output_base = Path(experiment_config["output_dir"])
    if not output_base.is_absolute():
        output_base = REPOSITORY_ROOT / output_base
    logger = ResultLogger(
        base_dir=output_base,
        experiment=str(experiment_config["name"]),
        seed=seed,
        config=config,
    )

    print(f"K = {len(updates)}")
    print(f"dimension = {int(federated_config['dimension'])}")
    print(f"subset_size = {int(secure_config['subset_size'])}")
    print()
    for query_count in [int(value) for value in secure_config["query_counts"]]:
        errors: list[float] = []
        for target_id in target_ids:
            oracle = SecureAggregateOracle(updates)
            estimate = estimate_client_update(
                target_id=target_id,
                oracle=oracle,
                num_queries=query_count,
                subset_size=int(secure_config["subset_size"]),
                seed=seed + 10_007 * target_id + query_count,
            )
            error = relative_state_error(
                estimate.update.state_dict, update_by_id[target_id].state_dict
            )
            errors.append(error)
            logger.log(
                {
                    "model": "mock_random_tensor",
                    "lora_rank": None,
                    "num_clients": len(updates),
                    "round_id": int(federated_config["round_id"]),
                    "client_id": target_id,
                    "prompt_id": None,
                    "attribution_layers": [],
                    "target_token": None,
                    "num_query_pairs": query_count,
                    "secure_query_count": estimate.secure_query_count,
                    "subset_size": int(secure_config["subset_size"]),
                    "noise_level": None,
                    "update_error_norm": error,
                    "oracle_score": None,
                    "estimated_score": None,
                }
            )
        print(f"M={query_count:<2}  relative_error={mean(errors):.6f}")
    print()
    print(f"Results: {logger.results_path}")
    return logger.run_dir


def main() -> None:
    args = _parse_args()
    config_paths = args.config or [REPOSITORY_ROOT / "configs" / "base.yaml"]
    config = load_config(config_paths, overrides=args.overrides)
    run_experiment(config)


if __name__ == "__main__":
    main()

