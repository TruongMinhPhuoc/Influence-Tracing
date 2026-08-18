"""Append-only JSONL experiment logging with unique run directories."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from fedllm_tracing.config import config_hash


def _slug(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    return sanitized or "run"


def create_run_directory(
    base_dir: str | Path,
    *,
    experiment: str,
    seed: int,
    timestamp: datetime | None = None,
) -> Path:
    moment = timestamp or datetime.now(timezone.utc)
    stem = f"{moment.strftime('%Y%m%dT%H%M%SZ')}_{_slug(experiment)}_seed{seed}"
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / stem
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


class ResultLogger:
    """Write a config snapshot and enriched JSONL records for one run."""

    def __init__(
        self,
        *,
        base_dir: str | Path,
        experiment: str,
        seed: int,
        config: Mapping[str, Any],
    ) -> None:
        self.experiment = experiment
        self.seed = seed
        self.config_hash = config_hash(config)
        self.run_dir = create_run_directory(
            base_dir, experiment=experiment, seed=seed
        )
        self.results_path = self.run_dir / "results.jsonl"
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(dict(config), sort_keys=True), encoding="utf-8"
        )

    def log(self, record: Mapping[str, Any]) -> None:
        enriched = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": self.experiment,
            "seed": self.seed,
            "config_hash": self.config_hash,
            **dict(record),
        }
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, sort_keys=True, default=str) + "\n")

