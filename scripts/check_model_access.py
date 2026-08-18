#!/usr/bin/env python
"""Validate Hugging Face model config/tokenizer access without downloading weights."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fedllm_tracing.models.loader import inspect_model_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    metadata = inspect_model_assets(
        args.model, local_files_only=args.local_files_only
    )
    print(json.dumps(asdict(metadata), indent=2))
    print("PREFLIGHT PASS (model weights were not downloaded)")


if __name__ == "__main__":
    main()
