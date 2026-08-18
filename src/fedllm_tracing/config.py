"""Small YAML configuration helpers with deterministic hashing."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml


Config = dict[str, Any]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """Return a recursive merge without mutating either input mapping."""
    result: Config = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def parse_overrides(items: Sequence[str]) -> Config:
    """Parse ``section.key=value`` command-line overrides using YAML values."""
    result: Config = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must have KEY=VALUE form: {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        keys = [part.strip() for part in dotted_key.split(".") if part.strip()]
        if not keys:
            raise ValueError(f"Override has an empty key: {item!r}")
        cursor: MutableMapping[str, Any] = result
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, MutableMapping):
                raise ValueError(f"Conflicting override path: {dotted_key!r}")
            cursor = child
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return result


def load_config(
    paths: Sequence[str | Path], *, overrides: Sequence[str] = ()
) -> Config:
    """Load YAML files from left to right, then apply CLI overrides."""
    config: Config = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Top-level YAML value must be a mapping: {path}")
        config = deep_merge(config, loaded)
    return deep_merge(config, parse_overrides(overrides))


def config_hash(config: Mapping[str, Any], length: int = 12) -> str:
    """Hash a config using canonical JSON serialization."""
    if length <= 0:
        raise ValueError("Hash length must be positive")
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]

