from __future__ import annotations

from fedllm_tracing.config import config_hash, deep_merge, parse_overrides


def test_deep_merge_and_overrides() -> None:
    base = {"federated": {"num_clients": 10, "dimension": 100}, "seed": 1}
    merged = deep_merge(base, {"federated": {"dimension": 200}})
    assert merged == {
        "federated": {"num_clients": 10, "dimension": 200},
        "seed": 1,
    }
    assert base["federated"]["dimension"] == 100
    assert parse_overrides(["federated.num_clients=4", "values=[1, 2]"]) == {
        "federated": {"num_clients": 4},
        "values": [1, 2],
    }


def test_config_hash_is_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

