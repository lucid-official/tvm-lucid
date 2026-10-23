"""TOML loading, overrides, and refusal of unknown keys."""

from pathlib import Path

import pytest

from tvm_lucid import config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads() -> None:
    cfg = config.load(ROOT / "configs" / "cifar10_s2.toml")
    assert cfg.train.betas == (0.9, 0.95)
    assert cfg.eval.nfe == (1, 2, 4)
    assert cfg.model["hidden_size"] == 384


def test_overrides_are_parsed_as_toml() -> None:
    cfg = config.load(
        ROOT / "configs" / "cifar10_s2.toml",
        ["train.batch_size=32", "train.lr=2e-4", "eval.nfe=[1, 8]", "model.depth=6", "name=x"],
    )
    assert cfg.train.batch_size == 32 and cfg.train.lr == 2e-4
    assert cfg.eval.nfe == (1, 8) and cfg.model["depth"] == 6 and cfg.name == "x"


def test_unknown_keys_are_refused() -> None:
    with pytest.raises(KeyError):
        config.load(ROOT / "configs" / "cifar10_s2.toml", ["train.batchsize=32"])


def test_round_trip_through_json_dict() -> None:
    cfg = config.load(ROOT / "configs" / "debug.toml")
    again = config.from_dict(cfg.to_dict())
    assert again == cfg
