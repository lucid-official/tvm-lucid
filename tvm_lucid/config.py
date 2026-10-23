"""Experiment configuration: a TOML file, plus ``--set section.key=value`` overrides.

Three typed sections drive the code (``data``, ``train``, ``eval``); the
``model`` section is passed through untouched as overrides of
:class:`tvm_lucid.tvm.TVMConfig`, so any field that config knows — the time
sampler, the EMA rates, the scaled parameterisation — can be set from the
file without this module listing it.
"""

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Where CIFAR-10 lives and how it is augmented."""

    root: str = "~/datasets"
    flip: bool = True


@dataclass
class TrainConfig:
    """Optimisation and the run's length."""

    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    epochs: int = 400
    hours: float = 84.0
    log_every: int = 25
    dynamics_every: int = 200
    keep_every_epochs: int = 10
    max_nonfinite: int = 10


@dataclass
class EvalConfig:
    """What is measured at the end of an epoch, and how often."""

    every_epochs: int = 1
    nfe: tuple[int, ...] = (1, 2, 4)
    grid_noise_seed: int = 1234
    probe_batch: int = 256
    probe_buckets: int = 10
    probe_seed: int = 4321
    fid_every_epochs: int = 5
    fid_samples: int = 5000
    fid_nfe: tuple[int, ...] = (1, 2)
    fid_batch: int = 250
    sample_batch: int = 500
    dashboard: bool = True  # runs/<run>/dashboard.png, rewritten in place every epoch
    dashboard_nfe: int = 8  # jumps shown in its noise -> data strip
    dashboard_every_steps: int = 0  # also redraw every N optimiser steps (0: epochs only)


@dataclass
class ExperimentConfig:
    """One run, fully specified."""

    name: str = "cifar10_s2"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    model: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready copy, for ``config.json`` in the run directory."""
        return dataclasses.asdict(self)


_SECTIONS: dict[str, type] = {"data": DataConfig, "train": TrainConfig, "eval": EvalConfig}


def _typed(cls: type, values: dict[str, Any]) -> Any:
    """Build a section dataclass, refusing keys it does not have."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise KeyError(f"[{cls.__name__}] has no field(s) {unknown}")
    coerced = {
        key: tuple(value) if isinstance(value, list) else value for key, value in values.items()
    }
    return cls(**coerced)


def _parse_override(item: str) -> tuple[list[str], Any]:
    """``"train.batch_size=32"`` → ``(["train", "batch_size"], 32)``.

    The value is read as TOML, so ``32``, ``1e-4``, ``true``, ``[1, 2]`` and
    ``"text"`` all mean what they would in the file.
    """
    if "=" not in item:
        raise ValueError(f"--set expects key=value, got {item!r}")
    key, raw = item.split("=", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw  # a bare word: take it as a string
    return key.strip().split("."), value


def from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """An :class:`ExperimentConfig` from nested plain data."""
    raw = dict(raw)
    sections = {name: _typed(cls, raw.pop(name, {})) for name, cls in _SECTIONS.items()}
    model = dict(raw.pop("model", {}))
    unknown = sorted(set(raw) - {"name", "seed"})
    if unknown:
        raise KeyError(f"unknown top-level key(s) {unknown}")
    return ExperimentConfig(
        name=raw.get("name", "cifar10_s2"),
        seed=int(raw.get("seed", 0)),
        model=model,
        **sections,
    )


def load(
    path: Path | None, overrides: list[str] = (), base: dict[str, Any] | None = None
) -> ExperimentConfig:
    """Read ``path`` (or start from ``base``), apply overrides, validate."""
    raw: dict[str, Any] = dict(base or {})
    if path is not None:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    raw = {k: dict(v) if isinstance(v, dict) else v for k, v in raw.items()}
    for item in overrides:
        keys, value = _parse_override(item)
        node = raw
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return from_dict(raw)


def resolve_root(cfg: ExperimentConfig) -> Path:
    """The dataset root with ``~`` expanded."""
    return Path(cfg.data.root).expanduser()
