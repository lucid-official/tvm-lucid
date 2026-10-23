"""Run directories, environment capture, JSONL logs and checkpoints.

A run directory is self-describing::

    runs/<name>-<timestamp>/
        config.json        resolved configuration
        env.json           machine, Lucid version and commit, start time
        metrics.jsonl      one record per logged step (training dynamics)
        epoch.jsonl        one record per epoch (evaluation)
        events.jsonl       resumes, skipped steps, new best FID, stops
        samples/           epoch_XXXX_nfeK.png grids
        ckpt/              last.lucid, best_fid.lucid, epoch_XXXX.lucid
        plots/             written by scripts/plot_dynamics.py
"""

import json
import math
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import lucid
import lucid.metal as metal


def new_run_dir(root: Path, name: str) -> Path:
    """``root/<name>-YYYYmmdd-HHMMSS``, created."""
    path = root / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    for sub in ("samples", "ckpt", "plots"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def _lucid_commit() -> str | None:
    source = Path(lucid.__file__).resolve().parent.parent
    try:
        return subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, Any]:
    """What a reader needs to reproduce or compare the run."""
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "lucid_version": lucid.__version__,
        "lucid_commit": _lucid_commit(),
        "device": metal.get_device_name(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _clean(value: Any) -> Any:
    """JSON has no NaN or infinity; write them as strings a reader can see."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class JsonlLog:
    """Append-only JSON-lines file; every record gets a wall-clock stamp."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = open(path, "a", buffering=1)

    def write(self, record: dict[str, Any]) -> None:
        record = {"time": round(time.time(), 3), **record}
        self._handle.write(json.dumps(_clean(record)) + "\n")

    def close(self) -> None:
        self._handle.close()


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Write atomically: a crash mid-write leaves the previous file intact."""
    tmp = path.with_name(path.name + ".tmp")
    lucid.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    state = lucid.load(path, map_location="metal")
    if not isinstance(state, dict):
        raise TypeError(f"{path} is not a tvm-lucid checkpoint")
    return state
