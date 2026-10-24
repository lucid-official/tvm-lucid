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
from importlib import metadata
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


def _lucid_distribution() -> str | None:
    """The installed Lucid release (None for an unpackaged checkout)."""
    try:
        return metadata.version("lucid-dl")
    except metadata.PackageNotFoundError:
        return None


def _lucid_checkout(source: Path) -> dict[str, Any] | None:
    """Commit and local-change flag when Lucid runs from a git checkout.

    A checkout's version string does not move between releases, so the commit
    is what identifies the code a run used.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except OSError, subprocess.CalledProcessError:
        return None


def environment() -> dict[str, Any]:
    """What a reader needs to reproduce or compare the run."""
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "lucid_version": lucid.__version__,
        "lucid_dl": _lucid_distribution(),
        "lucid_path": str(Path(lucid.__file__).resolve().parent),
        "lucid_checkout": _lucid_checkout(Path(lucid.__file__).resolve().parent.parent),
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
    lucid.save(state, str(tmp))  # str: lucid.save treats anything else as a file object
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    # lucid.load is lucid.serialization.load, which takes map_location (3.15.5's
    # top-level stub leaves it out).
    state = lucid.load(str(path), map_location="metal")
    if not isinstance(state, dict):
        raise TypeError(f"{path} is not a tvm-lucid checkpoint")
    return state
