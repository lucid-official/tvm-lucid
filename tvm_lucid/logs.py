"""Reading a run's JSONL logs back, for plots and the live dashboard."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every complete record; a line still being written is skipped."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def num(value: Any) -> float:
    """JSON numbers, and the "nan"/"inf" strings the logger writes for them."""
    try:
        return float(value)
    except TypeError, ValueError:
        return math.nan


def smooth(values: np.ndarray, beta: float = 0.9) -> np.ndarray:
    """Bias-corrected exponential moving average, NaN-tolerant."""
    out = np.empty_like(values)
    avg, weight = 0.0, 0.0
    for i, v in enumerate(values):
        if math.isfinite(v):
            avg = beta * avg + (1 - beta) * v
            weight = beta * weight + (1 - beta)
        out[i] = avg / weight if weight else math.nan
    return out
