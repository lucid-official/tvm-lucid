"""Plot a run's training dynamics and evaluation curves.

    python scripts/plot_dynamics.py runs/cifar10_s2-20261024-010203

Writes PNGs to ``<run>/plots/``.  Safe to run while training is going on —
it only reads the JSONL logs.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def num(value: Any) -> float:
    """JSON numbers, and the "nan"/"inf" strings the logger writes for them."""
    try:
        return float(value)
    except (TypeError, ValueError):
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


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def plot_steps(rows: list[dict[str, Any]], out: Path) -> None:
    steps = [r for r in rows if r.get("kind") == "step"]
    if not steps:
        return
    x = np.array([r["global_step"] for r in steps])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in (("loss", "total"), ("terminal", "terminal velocity"), ("fm", "flow matching")):
        y = np.array([num(r.get(key)) for r in steps])
        ax.plot(x, y, alpha=0.2)
        ax.plot(x, smooth(y), label=label)
    ax.set_yscale("log")
    ax.set_xlabel("optimiser step")
    ax.set_title("TVM objective (Eq. 15) and its two terms")
    ax.legend()
    save(fig, out / "loss.png")

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(x, [num(r.get("img_per_s")) for r in steps], color="tab:blue")
    ax1.set_ylabel("images / s", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(x, [num(r.get("peak_gib")) for r in steps], color="tab:red")
    ax2.set_ylabel("peak Metal memory (GiB)", color="tab:red")
    ax1.set_xlabel("optimiser step")
    ax1.set_title("throughput and memory")
    save(fig, out / "throughput.png")


def plot_dynamics(rows: list[dict[str, Any]], out: Path) -> None:
    dyn = [r for r in rows if r.get("kind") == "dynamics"]
    if not dyn:
        return
    x = np.array([r["global_step"] for r in dyn])

    # Bottom to top in network order: embeddings, blocks, final layer.
    rank = {"embed": 0, "final": 2}
    groups = sorted(
        (k for k in dyn[-1]["grad_norm"] if k != "total"), key=lambda g: (rank.get(g, 1), g)
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [1, 2]})
    ax1.plot(x, [num(r["grad_norm"]["total"]) for r in dyn], label="grad")
    ax1.set_yscale("log")
    ax1.set_title("global gradient norm")
    heat = np.array([[num(r["grad_norm"].get(g)) for r in dyn] for g in groups])
    image = ax2.imshow(
        np.log10(heat + 1e-12), aspect="auto", origin="lower", interpolation="nearest",
        extent=[x[0], x[-1], -0.5, len(groups) - 0.5],
    )  # fmt: skip
    ax2.set_yticks(range(len(groups)), groups, fontsize=7)
    ax2.set_xlabel("optimiser step")
    ax2.set_title("log10 gradient norm by group")
    fig.colorbar(image, ax=ax2)
    save(fig, out / "grad_norms.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, [num(r.get("update_ratio")) for r in dyn], label="‖Δθ‖ / ‖θ‖ per step")
    for name in ("target", "eval"):
        values = [num(r.get("ema_drift", {}).get(name)) for r in dyn]
        if any(math.isfinite(v) for v in values):
            ax.plot(x, values, label=f"EMA drift ({name}) ‖θ_ema − θ‖ / ‖θ‖")
    ax.set_yscale("log")
    ax.set_xlabel("optimiser step")
    ax.legend(fontsize=8)
    ax.set_title("update size and EMA lag")
    save(fig, out / "updates_ema.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in sorted(dyn[-1].get("activation_rms", {})):
        ax.plot(x, [num(r.get("activation_rms", {}).get(name)) for r in dyn], label=name)
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("activation RMS")
    ax.set_title("time-embedding / raw adaLN modulation activations (cf. paper Fig. 4)")
    ax.legend()
    save(fig, out / "activations.png")


def plot_epochs(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        return
    epochs = np.array([r["epoch"] for r in rows])

    fid_keys = sorted({k for r in rows for k in r if k.startswith("fid_nfe")})
    if fid_keys:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for key in fid_keys:
            pts = [(r["epoch"], num(r[key])) for r in rows if key in r]
            ax1.plot(*zip(*pts, strict=True), marker="o", label=key.replace("fid_", ""))
            is_key = key.replace("fid_", "is_")
            pts = [(r["epoch"], num(r[is_key])) for r in rows if is_key in r]
            ax2.plot(*zip(*pts, strict=True), marker="o", label=is_key.replace("is_", ""))
        ax1.set_title("FID (Lucid Inception-v3, vs. train split)")
        ax2.set_title("Inception Score")
        for ax in (ax1, ax2):
            ax.set_xlabel("epoch")
            ax.legend()
        save(fig, out / "quality.png")

    gap_keys = sorted({k for r in rows for k in r if k.startswith("flowmap_gap_")})
    if gap_keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        for key in gap_keys:
            pts = [(r["epoch"], num(r[key])) for r in rows if key in r]
            ax.plot(*zip(*pts, strict=True), label=key.replace("flowmap_gap_", "NFE "))
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title("flow-map consistency: MSE(few-step, many-step) on fixed noise")
        ax.legend()
        save(fig, out / "flowmap_gap.png")

    probed = [r for r in rows if "probe_fm_live" in r]
    if probed:
        s = probed[-1]["probe_s"]
        names = [n for n in ("live", "ema") if f"probe_fm_{n}" in probed[-1]]
        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4), squeeze=False)
        for ax, name in zip(axes[0], names, strict=True):
            heat = np.array([r[f"probe_fm_{name}"] for r in probed]).T
            image = ax.imshow(
                np.log10(heat), aspect="auto", origin="lower", interpolation="nearest",
                extent=[probed[0]["epoch"], probed[-1]["epoch"], s[0], s[-1]],
            )  # fmt: skip
            ax.set_xlabel("epoch")
            ax.set_ylabel("s")
            ax.set_title(f"log10 FM error by time ({name})")
            fig.colorbar(image, ax=ax)
        save(fig, out / "probe_by_time.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, [num(r.get("eval_seconds")) for r in rows], marker=".")
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds")
    ax.set_title("evaluation cost per epoch")
    save(fig, out / "eval_cost.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    out = args.run / "plots"
    out.mkdir(exist_ok=True)
    metrics = read_jsonl(args.run / "metrics.jsonl")
    plot_steps(metrics, out)
    plot_dynamics(metrics, out)
    plot_epochs(read_jsonl(args.run / "epoch.jsonl"), out)


if __name__ == "__main__":
    main()
