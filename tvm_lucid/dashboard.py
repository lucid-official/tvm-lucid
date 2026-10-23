"""A live training dashboard: one PNG, rewritten in place as training goes.

    runs/<name>-<timestamp>/dashboard.png     open it once and leave it open
    runs/<name>-<timestamp>/dashboard.html    the same image, reloading itself

Every write goes to a temporary file that then replaces ``dashboard.png`` in
one step, so a viewer never reads half an image.  Preview reloads the file
when it changes; the HTML page re-fetches it every 30 seconds in any browser.

A 2x2 grid of training curves above the sampler's denoising path:

    loss: per iteration and per epoch     samples: FID · held-out FM error
    optimisation: ‖∇θ‖ · update ratio     consistency: flow-map gap · sample std
    noise x_1 → data x_0 in ``eval.dashboard_nfe`` jumps of the evaluation
    network, one class per row, from the same noise at every redraw

The loss panel draws the three terms of Eq. 15 together, each both as the
running loss by optimiser step and as its epoch mean.  The full set of
dynamics plots is ``scripts/plot_dynamics.py``.
"""

import math
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import lucid  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from tvm_lucid.data import CLASSES  # noqa: E402
from tvm_lucid.logs import num, read_jsonl, smooth  # noqa: E402
from tvm_lucid.model import DEVICE  # noqa: E402
from tvm_lucid.tvm import TVM  # noqa: E402

# Each loss term in one colour, per iteration and per epoch alike.
_LOSSES = (
    ("loss", "total (Eq. 15)", "black"),
    ("terminal", "terminal velocity", "tab:blue"),
    ("fm", "flow matching", "tab:orange"),
)
_NFE_COLOURS = ("tab:red", "tab:purple", "tab:green", "tab:brown")

_HTML = """<!doctype html>
<meta charset="utf-8">
<title>TVM dashboard</title>
<style>body{margin:0;background:#fff}img{display:block;width:100%;height:auto}</style>
<img id="d" src="dashboard.png" alt="dashboard">
<script>
setInterval(() => { document.getElementById("d").src = "dashboard.png?" + Date.now(); }, 30000);
</script>
"""


def _positive(values: list[float]) -> np.ndarray:
    """For log axes: anything not strictly positive becomes a gap."""
    out = np.array(values, dtype=float)
    out[~(out > 0)] = math.nan
    return out


def _empty(ax: Axes, title: str, note: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.text(0.5, 0.5, note, ha="center", va="center", transform=ax.transAxes, color="0.5")
    ax.set_xticks([])
    ax.set_yticks([])


def _finish(ax: Axes, title: str, xlabel: str, twin: Axes | None = None, pad: float = 6.0) -> None:
    """Title, grid, and one legend for an axis and its twin."""
    ax.set_title(title, fontsize=12, pad=pad)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if twin is not None:
        twin.tick_params(labelsize=8)
        more = twin.get_legend_handles_labels()
        handles, labels = handles + more[0], labels + more[1]
    if handles:
        ax.legend(handles, labels, fontsize=8)


def _by_epoch(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """``(completed epochs, values)`` for the epoch rows that carry ``key``."""
    pts = [(r["epoch"] + 1, num(r[key])) for r in rows if key in r]
    return [p[0] for p in pts], [p[1] for p in pts]


class Dashboard:
    """Renders ``dashboard.png`` for one run from its logs and the live model.

    Parameters
    ----------
    run_dir : Path
        The run directory; the logs are read from it and the image written to it.
    nfe : int
        Jumps in the denoising path.
    seed : int
        Seed of its fixed noise.
    steps_per_epoch : int
        Optimiser steps per epoch, for the loss panel's epoch axis.
    fid_every : int
        ``eval.fid_every_epochs``, for the note shown until the first FID.
    data_std : float
        Standard deviation of the training images in [-1, 1], the reference
        line of the sample-std curves.
    """

    def __init__(
        self,
        run_dir: Path,
        nfe: int,
        seed: int,
        steps_per_epoch: int,
        fid_every: int,
        data_std: float,
    ) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "dashboard.png"
        self.nfe = nfe
        self.steps_per_epoch = steps_per_epoch
        self.fid_every = fid_every
        self.data_std = data_std
        rng = np.random.default_rng(seed)
        self.noise = lucid.tensor(
            rng.standard_normal((len(CLASSES), 3, 32, 32)).astype(np.float32), device=DEVICE
        )
        self.labels = lucid.tensor(np.arange(len(CLASSES)), device=DEVICE)
        html = run_dir / "dashboard.html"
        if not html.exists():
            html.write_text(_HTML)

    def update(self, model: TVM, epoch: float, step: int, note: str = "") -> None:
        """Sample the path, redraw every panel, replace ``dashboard.png``."""
        training = model.training
        model.eval()
        try:
            states = model.trajectory(
                len(CLASSES), labels=self.labels, steps=self.nfe, noise=self.noise
            )
            path = np.stack([x.to("cpu").numpy() for x in states])  # (nfe + 1, 10, 3, 32, 32)
        finally:
            model.train(training)
        decay = model.config.eval_ema_decay
        network = "live θ" if decay is None else "θ** (EMA)"
        if decay is not None and decay**step >= 0.01:
            # Without warm-up, θ** is still γ^step the initial weights: say so while it shows.
            network = f"θ** (EMA {decay:g}, still {decay**step:.0%} the initial weights)"
        fig = self._figure(path, epoch, step, note, network)
        tmp = self.path.with_name(".dashboard.tmp.png")
        fig.savefig(tmp, dpi=100)
        plt.close(fig)
        os.replace(tmp, self.path)

    # ── the figure ───────────────────────────────────────────────────────

    def _figure(
        self, path: np.ndarray, epoch: float, step: int, note: str, network: str
    ) -> plt.Figure:
        metrics = read_jsonl(self.run_dir / "metrics.jsonl")
        epochs = read_jsonl(self.run_dir / "epoch.jsonl")
        steps = [r for r in metrics if r.get("kind") == "step"]
        dyn = [r for r in metrics if r.get("kind") == "dynamics"]

        fig = plt.figure(figsize=(16, 17))
        grid = GridSpec(
            3, 2, figure=fig, height_ratios=[1, 1, 1.05],
            left=0.06, right=0.94, top=0.92, bottom=0.03, hspace=0.33, wspace=0.3,
        )  # fmt: skip
        self._loss(fig.add_subplot(grid[0, 0]), steps, epochs)
        self._samples(fig.add_subplot(grid[0, 1]), epochs)
        self._optimisation(fig.add_subplot(grid[1, 0]), dyn)
        self._consistency(fig.add_subplot(grid[1, 1]), epochs)

        # Ten classes as two halves side by side, so the path spans the width.
        half = path.shape[1] // 2
        left, right = fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])
        self._path(left, path[:, :half], CLASSES[:half])
        self._path(right, path[:, half:], CLASSES[half:])
        fig.text(
            0.5, left.get_position().y1 + 0.008,
            f"denoising path of {network}: noise x₁ → data x₀ in {path.shape[0] - 1} jumps "
            "(the same noise at every redraw; one class per row)",
            ha="center", va="bottom", fontsize=13,
        )  # fmt: skip

        fids = [num(r[k]) for r in epochs for k in r if k.startswith("fid_nfe")]
        fids = [f for f in fids if math.isfinite(f)]
        parts = [self.run_dir.name, f"epoch {epoch:.2f}", f"step {step:,}"]
        if steps:
            parts.append(f"loss {num(steps[-1].get('loss_mean', steps[-1].get('loss'))):.4g}")
        parts.append(f"best FID {min(fids):.2f}" if fids else "best FID —")
        parts.append(time.strftime("%Y-%m-%d %H:%M:%S"))
        if note:
            parts.append(note)
        fig.suptitle("   ·   ".join(parts), fontsize=14, y=0.985)
        return fig

    # ── the four panels ──────────────────────────────────────────────────

    def _loss(self, ax: Axes, steps: list[dict[str, Any]], epochs: list[dict[str, Any]]) -> None:
        """Running loss by step (lines) and epoch means (markers, at each epoch's last step)."""
        title = "loss: running, per iteration (lines) and per epoch (markers)"
        if not steps:
            return _empty(ax, title, "after the first logged step")
        x = [r["global_step"] for r in steps]
        for key, label, colour in _LOSSES:
            # The total, dashed and on top: it nearly equals the FM term early on.
            z, ls = (3, "--") if key == "loss" else (2, "-")
            raw = _positive([num(r.get(key)) for r in steps])
            mean = _positive([num(r.get(f"{key}_mean", r.get(key))) for r in steps])
            ax.plot(x, raw, color=colour, alpha=0.12, lw=0.7, zorder=z)
            ax.plot(x, smooth(mean, 0.8), color=colour, lw=1.4, ls=ls, zorder=z)
            ended = [
                (r["global_step"], num(r[f"train_{key}"])) for r in epochs if f"train_{key}" in r
            ]
            if ended:
                ax.plot(
                    [e[0] for e in ended], _positive([e[1] for e in ended]),
                    color=colour, ls="none", marker="o", ms=6, mec="white", mew=1.2, zorder=z,
                )  # fmt: skip
            # One legend entry per term: its line and its epoch marker together.
            ax.add_line(Line2D([], [], color=colour, lw=1.4, ls=ls, marker="o", ms=6, mec="white",
                               label=label))  # fmt: skip
        ax.set_yscale("log")
        per = max(self.steps_per_epoch, 1)
        top = ax.secondary_xaxis("top", functions=(lambda s: s / per, lambda e: e * per))
        top.set_xlabel("epoch", fontsize=9)
        top.tick_params(labelsize=8)
        _finish(ax, title, "optimiser step", pad=10.0)

    def _samples(self, ax: Axes, epochs: list[dict[str, Any]]) -> None:
        title = "samples: FID by NFE · held-out flow-matching error"
        prefix = "fid_nfe"
        keys = sorted(
            {k for r in epochs for k in r if k.startswith(prefix) and k[len(prefix) :].isdigit()},
            key=lambda k: int(k[len(prefix) :]),
        )
        ema = any("probe_fm_ema_mean" in r for r in epochs)
        probe = "probe_fm_ema_mean" if ema else "probe_fm_live_mean"
        if not keys and not any(probe in r for r in epochs):
            return _empty(ax, title, "after the first epoch")
        for key, colour in zip(keys, _NFE_COLOURS, strict=False):
            x, y = _by_epoch(epochs, key)
            ax.plot(x, y, color=colour, marker="o", ms=4, lw=1.6,
                    label=f"FID, {key[len(prefix):]} NFE")  # fmt: skip
        if keys:
            ax.set_ylabel("FID (Lucid Inception-v3, 5k vs train)", fontsize=9)
        else:
            note = f"FID every {self.fid_every} epochs" if self.fid_every else "FID off"
            ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=9, color="0.5")
            ax.set_yticks([])
        twin = ax.twinx()
        x, y = _by_epoch(epochs, probe)
        twin.plot(x, y, color="tab:green", ls="--", marker=".", lw=1.2,
                  label=f"held-out FM error ({'θ**' if ema else 'θ'})")  # fmt: skip
        twin.set_ylabel("flow-matching MSE on test images", color="tab:green", fontsize=9)
        _finish(ax, title, "epoch", twin)

    @staticmethod
    def _optimisation(ax: Axes, dyn: list[dict[str, Any]]) -> None:
        title = "optimisation: gradient norm · update ratio"
        if not dyn:
            return _empty(ax, title, "after the first dynamics step")
        x = [r["global_step"] for r in dyn]
        grad = _positive([num(r["grad_norm"].get("total")) for r in dyn])
        ax.plot(x, grad, color="0.6", alpha=0.5, lw=0.8)
        ax.plot(x, smooth(grad, 0.8), color="black", lw=1.6, label="global ‖∇θ‖")
        ax.set_yscale("log")
        ax.set_ylabel("gradient norm", fontsize=9)
        twin = ax.twinx()
        ratio = _positive([num(r.get("update_ratio")) for r in dyn])
        twin.plot(x, ratio, color="tab:blue", lw=1.2, label="‖Δθ‖ / ‖θ‖ per step")
        twin.set_yscale("log")
        twin.set_ylabel("update ratio", color="tab:blue", fontsize=9)
        _finish(ax, title, "optimiser step", twin)

    def _consistency(self, ax: Axes, epochs: list[dict[str, Any]]) -> None:
        """Do few-step samples land where many-step ones do, with the data's spread?"""
        title = "consistency: flow-map gap · sample std"
        gaps = sorted({k for r in epochs for k in r if k.startswith("flowmap_gap_")})
        stds = sorted({k for r in epochs for k in r if k.startswith("sample_std_nfe")})
        if not gaps and not stds:
            return _empty(ax, title, "after the first epoch")
        for key, colour in zip(gaps, _NFE_COLOURS, strict=False):
            x, y = _by_epoch(epochs, key)
            k, n = key.removeprefix("flowmap_gap_").split("v")
            ax.plot(x, _positive(y), color=colour, marker="o", ms=4, lw=1.6,
                    label=f"MSE({k}-step, {n}-step)")  # fmt: skip
        ax.set_yscale("log")
        ax.set_ylabel("flow-map gap (MSE, same noise)", fontsize=9)
        twin = ax.twinx()
        for key, colour in zip(stds, _NFE_COLOURS, strict=False):
            x, y = _by_epoch(epochs, key)
            twin.plot(x, y, color=colour, ls="--", lw=1.0, alpha=0.8,
                      label=f"sample std, {key.removeprefix('sample_std_nfe')} NFE")  # fmt: skip
        twin.axhline(self.data_std, color="black", ls=":", lw=1.0, label="data std")
        twin.set_ylabel("sample std", fontsize=9)
        _finish(ax, title, "epoch", twin)

    # ── the denoising path ───────────────────────────────────────────────

    @staticmethod
    def _path(ax: Axes, path: np.ndarray, classes: tuple[str, ...]) -> None:
        """Rows are classes, columns are the states at t = 1 … 0, 2-pixel gaps."""
        n_t, n_cls = int(path.shape[0]), int(path.shape[1])
        gap, side = 2, 32
        canvas = np.ones((n_cls * (side + gap) - gap, n_t * (side + gap) - gap, 3))
        pixels = np.clip((path.transpose(0, 1, 3, 4, 2) + 1.0) / 2.0, 0.0, 1.0)
        for j in range(n_t):
            for i in range(n_cls):
                top, left = i * (side + gap), j * (side + gap)
                canvas[top : top + side, left : left + side] = pixels[j, i]
        ax.imshow(canvas, interpolation="nearest")
        ax.set_xticks([j * (side + gap) + side / 2 for j in range(n_t)])
        ax.set_xticklabels([f"t={1 - j / (n_t - 1):.3g}" for j in range(n_t)], fontsize=8)
        ax.set_yticks([i * (side + gap) + side / 2 for i in range(n_cls)])
        ax.set_yticklabels(classes, fontsize=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
