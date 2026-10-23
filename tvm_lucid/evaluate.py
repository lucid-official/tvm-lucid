"""End-of-epoch evaluation: what the model generates and how well.

Four measurements, all on fixed inputs so epochs compare like with like:

* **Sample grids** — 10x10, one class per row, from fixed noise, at every
  NFE in ``eval.nfe``.  The same latent at every epoch, so a cell's
  evolution is the model's evolution.
* **Flow-map consistency** — the mean squared distance between the one-
  (or two-) step sample and the many-step one from the same noise.  TVM
  trains the jump from ``t`` to ``s`` to land where the ODE would; a large
  gap means the one-step map has not caught up with the field (the
  displacement error of Eq. 5, measured on the model's own trajectory).
* **Per-time velocity probe** — unconditional Flow Matching error
  ``||u(x_s, s) − (ε − x)||²`` on held-out test images, in ``probe_buckets``
  bands of ``s``, for the live and the evaluation network.  The terminal
  term is only as good as the velocity it is matched to; this shows where
  in time that velocity is still wrong.
* **FID and Inception Score** every ``fid_every_epochs`` — ``fid_samples``
  class-balanced samples per NFE in ``fid_nfe``, against the training split.
"""

import time
from pathlib import Path
from typing import Any

import lucid
import lucid.models as M
import numpy as np
from tqdm import tqdm

from tvm_lucid.config import EvalConfig
from tvm_lucid.data import to_unit
from tvm_lucid.inception import InceptionV3, reference_stats
from tvm_lucid.model import DEVICE
from tvm_lucid.stats import GaussianAccumulator, frechet_distance, inception_score


def save_grid(samples: np.ndarray, path: Path, rows: int = 10) -> None:
    """``(rows*cols, 3, 32, 32)`` in ``[-1, 1]`` → one PNG, one class per row."""
    from PIL import Image

    count = samples.shape[0]
    cols = count // rows
    pixels = ((np.clip(samples, -1.0, 1.0) + 1.0) * 127.5).round().astype(np.uint8)
    grid = pixels.reshape(rows, cols, 3, 32, 32).transpose(0, 3, 1, 4, 2)
    Image.fromarray(grid.reshape(rows * 32, cols * 32, 3)).save(path)


class Evaluator:
    """Holds the fixed inputs and the Inception network across epochs."""

    def __init__(
        self,
        cfg: EvalConfig,
        run_dir: Path,
        train_images: np.ndarray,
        test_images: np.ndarray,
        stats_cache: Path,
    ) -> None:
        self.cfg = cfg
        self.samples_dir = run_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(cfg.grid_noise_seed)
        self.grid_noise = lucid.tensor(
            rng.standard_normal((100, 3, 32, 32)).astype(np.float32), device=DEVICE
        )
        self.grid_labels = lucid.tensor(np.repeat(np.arange(10), 10), device=DEVICE)

        probe_rng = np.random.default_rng(cfg.probe_seed)
        pick = probe_rng.choice(len(test_images), size=cfg.probe_batch, replace=False)
        self.probe_x = lucid.tensor(to_unit(test_images[pick]), device=DEVICE)
        self.probe_eps = lucid.tensor(
            probe_rng.standard_normal((cfg.probe_batch, 3, 32, 32)).astype(np.float32),
            device=DEVICE,
        )
        self.probe_s = [(k + 0.5) / cfg.probe_buckets for k in range(cfg.probe_buckets)]

        self._train_images = train_images
        self._stats_cache = stats_cache
        self._inception: InceptionV3 | None = None
        self._reference: tuple[np.ndarray, np.ndarray] | None = None

    # ── grids and flow-map consistency ───────────────────────────────────

    def grids(self, model: M.TVMForImageGeneration, tag: str) -> dict[str, Any]:
        use_ema = model.config.eval_ema_decay is not None
        model.eval()
        outputs: dict[int, np.ndarray] = {}
        for nfe in self.cfg.nfe:
            out = model.generate(
                100, labels=self.grid_labels, steps=nfe, noise=self.grid_noise, use_ema=use_ema
            )
            outputs[nfe] = out.samples.to("cpu").numpy()
            save_grid(outputs[nfe], self.samples_dir / f"{tag}_nfe{nfe}.png")
        model.train()
        finest = max(outputs)
        record: dict[str, Any] = {}
        for nfe, samples in outputs.items():
            if nfe != finest:
                gap = float(((samples - outputs[finest]) ** 2).mean())
                record[f"flowmap_gap_{nfe}v{finest}"] = gap
            record[f"sample_std_nfe{nfe}"] = float(samples.std())
        return record

    # ── per-time velocity probe ──────────────────────────────────────────

    def probe(self, model: M.TVMForImageGeneration) -> dict[str, Any]:
        networks = {"live": model.tvm}
        if model.config.eval_ema_decay is not None:
            networks["ema"] = model.ema_network
        n = self.probe_x.shape[0]
        record: dict[str, Any] = {"probe_s": self.probe_s}
        with lucid.no_grad():
            for name, net in networks.items():
                net.eval()
                errors = []
                for s_value in self.probe_s:
                    s = lucid.full((n,), s_value, device=DEVICE)
                    x_s = (1.0 - s_value) * self.probe_x + s_value * self.probe_eps
                    u = net(x_s, s, s, None, 1.0).sample
                    v = self.probe_eps - self.probe_x
                    errors.append(float(((u - v) ** 2).mean().item()))
                if name == "live":
                    net.train()
                record[f"probe_fm_{name}"] = errors
                record[f"probe_fm_{name}_mean"] = float(np.mean(errors))
        return record

    # ── FID / IS ─────────────────────────────────────────────────────────

    def _ensure_inception(self) -> tuple[InceptionV3, tuple[np.ndarray, np.ndarray]]:
        if self._inception is None:
            self._inception = InceptionV3(DEVICE)
        if self._reference is None:
            self._reference = reference_stats(
                self._inception, self._train_images, self._stats_cache, self.cfg.fid_batch
            )
        return self._inception, self._reference

    def fid(self, model: M.TVMForImageGeneration) -> dict[str, Any]:
        inception, (mu_ref, sigma_ref) = self._ensure_inception()
        use_ema = model.config.eval_ema_decay is not None
        total, batch = self.cfg.fid_samples, self.cfg.sample_batch
        labels_all = np.arange(total) % 10
        record: dict[str, Any] = {"fid_samples": total}
        model.eval()
        for nfe in self.cfg.fid_nfe:
            started = time.time()
            acc = GaussianAccumulator(2048)
            probs: list[np.ndarray] = []
            for start in tqdm(
                range(0, total, batch), desc=f"FID {nfe}-NFE", leave=False, position=2
            ):
                labels = labels_all[start : start + batch]
                samples = model.generate(
                    len(labels),
                    labels=lucid.tensor(labels, device=DEVICE),
                    steps=nfe,
                    use_ema=use_ema,
                ).samples
                features, p = inception(samples)
                acc.update(features)
                probs.append(p)
            mu, sigma = acc.mean_cov()
            record[f"fid_nfe{nfe}"] = frechet_distance(mu, sigma, mu_ref, sigma_ref)
            is_mean, is_std = inception_score(np.concatenate(probs))
            record[f"is_nfe{nfe}"] = is_mean
            record[f"is_std_nfe{nfe}"] = is_std
            record[f"fid_seconds_nfe{nfe}"] = round(time.time() - started, 1)
        model.train()
        return record
