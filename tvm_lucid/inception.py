"""Inception-v3 features and class probabilities, from Lucid's model zoo.

The network is ``lucid.models.inception_v3_cls`` with its ImageNet weights:
the 2048-wide pooled features feed FID, the softmax feeds the Inception
Score.  It is **not** the TF-Inception graph published FIDs use, so these
numbers compare runs of this project with each other — the purpose here —
and not with tables in papers.
"""

from pathlib import Path

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.nn.functional as F
import numpy as np
from lucid import Tensor
from tqdm import tqdm

from tvm_lucid.data import to_unit
from tvm_lucid.stats import GaussianAccumulator

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class InceptionV3:
    """Callable: ``[-1, 1]`` images → ``(pooled features, class probabilities)``."""

    def __init__(self, device: str = "metal") -> None:
        self.device = device
        self.net = M.inception_v3_cls(pretrained=True).to(device).eval()
        self._pooled: Tensor | None = None
        self.net.avgpool.register_forward_hook(self._keep)
        self._mean = lucid.tensor(list(_MEAN), device=device).reshape(1, 3, 1, 1)
        self._std = lucid.tensor(list(_STD), device=device).reshape(1, 3, 1, 1)

    def _keep(self, _module: nn.Module, _inputs: object, output: Tensor) -> None:
        self._pooled = output

    def __call__(self, images: Tensor) -> tuple[np.ndarray, np.ndarray]:
        with lucid.no_grad():
            x = (images.clamp(-1.0, 1.0) + 1.0) / 2.0
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            logits = self.net((x - self._mean) / self._std).logits
            probs = F.softmax(logits, dim=-1)
        assert self._pooled is not None
        features = self._pooled.flatten(1).to("cpu").numpy().astype(np.float64)
        return features, probs.to("cpu").numpy().astype(np.float64)


def reference_stats(
    inception: InceptionV3, images: np.ndarray, cache: Path, batch: int = 250
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the real images' features, cached to ``cache``.

    Computed once over the whole training split (50k images) — the usual
    reference for CIFAR-10 FID — and reused by every run and every epoch.
    """
    if cache.exists():
        stored = np.load(cache)
        return stored["mu"], stored["sigma"]
    acc = GaussianAccumulator(2048)
    for start in tqdm(range(0, len(images), batch), desc="FID reference stats", leave=False):
        chunk = lucid.tensor(to_unit(images[start : start + batch]), device=inception.device)
        features, _ = inception(chunk)
        acc.update(features)
    mu, sigma = acc.mean_cov()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mu=mu, sigma=sigma, n=acc.n)
    return mu, sigma
