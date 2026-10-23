"""Fréchet distance and Inception Score — the numerics, with no model in sight.

Kept apart from the Inception wrapper so they can be tested without a GPU.
Everything is float64.
"""

import numpy as np


class GaussianAccumulator:
    """Running mean and covariance of feature batches, in one pass.

    Sums of features and of their outer products are accumulated in
    float64, which for 2048-wide features over 50k samples keeps the
    covariance accurate to well below what FID can resolve.
    """

    def __init__(self, dim: int) -> None:
        self.n = 0
        self._sum = np.zeros(dim, dtype=np.float64)
        self._outer = np.zeros((dim, dim), dtype=np.float64)

    def update(self, features: np.ndarray) -> None:
        f = np.asarray(features, dtype=np.float64)
        self.n += f.shape[0]
        self._sum += f.sum(axis=0)
        self._outer += f.T @ f

    def mean_cov(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean and unbiased covariance."""
        if self.n < 2:
            raise ValueError("need at least two samples for a covariance")
        mu = self._sum / self.n
        cov = (self._outer - self.n * np.outer(mu, mu)) / (self.n - 1)
        return mu, cov


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Symmetric square root of a symmetric positive semi-definite matrix."""
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    r"""FID between two Gaussians.

    .. math::

        \lVert\mu_1 - \mu_2\rVert^2 + \operatorname{Tr}(\Sigma_1 + \Sigma_2
        - 2(\Sigma_1\Sigma_2)^{1/2})

    The trace of the non-symmetric :math:`(\Sigma_1\Sigma_2)^{1/2}` equals
    the sum of square roots of the eigenvalues of the symmetric
    :math:`\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}`, which is computed instead:
    no complex arithmetic, no ``sqrtm`` iteration to stall.
    """
    diff = mu1 - mu2
    root = _psd_sqrt(sigma1)
    inner = root @ sigma2 @ root
    eig = np.linalg.eigvalsh((inner + inner.T) / 2)
    trace_sqrt = float(np.sqrt(np.clip(eig, 0.0, None)).sum())
    value = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * trace_sqrt)
    return max(value, 0.0)


def inception_score(probs: np.ndarray, splits: int = 10) -> tuple[float, float]:
    """Mean and standard deviation of ``exp(E[KL(p(y|x) || p(y))])`` over splits."""
    probs = np.asarray(probs, dtype=np.float64)
    scores = []
    for part in np.array_split(probs, splits):
        marginal = part.mean(axis=0, keepdims=True)
        kl = (part * (np.log(part + 1e-12) - np.log(marginal + 1e-12))).sum(axis=1)
        scores.append(float(np.exp(kl.mean())))
    return float(np.mean(scores)), float(np.std(scores))
