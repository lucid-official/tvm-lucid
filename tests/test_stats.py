"""FID / IS numerics against independent references."""

import numpy as np
import pytest

from tvm_lucid.stats import GaussianAccumulator, frechet_distance, inception_score


def _random_cov(rng: np.random.Generator, dim: int) -> np.ndarray:
    a = rng.standard_normal((dim, dim))
    return a @ a.T / dim + 0.1 * np.eye(dim)


def test_identical_gaussians_are_at_distance_zero() -> None:
    rng = np.random.default_rng(0)
    mu, sigma = rng.standard_normal(16), _random_cov(rng, 16)
    assert frechet_distance(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-8)


def test_a_mean_shift_adds_its_squared_length() -> None:
    rng = np.random.default_rng(1)
    mu, sigma = rng.standard_normal(16), _random_cov(rng, 16)
    shift = rng.standard_normal(16)
    expected = float(shift @ shift)
    assert frechet_distance(mu, sigma, mu + shift, sigma) == pytest.approx(expected, rel=1e-8)


def test_matches_scipy_sqrtm() -> None:
    linalg = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(2)
    mu1, mu2 = rng.standard_normal(32), rng.standard_normal(32)
    s1, s2 = _random_cov(rng, 32), _random_cov(rng, 32)
    covmean = linalg.sqrtm(s1 @ s2).real
    reference = (mu1 - mu2) @ (mu1 - mu2) + np.trace(s1 + s2 - 2 * covmean)
    assert frechet_distance(mu1, s1, mu2, s2) == pytest.approx(reference, rel=1e-6)


def test_accumulator_matches_numpy_in_batches() -> None:
    rng = np.random.default_rng(3)
    features = rng.standard_normal((1000, 8)) * 3 + 1
    acc = GaussianAccumulator(8)
    for chunk in np.array_split(features, 7):
        acc.update(chunk)
    mu, cov = acc.mean_cov()
    np.testing.assert_allclose(mu, features.mean(0), rtol=1e-10)
    np.testing.assert_allclose(cov, np.cov(features, rowvar=False), rtol=1e-8, atol=1e-10)


def test_inception_score_bounds() -> None:
    uniform = np.full((100, 10), 0.1)
    assert inception_score(uniform)[0] == pytest.approx(1.0)
    confident = np.eye(10)[np.arange(1000) % 10] * (1 - 1e-6) + 1e-7
    assert inception_score(confident)[0] == pytest.approx(10.0, rel=1e-3)
