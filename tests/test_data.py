"""Batching is deterministic and a mid-epoch resume sees the same batches."""

import numpy as np

from tvm_lucid.data import EpochLoader, to_unit


def _toy(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(n * 3 * 32 * 32) % 251).astype(np.uint8).reshape(n, 3, 32, 32)
    return x, np.arange(n, dtype=np.int64) % 10


def test_batches_cover_an_epoch_once() -> None:
    x, y = _toy()
    loader = EpochLoader(x, y, batch_size=8, seed=0, flip=False)
    seen = np.concatenate([labels for _, labels in loader.epoch(0)])
    assert loader.steps_per_epoch == 5 and len(seen) == 40


def test_resume_mid_epoch_reproduces_the_rest() -> None:
    x, y = _toy()
    loader = EpochLoader(x, y, batch_size=8, seed=7, flip=True)
    full = list(loader.epoch(3))
    resumed = list(loader.epoch(3, start=2))
    for (a, la), (b, lb) in zip(full[2:], resumed, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(la, lb)


def test_epochs_differ_and_flips_are_mirrors() -> None:
    x, y = _toy()
    loader = EpochLoader(x, y, batch_size=8, seed=0, flip=True)
    first = next(loader.epoch(0))[0]
    second = next(loader.epoch(1))[0]
    assert not np.array_equal(first, second)
    # every image is its source or the source mirrored left-right
    images, labels = next(loader.epoch(0))
    sources = to_unit(x)
    for image in images:
        assert any(
            np.array_equal(image, s) or np.array_equal(image, s[:, :, ::-1]) for s in sources
        )
    assert images.min() >= -1.0 and images.max() <= 1.0
