"""CIFAR-10: download, verify, load, and deterministic, resumable batching.

Every batch is a pure function of ``(seed, epoch, batch index)`` — the
epoch's permutation and each batch's flip mask come from generators seeded
with those numbers — so a run resumed in the middle of an epoch sees
exactly the batches it would have seen without the interruption.
"""

import hashlib
import pickle
import tarfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)  # fmt: skip


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cifar10(root: Path) -> Path:
    """The extracted ``cifar-10-batches-py`` folder, downloading it if absent."""
    folder = root / "cifar-10-batches-py"
    if (folder / "data_batch_1").exists():
        return folder
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-10-python.tar.gz"
    if not archive.exists():
        print(f"downloading CIFAR-10 to {archive} ...")
        urllib.request.urlretrieve(URL, archive)
    if _md5(archive) != MD5:
        raise RuntimeError(f"{archive} failed its md5 check; delete it and retry")
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    return folder


def _read(folder: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for name in names:
        with open(folder / name, "rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        images.append(batch[b"data"])
        labels.extend(batch[b"labels"])
    x = np.concatenate(images).reshape(-1, 3, 32, 32)
    return x, np.asarray(labels, dtype=np.int64)


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """``uint8 (N, 3, 32, 32)`` images and ``int64`` labels of one split."""
    folder = ensure_cifar10(root)
    if split == "train":
        return _read(folder, [f"data_batch_{i}" for i in range(1, 6)])
    if split == "test":
        return _read(folder, ["test_batch"])
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")


def to_unit(images: np.ndarray) -> np.ndarray:
    """``uint8`` pixels to ``float32`` in ``[-1, 1]``."""
    return images.astype(np.float32) / 127.5 - 1.0


class EpochLoader:
    """Deterministic shuffled batches, one epoch at a time.

    Parameters
    ----------
    images, labels : numpy arrays
        The training split.
    batch_size : int
        Images per batch; the last partial batch of an epoch is dropped.
    seed : int
        Run seed.
    flip : bool
        Mirror each image horizontally with probability 1/2.
    """

    def __init__(
        self, images: np.ndarray, labels: np.ndarray, batch_size: int, seed: int, flip: bool
    ) -> None:
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.seed = seed
        self.flip = flip

    @property
    def steps_per_epoch(self) -> int:
        return len(self.images) // self.batch_size

    def epoch(self, epoch: int, start: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Batches ``start, start + 1, ...`` of ``epoch``."""
        order = np.random.default_rng([self.seed, epoch]).permutation(len(self.images))
        for index in range(start, self.steps_per_epoch):
            take = order[index * self.batch_size : (index + 1) * self.batch_size]
            images = to_unit(self.images[take])
            if self.flip:
                rng = np.random.default_rng([self.seed, epoch, index])
                mirrored = rng.random(len(take)) < 0.5
                images[mirrored] = images[mirrored, :, :, ::-1]
            yield images, self.labels[take]
