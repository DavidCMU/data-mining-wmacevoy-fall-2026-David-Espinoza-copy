"""
MNIST: download once, cache as one .npz, hand back NumPy arrays.

The raw files are in IDX format: a big-endian header (magic number, then the
size of each dimension) followed by raw uint8 bytes. Parsing that is ten lines,
so we do it ourselves rather than pull in torchvision for one dataset.

    python mnist_data.py        # download into data/mnist.npz and summarize
"""
import gzip
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
CACHE = DATA_DIR / "mnist.npz"

# Yann LeCun's original site now often refuses downloads; these mirrors carry
# byte-identical files. Tried in order.
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}

# Pixel mean and std over the 60,000 training images (pixels scaled to [0, 1]).
# Standardizing with these is the usual first step for MNIST.
MEAN = 0.1307
STD = 0.3081


def parse_idx(raw: bytes) -> np.ndarray:
    """Decode an (ungzipped) IDX file into a uint8 array of the stated shape."""
    zero, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    if zero != 0 or dtype_code != 0x08:
        raise ValueError(f"not a uint8 IDX file (magic {raw[:4].hex()})")
    shape = struct.unpack(f">{ndim}I", raw[4:4 + 4 * ndim])
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"IDX header says {shape} but found {data.size} values")
    return data.reshape(shape)


def _fetch(name: str) -> bytes:
    errors = []
    for base in MIRRORS:
        try:
            with urllib.request.urlopen(base + name, timeout=60) as resp:
                return gzip.decompress(resp.read())
        except OSError as e:           # URLError, timeouts, bad gzip
            errors.append(f"{base}: {e}")
    raise RuntimeError(f"could not download {name}:\n  " + "\n  ".join(errors))


def download(path: Path = CACHE) -> Path:
    """Fetch all four files and save them as one compressed .npz."""
    arrays = {}
    for key, name in FILES.items():
        print(f"downloading {name} ...", file=sys.stderr)
        arrays[key] = parse_idx(_fetch(name))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load(path: Path = CACHE) -> dict[str, np.ndarray]:
    """Return x_train (60000, 28, 28) uint8, y_train (60000,), x_test, y_test."""
    if not path.exists():
        download(path)
    with np.load(path) as f:
        return {k: f[k] for k in FILES}


def split(x: np.ndarray, y: np.ndarray, val_size: int, seed: int = 0):
    """Shuffle once with a fixed seed and carve off a validation set.

    The test set is never touched during training; validation is what we
    watch to decide when the model has stopped generalizing.
    """
    idx = np.random.default_rng(seed).permutation(len(x))
    val, train = idx[:val_size], idx[val_size:]
    return (x[train], y[train]), (x[val], y[val])


def standardize(x: np.ndarray) -> np.ndarray:
    """uint8 (N, 28, 28) -> float32 (N, 1, 28, 28), zero mean and unit std."""
    return ((x.astype(np.float32) / 255.0 - MEAN) / STD)[:, None, :, :]


if __name__ == "__main__":
    d = load()
    for k, v in d.items():
        print(f"{k:8s} {str(v.shape):16s} {v.dtype}")
    print("train label counts:", np.bincount(d["y_train"]).tolist())
