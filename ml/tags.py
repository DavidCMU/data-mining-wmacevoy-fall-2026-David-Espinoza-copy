"""
Extra labels on MNIST images, kept beside the dataset rather than in it.

A tag lives in tags/<name>.csv as rows of (split, index, label, score).
MNIST's image order is fixed by the IDX files, so an index names the same image
everywhere.

    crossed_seven   a 7 with a bar through the stem (European style)
                    label: yes | no | unsure

The labels were reviewed by eye, not computed. A pixel heuristic
(crossbar_score) only chose which 7s to review: every 7 scoring
REVIEW_MIN or more, about 1,100 of 7,293. A 7 not in the file scored lower
and counts as not crossed; none of 48 random low-scoring 7s checked by hand
was crossed. The heuristic alone is too loose to use as the tag: about 1 in 5
of the 7s it flags have a looped or curled top, not a crossbar.

    python tags.py              # summarize the tags
"""
import csv
from pathlib import Path

import numpy as np

import mnist_data

TAG_DIR = Path(__file__).parent / "tags"
INK = 100              # pixel value counted as ink
REVIEW_MIN = 3         # crossbar score that sent a 7 for review


def longest_runs(x: np.ndarray) -> np.ndarray:
    """uint8 (N, 28, 28) -> (N, 28): the longest run of ink in each row."""
    ink = x > INK
    out = np.zeros(ink.shape[:2], int)
    run = np.zeros(ink.shape[:2], int)
    for c in range(ink.shape[2]):
        run = np.where(ink[:, :, c], run + 1, 0)
        out = np.maximum(out, run)
    return out


def crossbar_score(x: np.ndarray) -> np.ndarray:
    """How much wider the widest mid-stem row is than the stem around it.

    The top bar is skipped (first quarter of the digit's height) and so is the
    foot. A crossed 7 has a row a few pixels wider than the rows 2-4 above and
    below it; a plain stem, even a slanted one, widens gradually.
    """
    runs = longest_runs(x)
    scores = np.zeros(len(x))
    for n in range(len(x)):
        rows = np.flatnonzero((x[n] > INK).any(1))
        top, bot = rows[0], rows[-1]
        h = bot - top
        L = runs[n]
        for r in range(top + max(5, h // 4), bot - max(3, h // 5) + 1):
            lo = max(top + 3, r - 4)
            above = L[lo:r - 1].min() if r - 1 > lo else L[r]
            below = L[r + 2:r + 5].min() if r + 2 < 28 else L[r]
            scores[n] = max(scores[n], L[r] - max(above, below))
    return scores


def read(name: str, tag_dir: Path = TAG_DIR) -> list[dict]:
    with open(tag_dir / f"{name}.csv") as f:
        return list(csv.DictReader(f))


def load(name: str, split: str, n: int, labels=("yes",), tag_dir: Path = TAG_DIR) -> np.ndarray:
    """Boolean mask over the n images of `split`, True where `name` has one of `labels`."""
    mask = np.zeros(n, bool)
    mask[[int(r["index"]) for r in read(name, tag_dir)
          if r["split"] == split and r["label"] in labels]] = True
    return mask


if __name__ == "__main__":
    data = mnist_data.load()
    rows = read("crossed_seven")
    for split in ("train", "test"):
        n7 = int((data[f"y_{split}"] == 7).sum())
        got = [r["label"] for r in rows if r["split"] == split]
        counts = ", ".join(f"{k} {got.count(k)}" for k in ("yes", "no", "unsure"))
        print(f"crossed_seven {split:5} {got.count('yes'):4} of {n7} sevens crossed "
              f"({got.count('yes') / n7:.0%}); reviewed {len(got)}: {counts}")
