"""
Rebalance crossed and plain 7s in the training set.

Only about 1 in 8 training 7s is crossed (tags/crossed_seven.csv), and the
model gets crossed 7s wrong several times as often as plain ones. Each epoch,
train.py can add extra crossed 7s until the 7 pool is half crossed, half plain:

    none        no extras: the baseline, no bias correction
    oversample  extras are crossed 7s repeated at random
    augment     extras are crossed 7s each randomly shifted (up to 2 px),
                rotated (up to 10 degrees) and scaled (0.9-1.1x), so the
                network sees new variants rather than the same images again

Either way the extras make 7 a more common class overall, which can push the
model to call other digits 7: watch the false 7s, not just the crossed ones.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import mnist_data
import tags

BALANCE = ("none", "oversample", "augment")
SHIFT = 2              # pixels
ROTATE = 10            # degrees
SCALE = 0.1            # +/- fraction


def crossed_mask(train_index: np.ndarray, n_total: int) -> np.ndarray:
    """True where the training image at each position of train_index is a crossed 7."""
    return tags.load("crossed_seven", "train", n_total)[train_index]


def n_extra(y: np.ndarray, crossed: np.ndarray) -> int:
    """Extra crossed 7s needed to make the 7 pool 50:50."""
    n_crossed = int(crossed.sum())
    n_plain = int((y == 7).sum()) - n_crossed
    return max(0, n_plain - n_crossed) if n_crossed else 0


def jitter(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random shift, rotation and scale of standardized (N, 1, 28, 28) images."""
    n = len(x)

    def uniform(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=gen)

    angle = uniform(-ROTATE, ROTATE) * math.pi / 180
    scale = uniform(1 - SCALE, 1 + SCALE)
    shift = uniform(-SHIFT, SHIFT), uniform(-SHIFT, SHIFT)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos, -sin, shift[0] * 2 / 28], 1),
        torch.stack([sin, cos, shift[1] * 2 / 28], 1),
    ], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    # Shift so the background is 0, which is what grid_sample pads with.
    bg = -mnist_data.MEAN / mnist_data.STD
    return F.grid_sample(x - bg, grid, align_corners=False) + bg


def extras(x_crossed: torch.Tensor, n: int, kind: str, gen: torch.Generator) -> torch.Tensor:
    """n extra crossed 7s drawn (with replacement) from x_crossed, standardized."""
    pick = torch.randint(len(x_crossed), (n,), generator=gen)
    x = x_crossed[pick]
    return jitter(x, gen) if kind == "augment" else x
