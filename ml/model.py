"""
Two networks for MNIST, small enough to train on a laptop CPU in minutes.

    mlp   flatten -> 256 -> 128 -> 10        ~235k weights, ~97-98% test
    cnn   two conv+pool blocks -> 128 -> 10  ~206k weights, ~99% test

The CNN wins with fewer weights because a convolution reuses the same small
filter at every position: a stroke is a stroke wherever it is in the image.
"""
import torch
from torch import nn


def mlp(dropout: float = 0.2) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(128, 10),
    )


def cnn(dropout: float = 0.2) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x14x14
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x7x7
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(128, 10),
    )


ARCHITECTURES = {"cnn": cnn, "mlp": mlp}


def build(arch: str, dropout: float = 0.2) -> nn.Module:
    return ARCHITECTURES[arch](dropout)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def first_layer_filters(model: nn.Module) -> torch.Tensor:
    """What the first layer looks for, as (n, 28, 28) or (n, 3, 3) images.

    For the MLP each of the 256 hidden units has one weight per pixel, so its
    weights *are* a 28x28 picture. For the CNN they are the 3x3 conv kernels.
    """
    for layer in model:
        if isinstance(layer, nn.Conv2d):
            return layer.weight.detach()[:, 0]
        if isinstance(layer, nn.Linear):
            return layer.weight.detach().reshape(-1, 28, 28)
    raise ValueError("model has no Conv2d or Linear layer")
