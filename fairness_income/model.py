"""
model.py — two binary classifiers for the Adult income task, mirroring the
cnn/mlp choice in ml/model.py (there: image architectures; here: tabular
model capacity).

Both output raw logits (no sigmoid) — train.py applies
BCEWithLogitsLoss, which is numerically more stable than sigmoid+BCE
separately, and runs.py/app.py apply sigmoid only when turning logits into
a probability for display or thresholding.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from choices import MODEL_CHOICES


class LogisticRegression(nn.Module):
    """The simplest honest baseline: one linear layer. Good for checking
    whether a mitigation's effect survives even with no model capacity to
    spare, and fast enough to sanity-check the whole pipeline quickly."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MLP(nn.Module):
    """Two hidden layers with dropout — enough capacity to pick up
    interactions between features that a linear model can't."""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_model(name: str, n_features: int, hidden: int = 64, dropout: float = 0.2) -> nn.Module:
    if name == "logreg":
        return LogisticRegression(n_features)
    if name == "mlp":
        return MLP(n_features, hidden=hidden, dropout=dropout)
    raise ValueError(f"Unknown model {name!r}; choices are {MODEL_CHOICES}")
