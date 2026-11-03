"""
choices.py — the handful of plain-string constants that both the torch-free
dashboard process (app.py, runs.py) and the training process (train.py,
model.py) need to agree on. Kept in its own file, with zero imports, so that
importing it never pulls in torch — that's what lets app.py build its
sidebar selectboxes (model choice, etc.) without importing model.py, which
does `import torch`.
"""
from __future__ import annotations

MODEL_CHOICES = ("logreg", "mlp")
