"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNGs (and request deterministic algorithms if desired)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
