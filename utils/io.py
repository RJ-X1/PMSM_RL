"""I/O placeholders for checkpoints and experiment artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist model/training state.

    TODO: Decide on serialization format (`torch.save` vs alternatives) and
    checkpoint schema once training internals are finalized.
    """
    _ = Path(path)
    _ = payload
    raise NotImplementedError('TODO: implement checkpoint saving')


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load model/training state from a checkpoint file."""
    _ = Path(path)
    # TODO: Implement deserialization matching `save_checkpoint` schema.
    raise NotImplementedError('TODO: implement checkpoint loading')
