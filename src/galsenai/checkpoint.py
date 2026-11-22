"""Checkpoint I/O. Files hold only tensors and plain Python values, so they load with
``weights_only=True`` and never execute pickled code."""

from __future__ import annotations

from pathlib import Path

import torch

from galsenai.config import ModelConfig
from galsenai.model import GPT


def save_checkpoint(model: GPT, path: Path) -> None:
    """Atomically write the config and weights of a model."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": model.cfg.to_dict(), "model_state_dict": model.state_dict()}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> GPT:
    """Rebuild a model from a checkpoint written by ``save_checkpoint``."""
    data = torch.load(path, map_location=device, weights_only=True)
    model = GPT(ModelConfig.from_dict(data["config"]))
    model.load_state_dict(data["model_state_dict"])
    return model.to(device)
