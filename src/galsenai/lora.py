"""Low-rank adaptation (LoRA) of the linear layers inside the transformer blocks."""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn

from galsenai.model import GPT


class LinearWithLoRA(nn.Module):
    """Frozen linear layer plus a trainable low-rank update ``alpha/rank * x A B``."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.linear = linear
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.randn(linear.in_features, rank) / math.sqrt(rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, linear.out_features))  # zero: starts as identity

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + self.scale * (x @ self.lora_a @ self.lora_b)


def _wrap_linears(module: nn.Module, rank: int, alpha: float) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearWithLoRA(child, rank, alpha))
        else:
            _wrap_linears(child, rank, alpha)


def get_lora_model(model: GPT, rank: int = 4, alpha: float = 8.0) -> GPT:
    """Return a deep copy of the model with LoRA on every block Linear; only LoRA params train.

    The embedding and the (tied) output head stay frozen and unwrapped.
    """
    lora_model = copy.deepcopy(model)
    _wrap_linears(lora_model.blocks, rank, alpha)
    for name, param in lora_model.named_parameters():
        param.requires_grad = "lora_" in name
    return lora_model


def trainable_fraction(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    params = list(model.parameters())
    return sum(p.numel() for p in params if p.requires_grad), sum(p.numel() for p in params)
