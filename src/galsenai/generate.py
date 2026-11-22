"""Autoregressive sampling with a KV cache."""

from __future__ import annotations

from collections.abc import Collection

import torch
from torch import Tensor

from galsenai.model import GPT


def filter_logits(logits: Tensor, top_k: int | None, top_p: float | None) -> Tensor:
    """Mask logits outside the top-k set and the top-p nucleus (the top token always stays)."""
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        drop = probs.cumsum(dim=-1) - probs > top_p  # mass before this token already covers top_p
        logits = torch.empty_like(logits).scatter(-1, order, sorted_logits.masked_fill(drop, float("-inf")))
    return logits


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_ids: Collection[int] = (),
    generator: torch.Generator | None = None,
) -> Tensor:
    """Extend a (1, T) prompt by up to max_new_tokens, stopping after an EOS token.

    ``temperature=0`` is greedy decoding. Generation also stops at ``block_size`` tokens.
    """
    if prompt.ndim != 2 or prompt.size(0) != 1:
        raise ValueError("prompt must have shape (1, T)")
    if temperature < 0 or max_new_tokens < 0:
        raise ValueError("temperature and max_new_tokens must be >= 0")

    was_training = model.training
    model.eval()
    try:
        tokens = prompt
        out = model(prompt)
        for _ in range(max_new_tokens):
            if tokens.size(1) >= model.cfg.block_size:
                break
            logits = out.logits[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = filter_logits(logits / temperature, top_k, top_p).softmax(dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            tokens = torch.cat((tokens, nxt), dim=1)
            if nxt.item() in eos_ids:
                break
            out = model(nxt, past=out.past)
        return tokens
    finally:
        model.train(was_training)
