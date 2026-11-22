"""Compact decoder-only transformer: RMSNorm, RoPE, grouped-query attention, SwiGLU (or nanochat's relu^2).

Structure follows nanoGPT (one config, fused QKV, ``scaled_dot_product_attention``,
tied embeddings) with the components on-device models such as Gemma rely on, plus a
KV cache so decoding costs O(1) new work per token.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from galsenai.config import ModelConfig

KVCache = tuple[Tensor, Tensor]  # keys and values, each (batch, n_kv_head, seq, head_dim)


class Output(NamedTuple):
    logits: Tensor
    loss: Tensor | None
    past: list[KVCache]


def rotary_cos_sin(positions: Tensor, head_dim: int, base: float) -> tuple[Tensor, Tensor]:
    """cos/sin tables of shape (len(positions), head_dim // 2), computed in float32."""
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    angles = torch.outer(positions.float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate pairs (x[..., :d/2], x[..., d/2:]) by position-dependent angles."""
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, window: int | None) -> None:
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.window, self.dropout = window, cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, (cfg.n_head + 2 * cfg.n_kv_head) * cfg.head_dim, bias=False)
        self.proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.n_embd, bias=False)
        self.qk_norm, self.eps = cfg.qk_norm, cfg.norm_eps

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, past: KVCache | None) -> tuple[Tensor, KVCache]:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(
            [self.n_head * self.head_dim, self.n_kv_head * self.head_dim, self.n_kv_head * self.head_dim], dim=-1
        )
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if self.qk_norm:
            q, k = F.rms_norm(q, (self.head_dim,), eps=self.eps), F.rms_norm(k, (self.head_dim,), eps=self.eps)

        offset = 0
        if past is not None:
            offset = past[0].size(2)
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        cache = (k, v)

        group = self.n_head // self.n_kv_head
        if group > 1:  # repeat KV heads (portable across exporters, unlike enable_gqa)
            k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)

        dropout_p = self.dropout if self.training else 0.0
        if offset == 0 and self.window is None:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            q_pos = torch.arange(offset, offset + T, device=x.device)[:, None]
            k_pos = torch.arange(offset + T, device=x.device)[None, :]
            mask = k_pos <= q_pos
            if self.window is not None:
                mask &= k_pos > q_pos - self.window
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        return self.proj(y.transpose(1, 2).reshape(B, T, -1)), cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.n_embd, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class ReLU2(nn.Module):
    """nanochat's MLP: down(relu(up(x))**2). Same attribute names as SwiGLU so init rules apply."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(cfg.n_embd, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.down(F.relu(self.up(x)).square()))


def make_norm(cfg: ModelConfig) -> nn.RMSNorm:
    return nn.RMSNorm(cfg.n_embd, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = Attention(cfg, cfg.window_for_layer(layer))
        self.norm2 = make_norm(cfg)
        self.mlp = ReLU2(cfg) if cfg.mlp == "relu2" else SwiGLU(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, past: KVCache | None) -> tuple[Tensor, KVCache]:
        h, cache = self.attn(self.norm1(x), cos, sin, past)
        x = x + h
        return x + self.mlp(self.norm2(x)), cache


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        if cfg.layer_lambdas:  # nanochat init: residual scale 1.15 -> 1.05, x0 skip 0.20 -> 0.05 with depth
            frac = torch.linspace(0, 1, cfg.n_layer)
            self.resid_lambdas = nn.Parameter(1.15 - 0.10 * frac)
            self.x0_lambdas = nn.Parameter(0.20 - 0.15 * frac)
        # GPT-2 scaling keeps the residual stream variance stable as depth grows.
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Unique parameters (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: Tensor, targets: Tensor | None = None, past: list[KVCache] | None = None) -> Output:
        _, T = idx.shape
        offset = 0 if past is None else past[0][0].size(2)
        if offset + T > self.cfg.block_size:
            raise ValueError(f"sequence length {offset + T} exceeds block_size {self.cfg.block_size}")
        cos, sin = rotary_cos_sin(
            torch.arange(offset, offset + T, device=idx.device), self.cfg.head_dim, self.cfg.rope_base
        )

        x = x0 = self.drop(self.tok_emb(idx))
        cache: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            if self.cfg.layer_lambdas:
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x, kv = block(x, cos, sin, None if past is None else past[i])
            cache.append(kv)
        logits = self.lm_head(self.norm(x))
        if (c := self.cfg.logit_softcap) is not None:
            logits = c * torch.tanh(logits.float() / c)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=self.cfg.ignore_index)
        return Output(logits, loss, cache)
