"""Model configuration with validation at construction time."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters of a decoder-only transformer (Llama/Gemma family).

    ``n_kv_head == n_head`` is multi-head attention, ``1`` is multi-query attention,
    anything in between is grouped-query attention (smaller KV cache on device).
    ``sliding_window`` turns on local attention; every ``global_every``-th layer stays
    global (Gemma 3 style interleaving).
    """

    vocab_size: int
    block_size: int = 1024
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 2
    n_embd: int = 512
    ffn_hidden: int | None = None
    dropout: float = 0.0
    rope_base: float = 10_000.0
    norm_eps: float = 1e-6
    sliding_window: int | None = None
    global_every: int = 6
    tie_embeddings: bool = True
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_kv_head, self.n_embd) < 1:
            raise ValueError("vocab_size, block_size, n_layer, n_head, n_kv_head and n_embd must be >= 1")
        if self.n_embd % self.n_head:
            raise ValueError(f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}")
        if self.n_head % self.n_kv_head:
            raise ValueError(f"n_head={self.n_head} must be divisible by n_kv_head={self.n_kv_head}")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be >= 1")
        if self.global_every < 1:
            raise ValueError("global_every must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def hidden_dim(self) -> int:
        """SwiGLU hidden size: 8/3 * n_embd rounded up to a multiple of 64 unless set."""
        if self.ffn_hidden is not None:
            return self.ffn_hidden
        return -(-int(8 * self.n_embd / 3) // 64) * 64

    def window_for_layer(self, layer: int) -> int | None:
        """Attention window of a layer, or None for global attention."""
        if self.sliding_window is None or (layer + 1) % self.global_every == 0:
            return None
        return self.sliding_window

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(**data)
