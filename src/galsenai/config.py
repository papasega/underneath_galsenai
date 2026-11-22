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
    # nanochat-style switches (all off by default = the Llama/Gemma recipe above)
    mlp: str = "swiglu"  # "swiglu" or "relu2" (relu(x)**2, 4x hidden)
    qk_norm: bool = False  # parameter-free RMSNorm on queries and keys after RoPE
    norm_affine: bool = True  # False = RMSNorm without learnable scale
    logit_softcap: float | None = None  # logits = c * tanh(logits / c)
    layer_lambdas: bool = False  # learnable per-layer residual scale and x0 (embedding) skip

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
        if self.mlp not in ("swiglu", "relu2"):
            raise ValueError("mlp must be 'swiglu' or 'relu2'")
        if self.logit_softcap is not None and self.logit_softcap <= 0:
            raise ValueError("logit_softcap must be > 0")
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
        if self.mlp == "relu2":
            return 4 * self.n_embd
        return -(-int(8 * self.n_embd / 3) // 64) * 64

    @classmethod
    def nanochat(cls, vocab_size: int, n_head: int = 8, **overrides: Any) -> ModelConfig:
        """Preset following karpathy/nanochat's GPT: relu^2 MLP, QK-norm, parameter-free RMSNorm, untied
        embeddings, logit soft-cap 15, RoPE base 1e5, SSSL sliding windows, residual/x0 lambdas, full MHA.
        Not ported: value embeddings, smear gate, FP8, flash-attention (see README)."""
        defaults: dict[str, Any] = dict(
            mlp="relu2",
            qk_norm=True,
            norm_affine=False,
            tie_embeddings=False,
            logit_softcap=15.0,
            layer_lambdas=True,
            rope_base=100_000.0,
            sliding_window=256,
            global_every=4,
            n_kv_head=n_head,
        )
        return cls(vocab_size=vocab_size, n_head=n_head, **{**defaults, **overrides})

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
