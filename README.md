# galsenai

A small, readable decoder-only transformer (RMSNorm, RoPE, grouped-query attention, SwiGLU, KV cache,
optional sliding-window layers, LoRA) meant to be trained from scratch and run on device.

Status: the training-side library is tested; on-device export is **not implemented yet** (see Roadmap).

## Install

```bash
pip install -e ".[dev]"
pytest -q && ruff check . && mypy
```

## Use

```python
import torch
from galsenai import GPT, ModelConfig, generate

model = GPT(ModelConfig(vocab_size=16_394, n_layer=8, n_head=8, n_kv_head=2, n_embd=512))
print(f"{model.num_params() / 1e6:.1f}M parameters")  # 30.9M with this config

prompt = torch.tensor([[1, 2, 3]])
tokens = generate(model, prompt, max_new_tokens=20, temperature=0.8, top_k=50)
```

Chat with a trained checkpoint and a minbpe tokenizer:

```bash
python -m galsenai.chat --checkpoint run.pth --tokenizer tokenizer.model
```

## Design

| Choice | Why |
|---|---|
| RoPE, no learned positions | Relative positions; no table tied to the training length |
| Grouped-query attention (`n_kv_head`) | KV cache shrinks by `n_head / n_kv_head`, the main memory cost on a phone |
| Sliding-window layers (`sliding_window`, `global_every`) | Bounded attention cost, interleaved with global layers as in Gemma 3 |
| RMSNorm, SwiGLU, tied embeddings, no biases | Fewer parameters and ops per layer |
| KV-cache decoding | One new token costs O(1) projections, not a full forward pass |

The structure (one config, fused QKV, `scaled_dot_product_attention`) follows
[nanoGPT](https://github.com/karpathy/nanoGPT), which is deprecated upstream in favour of nanochat and ships
GPT-2 components (LayerNorm, learned positions, GELU, full multi-head attention). This package keeps its
compactness but uses the components on-device models rely on.

## Tests

`tests/` checks properties, not implementation: causality, KV-cache equals full forward (full, GQA, MQA, sliding
window), the sliding-window receptive field, RoPE relative-position invariance, parameter count against a closed
form, loss masking, memorization of a short sequence, sampling filters, checkpoint round trip, LoRA identity at
initialization.

## Data preparation and privacy

`data/prepare_data_galesnai.py` removes WhatsApp system lines and replaces every phone number with a keyed short
hash (`TEL_ab12cd34`). The key and the token-to-number table live in `data/private/` and never leave the machine.
Raw exports, media and backups are git-ignored. Display names are **not** anonymized.

```bash
python data/prepare_data_galesnai.py --dry-run   # count only
python data/prepare_data_galesnai.py             # backup, filter, pseudonymize
```

## Roadmap

1. Train and evaluate a first checkpoint on the cleaned data (perplexity on a held-out split).
2. Export: weight quantization (int8/int4), then LiteRT or GGUF, with an accuracy check against the PyTorch model.
3. Measure on-device memory and tokens/second on a real phone.

## Attribution

`minbpe/` is Andrej Karpathy's tokenizer (MIT, see `minbpe/LICENSE`). The course notebooks this project grew
from come from Imad Saddik's "Train your language model" course; check its license before redistributing any
course material.
