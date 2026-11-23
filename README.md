# GalsenAI Underneath
### Unlocking the knowledge hidden in conversations.
### From conversations to collective intelligence.
##### WhatsApp Discussions Questions/Réponses

<p align="center">
  <img src="assets/wordcloud.png" alt="Word cloud of the anonymized chat, in the shape of Senegal" width="640">
</p>

Turn the GalsenAI "Questions/Réponses" WhatsApp discussion into a small French chat assistant, and compare two ways
of getting there on one laptop (Apple M5 Pro, 24 GB):

1. **Fine-tune Gemma 4 E2B** with LoRA (MLX).
2. **Train `galsenai`**, a compact nanochat-style decoder-only transformer (12M parameters) from scratch, in this repo.

Both are trained on the same 68 conversations and scored on the same 8 held-out ones.

## Pipeline

```
WhatsApp export ──► prepare_data_galesnai.py (system lines out, phone numbers → keyed hash)
                └─► anonymize_chat.py (random sender labels, names/links/numbers removed, dates → month)
conversations (data/private/supervised_fine_tuning_galsenai.json, 76 Q&A conversations)
   └─► prepare_sft_mlx.py ──► train.jsonl (68) / valid.jsonl (8)
         ├─► train_lora.sh        Gemma 4 E2B + LoRA            (config_gemma.yml)
         └─► train_tokenizer.py → galsenai.train base → sft     (config_nanochat.yml)
                                  └─► run_eval.sh ──► results/eval_compare.md
                                  └─► webapp/  chat UI on any OpenAI-compatible server (vLLM, mlx_lm.server)
```

Data format: a list of conversations, each a list of `{"role": "user" | "assistant", "content": ...}` turns.

## Install

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
pytest -q && ruff check . && mypy
```

Gemma fine-tuning and evaluation need [`mlx-lm`](https://github.com/ml-explore/mlx-lm), installed separately
(it is not a project dependency): `uv tool install mlx-lm`.

## Fine-tune Gemma 4 E2B (LoRA)

All hyper-parameters, including the number of **epochs**, are in `config_gemma.yml`; `train_lora.sh` turns `epochs`
into `--iters` (mlx-lm only knows iterations) and evaluates and saves once per epoch.

```bash
python3 scripts/prepare_sft_mlx.py      # fixed split: 68 train / 8 valid
./scripts/train_lora.sh                 # reads config_gemma.yml; extra flags override, e.g. --learning-rate 5e-5
```

Settings that fit 24 GB: LoRA rank 8 on the top 16 layers, batch 2, gradient checkpointing (needed: without it rank 64
with batch 8 runs out of Metal memory). Rank 32 on all layers at learning rate 1e-4 **diverged** (validation loss
3.7 → 6.6), so it is not the default. Validation loss is lowest after **epoch 1** and rises afterwards (overfitting on
68 conversations), so 1–2 epochs is the safer choice than the 5 in the file.

## The `galsenai` model (nanochat style)

A small decoder-only transformer: RMSNorm, RoPE, grouped-query attention, SwiGLU, KV cache, optional sliding-window
layers, LoRA. `ModelConfig.nanochat(vocab_size, ...)` switches on the parts of
[nanochat](https://github.com/karpathy/nanochat)'s GPT: relu² MLP, QK-norm, parameter-free RMSNorm, untied embeddings,
logit soft-cap 15, RoPE base 1e5, SSSL sliding windows, per-layer residual/x0 lambdas. Not ported: value embeddings,
smear gate, FP8, flash-attention, the Muon optimizer (AdamW is used). The chat format is nanochat's (`<|bos|>`,
`<|user_start|>`, `<|assistant_start|>`, ...; loss on assistant tokens only).

```bash
python3 scripts/anonymize_chat.py CHAT.txt data/private/chat_anonymized.txt   # corpus for tokenizer and base stage
python3 scripts/train_tokenizer.py                                            # BPE 2048 + 5 chat specials
python -m galsenai.train --stage base                                         # next-token on chat text
python -m galsenai.train --stage sft                                          # assistant tokens only
python -m galsenai.chat --checkpoint data/private/checkpoints/galsenai_sft.pt \
                        --tokenizer data/private/galsenai_tok/galsenai.model
```

Config and epochs: `config_nanochat.yml` (override with `--set sft.epochs=3`). Pre-training on the raw chat gives the
model its vocabulary but also lets it see the source of the validation answers (see the evaluation caveats).

```python
import torch
from galsenai import GPT, ModelConfig, generate

model = GPT(ModelConfig.nanochat(vocab_size=2053, n_layer=6, n_head=6, n_embd=384))
tokens = generate(model, torch.tensor([[1, 2, 3]]), max_new_tokens=20, temperature=0.8, top_k=50)
```

| Choice | Why |
|---|---|
| RoPE, no learned positions | Relative positions; no table tied to the training length |
| Grouped-query attention (`n_kv_head`) | KV cache shrinks by `n_head / n_kv_head` (the nanochat preset uses full MHA) |
| Sliding-window layers (`sliding_window`, `global_every`) | Bounded attention cost, interleaved with global layers as in Gemma 3 |
| No biases, KV-cache decoding | Fewer parameters; one new token costs O(1) projections |

The structure follows [nanoGPT](https://github.com/karpathy/nanoGPT) (deprecated upstream in favour of nanochat).

## Evaluation

`./scripts/run_eval.sh` scores every model on the same 10 held-out assistant turns (gold history, greedy decoding) and
writes `results/eval_compare.md`: bits per byte (tokenizer-independent), perplexity per token, ROUGE-L, token F1,
distinct-2 and length ratio.

| Model | Params | BPB ↓ | PPL/token ↓ | ROUGE-L ↑ | Token F1 ↑ | Distinct-2 ↑ | Len ratio | Empty |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 E2B, no fine-tuning | 4629M | 1.130 | 30.6 | 0.082 | 0.144 | 0.86 | 8.37 | 0% |
| Gemma 4 E2B + LoRA, 5 epochs | 4629M | 1.378 | 65.0 | 0.150 | 0.244 | 0.98 | 0.80 | 0% |
| Gemma 4 E2B + LoRA, epoch 1 | 4629M | **0.854** | **13.3** | **0.196** | **0.272** | 0.69 | 0.99 | 0% |
| galsenai 12M, base + SFT | 12M | 1.825 | 65.5 | 0.149 | 0.191 | 0.76 | 0.47 | 0% |
| galsenai 12M, SFT only | 12M | 2.279 | 185.1 | 0.124 | 0.152 | 0.82 | 0.38 | 0% |

How to read it:
- **BPB** (bits per byte of the reference answer) and **PPL/token** score the reference answer under the model. BPB is
  comparable across tokenizers, PPL/token is not.
- **ROUGE-L / Token F1** measure word overlap with the reference. Low absolute values are normal: the references are
  paraphrases of chat opinions.
- **Distinct-2** is the share of unique word bigrams (low means repetition loops). **Len ratio** is generated words
  over reference words: base Gemma is 8x too long, `galsenai` is about half as long as the reference.
- Gemma LoRA at 5 epochs is worse than the untuned model on BPB (1.38 vs 1.13): overfitting. Epoch 1 is the best row on
  every quality metric.

Caveats: 10 turns from 8 conversations, so only large gaps mean something. `galsenai` base + SFT saw the raw chat the
validation answers come from (optimistic); "SFT only" is the leak-free row. Gemma answers with its thinking mode on
(same template as training) and the thought block is stripped before scoring.

## Full fine-tuning and model merging (local)

`config_gemma_full.yml` trains the real weights of the top 16 blocks of the bf16 `google/gemma-4-E2B-it` (Adafactor, lr 2e-5);
LoRA instead trains small adapters on the 4-bit model. Everything is driven by `scripts/run_full.sh`:

```
./scripts/run_full.sh train                 # add flags to override, e.g. --learning-rate 1e-5 --num-layers -1
./scripts/run_full.sh fuse                  # -> data/private/models/gemma4-full-ft
./scripts/run_full.sh merge 20 40 60 80     # base + full weights: gemma4-full-merge-w<pct> (pct = % of the fine-tuned model)
./scripts/run_full.sh eval                  # validation metrics for every gemma4-full-* model + the bf16 base
```

Every `data/private/models/gemma4-full-*` directory (including the `merge-w<pct>` ones) shows up in the chat app's model picker.
Outputs (`results/*full*`, `results/*merge*`, `data/private/`) are git-ignored. Useful variants: `./scripts/run_full.sh ckpt STEP`
(checkpoint at iteration STEP), `FULL=ep1 ./scripts/run_full.sh merge 40 60` (merge a specific fused model).

Outcome (10 validation turns, table in `results/eval_compare.md`, per-model details in `results/gemma-full-*.json`): full
fine-tuning (BPB 1.083, val loss 3.42 vs 2.66 for LoRA) overfits after 1 epoch, and training all blocks diverges. Merging with
the base helps: `merge-w40` BPB 0.864, `merge-w60` ROUGE-L 0.163 / F1 0.235. None beats `gemma-lora-1ep` (BPB 0.854,
ROUGE-L 0.196, F1 0.272), so LoRA stays the served model. Knowledge retention is not measured by these metrics.

## RAG (retrieval over the anonymized chat)

`webapp/rag.py` indexes `data/private/chat_anonymized.txt` (windows of 4 messages) with BM25, `google/embeddinggemma-2`
(text only, `pip install -e .[rag]`) or both (reciprocal rank fusion). The chat app has a RAG selector; passages are injected
in the last user turn and shown under the answer. `scripts/rag_contexts.py` scores retrieval (hit@k, MRR, context recall, chunk
sizes), `scripts/run_eval_rag.sh` evaluates Gemma + RAG on the validation turns, `scripts/eval_rag.py` runs hand-written questions.
Results go to `results/local/` and `results/eval_compare.md`.

## Chat web app

`webapp/` is a minimal chat UI (GalsenAI logo and colours: green top bar and footer, white conversation area, left
panel listing the group's themes; a click sends a natural question to the model) with a model picker (`galsenai-gemma`: any OpenAI-compatible server; `galsenai-nanochat`: the 12M checkpoint, generated in-process) and ChatGPT-style rendered Markdown answers. vLLM needs a GPU machine; on the Mac
`./scripts/serve.sh` starts `mlx_lm.server` (with the LoRA adapter if present) and the web app together (Ctrl-C stops both):

```bash
./scripts/serve.sh                      # then open http://127.0.0.1:8080
# GPU machine instead
vllm serve google/gemma-4-E2B-it --port 8000
VLLM_BASE_URL=http://HOST:8000/v1 .venv/bin/python webapp/server.py
```

With `mlx_lm.server` the request model name must be `default_model` to use the preloaded model and its adapter (`serve.sh`
sets it and passes the real name to the UI via `ACTIVE_MODEL`); any other name loads that model without the adapter.

Optional environment variables: `MODEL`, `SYSTEM_PROMPT`, `API_KEY`. Standard library only.

## Report

The discussion statistics (themes, top voices, word cloud) are in [`results/STATISTICS.md`](results/STATISTICS.md).

`report/rapport_galsenai_underneath.pdf` is the only report file in the repository (3 pages, French: cover with a map of
Senegal and the GalsenAI logo, what was built, results, discussion statistics). Its LaTeX sources (`report/*.tex`, `assets/`) are **kept local and git-ignored**; `senegal.tex` is generated by
`scripts/make_senegal_map.py` and `stats_data.tex` by `scripts/chat_figures.py`.

Rebuild: `python3 scripts/make_senegal_map.py`, `uv run --no-project --with wordcloud python scripts/chat_figures.py`,
then `cd report && latexmk -xelatex main.tex`.

## Tests

`tests/` checks properties, not implementation: causality, KV-cache equals full forward (including the nanochat
variant), sliding-window receptive field, RoPE relative-position invariance, parameter count, loss masking,
memorization of a short sequence, sampling filters, checkpoint round trip, LoRA identity at initialization, the chat
format and assistant-only mask, and the anonymizer.

## Data preparation and privacy

- `data/prepare_data_galesnai.py` removes WhatsApp system lines and replaces every phone number with an HMAC-SHA256
  token (`TEL_ab12cd34`) under a secret local key. The key and the token-to-number table live in `data/private/` and
  never leave the machine. Display names are **not** anonymized by this step.
- `scripts/anonymize_chat.py` is the step before sharing text. Two layers of protection exist and they are not the
  same thing: pseudonymization (the `TEL_` tokens, reversible by whoever holds the key) and anonymization (this script,
  where nothing is kept that links back to a person).

  | In the text | Treatment |
  |---|---|
  | Sender (display name or `TEL_` token) | Random label `U01`, `U02`, ... drawn fresh each run; **the mapping is never written** |
  | The same names or `TEL_` tokens quoted inside messages | Same label as the sender, so threads stay readable |
  | Links, e-mails, `@mentions`, phone-like numbers | `[lien]`, `[email]`, `[mention]`, `[numero]` |
  | Date and time | Year-month only |
  | Other people's names in free text, extra identifying details | **Not detected**: pass them with `--redact-words` |

  Dropping the mapping table makes re-identification harder but does not make the data anonymous: in a community of
  about 100 members, an opinion, a job or a writing style can identify someone. Before sharing outside the group,
  proofread a sample, remove very personal messages, and get the administrators' consent. Under GDPR, data that is
  only pseudonymized is still personal data. This is not legal advice.

  ```bash
  python3 scripts/anonymize_chat.py data/whatsapp_questions_reponses/CHAT.txt data/private/chat_anonymized.txt
  python3 scripts/anonymize_chat.py CHAT.txt OUT.txt --redact-words names.txt   # one word or name per line
  ```

  The anonymized file is also the corpus the `galsenai` tokenizer and base stage train on.
- Raw exports, media, backups, trained adapters and checkpoints are git-ignored. Model outputs in `results/` quote the
  chat: do not publish them from a public repository without review.

```bash
python data/prepare_data_galesnai.py --dry-run   # count only
python data/prepare_data_galesnai.py             # backup, filter, pseudonymize
```

## Roadmap

1. Pick the Gemma epoch count on a larger validation set, and add more conversations (76 is small).
2. Export: weight quantization (int8/int4), then LiteRT or GGUF, with an accuracy check against the PyTorch model.
3. Measure on-device memory and tokens/second on a real phone.

## Attribution

`minbpe/` is Andrej Karpathy's tokenizer (MIT, see `minbpe/LICENSE`). The model structure follows nanoGPT and nanochat
(Karpathy, MIT). The course notebooks in `legacy/` come from Imad Saddik's "Train your language model" course; check
its license before redistributing any course material.
