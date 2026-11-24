> Historical v1 results. Current assistant experiments: [v4 comparison](assistant_v4_compare.md).

# Evaluation: Gemma 4 E2B vs galsenai

10 held-out validation turns, greedy decoding. Regenerate with `scripts/run_eval.sh`.

| Model | BPB ↓ | ROUGE-L ↑ | Token F1 ↑ | Distinct-2 ↑ | Grounded ↑ | What it is |
|---|---|---|---|---|---|---|
| gemma-base | 1.130 | 0.082 | 0.144 | 0.86 | – | Gemma 4 E2B (4-bit QAT), no fine-tuning |
| gemma-lora | 1.378 | 0.150 | 0.244 | 0.98 | – | + LoRA r8, 16 layers, 5 epochs (config_gemma.yml) |
| gemma-lora-1ep | 0.854 | 0.196 | 0.272 | 0.69 | – | + LoRA, checkpoint after epoch 1 (lowest val loss) |
| galsenai-sft | 1.825 | 0.149 | 0.191 | 0.76 | – | nanochat-style 12M model, base on chat text + SFT |
| galsenai-sft-only | 2.279 | 0.124 | 0.152 | 0.82 | – | same model, SFT only (no raw-chat pre-training) |
| gemma-base-rag-bm25 | 0.983 | 0.179 | 0.328 | 0.96 | 0.47 | Gemma 4 E2B (4-bit QAT) + RAG: BM25 top-4 passages |
| gemma-base-rag-dense | 0.938 | 0.181 | 0.324 | 0.96 | 0.44 | Gemma 4 E2B (4-bit QAT) + RAG: embeddinggemma-2 top-4 passages |
| gemma-base-rag-hybrid | 0.964 | 0.184 | 0.333 | 0.96 | 0.44 | Gemma 4 E2B (4-bit QAT) + RAG: BM25 + embeddinggemma-2 (RRF) top-4 |
| gemma-base-rag-oracle | 0.946 | 0.209 | 0.336 | 0.98 | 0.43 | Gemma 4 E2B (4-bit QAT) + RAG: the single best passage (upper bound, uses the reference) |
| gemma-lora-rag-bm25 | 1.184 | 0.177 | 0.287 | 0.98 | 0.50 | + LoRA r8 + RAG: BM25 top-4 passages |
| gemma-lora-rag-dense | 1.123 | 0.195 | 0.322 | 0.98 | 0.51 | + LoRA r8 + RAG: embeddinggemma-2 top-4 passages |
| gemma-lora-rag-hybrid | 1.142 | 0.170 | 0.302 | 0.98 | 0.46 | + LoRA r8 + RAG: BM25 + embeddinggemma-2 (RRF) top-4 |
| gemma-lora-rag-oracle | 1.157 | 0.173 | 0.277 | 0.99 | 0.45 | + LoRA r8 + RAG: the single best passage (upper bound, uses the reference) |
| gemma-lora-1ep-rag-bm25 | 0.734 | 0.171 | 0.290 | 0.84 | 0.59 | + LoRA + RAG: BM25 top-4 passages |
| gemma-lora-1ep-rag-dense | 0.705 | 0.201 | 0.319 | 0.87 | 0.58 | + LoRA + RAG: embeddinggemma-2 top-4 passages |
| gemma-lora-1ep-rag-hybrid | 0.722 | 0.180 | 0.296 | 0.82 | 0.57 | + LoRA + RAG: BM25 + embeddinggemma-2 (RRF) top-4 |
| gemma-lora-1ep-rag-oracle | 0.701 | 0.254 | 0.375 | 0.91 | 0.53 | + LoRA + RAG: the single best passage (upper bound, uses the reference) |
| gemma-full-ep1 | 1.083 | 0.125 | 0.163 | 0.76 | – | full fine-tuning (top blocks, config_gemma_full.yml) |
| gemma-full-merge-w40 | 0.864 | 0.118 | 0.182 | 0.66 | – | bf16 base + 40 % full fine-tuned weights, 60 % base |
| gemma-full-merge-w60 | 0.905 | 0.163 | 0.235 | 0.88 | – | bf16 base + 60 % full fine-tuned weights, 40 % base |
| gemma-base-bf16 | 1.557 | 0.078 | 0.137 | 0.86 | – | Gemma 4 E2B bf16, no fine-tuning (fair base for the full models) |

## RAG retrieval

| Retriever | hit@5 | MRR | Context recall |
|---|---|---|---|
| bm25 | 90% | 0.38 | 0.37 |
| dense | 70% | 0.44 | 0.45 |
| hybrid | 90% | 0.45 | 0.39 |

## RAG chunk size (hybrid)

| Chunking | Passages | hit@5 | MRR | Context recall |
|---|---|---|---|---|
| window=2,stride=1 | 3134 | 70% | 0.37 | 0.29 |
| window=4,stride=2 | 1566 | 90% | 0.45 | 0.39 |
| window=8,stride=4 | 782 | 90% | 0.72 | 0.49 |

## Notes

- **BPB**: bits per byte of the reference answer, comparable across tokenizers. **ROUGE-L / Token F1**: overlap with the reference. **Distinct-2**: low = repetition loops. **Grounded**: share of answer words found in the retrieved passages.
- **Tiny sample** (10 turns): only large gaps mean something.
- **Leakage**: `galsenai-sft` was pre-trained on the source chat (optimistic); `galsenai-sft-only` is leak-free. RAG indexes the same chat, so `oracle` is an upper bound.
