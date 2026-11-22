"""Merge results/*.json into results/eval_compare.md (run after the eval_* scripts)."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAG_METHODS = ["bm25", "dense", "hybrid", "oracle"]
BASES = ["gemma-base", "gemma-lora", "gemma-lora-1ep"]
ORDER = [*BASES, "galsenai-sft", "galsenai-sft-only", *(f"{b}-rag-{m}" for b in BASES for m in RAG_METHODS)]
RAG_NOTE = {
    "bm25": "BM25 top-4 passages", "dense": "embeddinggemma-2 top-4 passages",
    "hybrid": "BM25 + embeddinggemma-2 (RRF) top-4", "oracle": "the single best passage (upper bound, uses the reference)",
}
NOTES = {
    "gemma-base": "Gemma 4 E2B (4-bit QAT), no fine-tuning",
    "gemma-lora": "+ LoRA r8, 16 layers, 5 epochs (config_gemma.yml)",
    "gemma-lora-1ep": "+ LoRA, checkpoint after epoch 1 (lowest val loss)",
    "galsenai-sft": "nanochat-style 12M model, base on chat text + SFT",
    "galsenai-sft-only": "same model, SFT only (no raw-chat pre-training)",
}
NOTES.update({f"{b}-rag-{m}": f"{NOTES[b].split(',')[0].split(' +')[0]} + RAG: {RAG_NOTE[m]}" for b in BASES for m in RAG_METHODS})
# full fine-tuning / merge rows are discovered from the files (names: gemma-base-bf16, gemma-full-ft, gemma-full-merge-w40, ...)
for f in sorted(ROOT.glob("results/gemma-full-*.json")) + [ROOT / "results/gemma-base-bf16.json"]:
    n = f.stem
    if f.exists() and n not in ORDER:
        ORDER.append(n)
        pct = re.search(r"merge-w(\d+)", n)
        rag = re.search(r"-rag-(\w+)", n)
        NOTES[n] = (
            "Gemma 4 E2B bf16, no fine-tuning (fair base for the full models)" if n == "gemma-base-bf16"
            else (f"bf16 base + {pct[1]} % full fine-tuned weights, {100 - int(pct[1])} % base" if pct
                  else "full fine-tuning (top blocks, config_gemma_full.yml)")
            + (f" + RAG: {RAG_NOTE[rag[1]]}" if rag else "")
        )
res = {
    n: json.loads((ROOT / f"results/{n}.json").read_text(encoding="utf-8"))
    for n in ORDER
    if (ROOT / f"results/{n}.json").exists()
}
cols = [
    ("bits_per_byte", "BPB ↓", "{:.3f}"),
    ("perplexity", "PPL/token ↓", "{:.1f}"),
    ("rouge_l", "ROUGE-L ↑", "{:.3f}"),
    ("token_f1", "Token F1 ↑", "{:.3f}"),
    ("distinct_2", "Distinct-2 ↑", "{:.2f}"),
    ("length_ratio", "Len ratio", "{:.2f}"),
    ("empty_rate", "Empty", "{:.0%}"),
    ("groundedness", "Grounded ↑", "{:.2f}"),
    ("cited", "Cites [n] ↑", "{:.0%}"),
]
n = next(iter(res.values()))["metrics"]["n_turns"]
out = [
    "# Evaluation: Gemma 4 E2B LoRA vs galsenai (nanochat-style)\n",
    f"Same {n} assistant turns of the held-out validation conversations (`data/private/mlx_sft/valid.jsonl`, never trained on), "
    "gold history as context, greedy decoding. Regenerate with `scripts/run_eval.sh`.\n",
    "| Model | Params | " + " | ".join(c[1] for c in cols) + " | What it is |",
    "|---|---|" + "---|" * len(cols) + "---|",
]
for name, r in res.items():
    out.append(
        f"| {name} | {r['params_m']:.0f}M | "
        + " | ".join(f.format(r["metrics"][k]) if k in r["metrics"] else "–" for k, _, f in cols)
        + f" | {NOTES[name]} |"
    )
ctx_file = ROOT / "results/local/rag_ctx.json"
if ctx_file.exists():
    ret = json.loads(ctx_file.read_text(encoding="utf-8"))["retrieval"]
    out += [
        "\n## RAG: retrieval quality (same turns)\n",
        "Query = last user message. The *oracle* passage is the one whose content words best cover the reference answer; "
        "hit@k = that passage is in the top k; context recall = share of the reference's content words found in the 4 passages given to the model.\n",
        "| Retriever | hit@1 | hit@3 | hit@5 | MRR | Context recall |", "|---|---|---|---|---|---|",
    ]
    out += [f"| {m} | {v['hit@1']:.0%} | {v['hit@3']:.0%} | {v['hit@5']:.0%} | {v['mrr']:.2f} | {v['context_recall']:.2f} |"
            for m, v in ret.items() if "mrr" in v]
    out.append(f"| oracle (1 passage) | – | – | – | – | {ret['oracle']['context_recall']:.2f} |")
sweep_file = ROOT / "results/local/rag_chunk_sweep.json"
if sweep_file.exists():
    out += ["\n## RAG: chunk size (messages per passage, stride)\n", "| Chunking | Passages | Retriever | hit@5 | MRR | Context recall |", "|---|---|---|---|---|---|"]
    for cfg, v in json.loads(sweep_file.read_text(encoding="utf-8")).items():
        out += [f"| {cfg} | {v['passages']} | {m} | {v[m]['hit@5']:.0%} | {v[m]['mrr']:.2f} | {v[m]['context_recall']:.2f} |" for m in ("bm25", "dense", "hybrid")]
out += [
    "\n## How to read it\n",
    "- **RAG rows** (`*-rag-*`): same turns, but the last user message is preceded by retrieved chat passages and an instruction to answer from them. BPB / PPL are then conditioned on the passages. **Grounded**: share of the answer's content words (4+ letters) present in the passages. **Cites [n]**: answers containing a citation.",
    "- **RAG leakage caveat**: the reference answers were written from the same chat that the retriever indexes, so RAG is tested on questions whose source is in the corpus (the intended use), not on unseen knowledge. `oracle` shows the ceiling when the right chunk is given.",
    "- **BPB** (bits per byte of the reference answer) and **PPL/token** score the reference answer under the model; BPB is comparable across tokenizers, PPL/token is not.",
    "- **ROUGE-L / Token F1**: word overlap between the generated and the reference answer. Low absolute values are normal: the references are paraphrases of chat opinions.",
    "- **Distinct-2**: share of unique word bigrams (low = repetition loops). **Len ratio**: generated words / reference words.",
    "\n## Caveats\n",
    f"- **Tiny sample**: {n} turns from 8 conversations. Differences of a few hundredths are noise; only large gaps mean something.",
    "- **Leakage**: `galsenai-sft` was pre-trained on the anonymized chat from which the validation Q&A were extracted, so its numbers are optimistic; `galsenai-sft-only` is the leak-free galsenai row. Gemma never saw the chat.",
    "- Gemma answers with its thinking mode on (same chat template as training); the thought block is removed before scoring. An unfinished thought counts as an empty answer.",
    "- Gemma is 4.6B parameters pre-trained on trillions of tokens; the galsenai model is 12M parameters trained on ~130k tokens. The comparison shows what each recipe reaches with this data, not that the architectures are equivalent.",
    "- Only the validation loss was used to pick checkpoints (best epoch for galsenai, epoch 1 vs 5 for Gemma); with 8 conversations that choice is noisy.",
    "\n## Sample outputs (first 3 turns)\n",
]
first = next(iter(res.values()))["samples"]
for i in range(min(3, len(first))):
    out.append(f"**Reference {i + 1}:** {first[i]['ref'][:300]}\n")
    for name, r in res.items():
        out.append(f"- *{name}*: {r['samples'][i]['pred'][:300] or '(empty)'}")
    out.append("")
(ROOT / "results/eval_compare.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote results/eval_compare.md")
