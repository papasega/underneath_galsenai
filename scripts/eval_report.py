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
    ("rouge_l", "ROUGE-L ↑", "{:.3f}"),
    ("token_f1", "Token F1 ↑", "{:.3f}"),
    ("distinct_2", "Distinct-2 ↑", "{:.2f}"),
    ("groundedness", "Grounded ↑", "{:.2f}"),
]
n = next(iter(res.values()))["metrics"]["n_turns"]
out = [
    "# Evaluation: Gemma 4 E2B vs galsenai\n",
    f"{n} held-out validation turns, greedy decoding. Regenerate with `scripts/run_eval.sh`.\n",
    "| Model | " + " | ".join(c[1] for c in cols) + " | What it is |",
    "|---|" + "---|" * len(cols) + "---|",
]
for name, r in res.items():
    out.append(
        f"| {name} | "
        + " | ".join(f.format(r["metrics"][k]) if k in r["metrics"] else "–" for k, _, f in cols)
        + f" | {NOTES[name]} |"
    )
ctx_file = ROOT / "results/local/rag_ctx.json"
if ctx_file.exists():
    ret = json.loads(ctx_file.read_text(encoding="utf-8"))["retrieval"]
    out += ["\n## RAG retrieval\n", "| Retriever | hit@5 | MRR | Context recall |", "|---|---|---|---|"]
    out += [f"| {m} | {v['hit@5']:.0%} | {v['mrr']:.2f} | {v['context_recall']:.2f} |" for m, v in ret.items() if "mrr" in v]
sweep_file = ROOT / "results/local/rag_chunk_sweep.json"
if sweep_file.exists():
    out += ["\n## RAG chunk size (hybrid)\n", "| Chunking | Passages | hit@5 | MRR | Context recall |", "|---|---|---|---|---|"]
    for cfg, v in json.loads(sweep_file.read_text(encoding="utf-8")).items():
        out.append(f"| {cfg} | {v['passages']} | {v['hybrid']['hit@5']:.0%} | {v['hybrid']['mrr']:.2f} | {v['hybrid']['context_recall']:.2f} |")
out += [
    "\n## Notes\n",
    "- **BPB**: bits per byte of the reference answer, comparable across tokenizers. **ROUGE-L / Token F1**: overlap with the reference. "
    "**Distinct-2**: low = repetition loops. **Grounded**: share of answer words found in the retrieved passages.",
    f"- **Tiny sample** ({n} turns): only large gaps mean something.",
    "- **Leakage**: `galsenai-sft` was pre-trained on the source chat (optimistic); `galsenai-sft-only` is leak-free. "
    "RAG indexes the same chat, so `oracle` is an upper bound.",
]
(ROOT / "results/eval_compare.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote results/eval_compare.md")
