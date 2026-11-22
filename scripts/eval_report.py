"""Merge results/*.json into results/eval_compare.md (run after the eval_* scripts)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["gemma-base", "gemma-lora", "gemma-lora-1ep", "galsenai-sft", "galsenai-sft-only"]
NOTES = {
    "gemma-base": "Gemma 4 E2B (4-bit QAT), no fine-tuning",
    "gemma-lora": "+ LoRA r8, 16 layers, 5 epochs (config_gemma.yml)",
    "gemma-lora-1ep": "+ LoRA, checkpoint after epoch 1 (lowest val loss)",
    "galsenai-sft": "nanochat-style 12M model, base on chat text + SFT",
    "galsenai-sft-only": "same model, SFT only (no raw-chat pre-training)",
}
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
        + " | ".join(f.format(r["metrics"][k]) for k, _, f in cols)
        + f" | {NOTES[name]} |"
    )
out += [
    "\n## How to read it\n",
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
