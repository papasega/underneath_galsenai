"""Evaluate Gemma (optionally with the LoRA adapter) on the validation conversations.

Needs mlx-lm, so run it with the mlx-lm interpreter, not the project venv:
    "$HOME/Library/Application Support/pipx/venvs/mlx-lm/bin/python" scripts/eval_gemma.py --name gemma-lora --adapter data/private/adapters/gemma4_e2b_galsenai
"""

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from eval_metrics import assistant_turns, summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--model", default="mlx-community/gemma-4-E2B-it-qat-4bit")
ap.add_argument("--adapter", type=Path)
ap.add_argument("--data", type=Path, default=ROOT / "data/private/mlx_sft/valid.jsonl")
ap.add_argument("--max-new-tokens", type=int, default=1500, help="Gemma may think first, so it needs room")
ap.add_argument("--out", type=Path)
a = ap.parse_args()

model, tok = load(a.model, adapter_path=str(a.adapter) if a.adapter else None)
rows = []
for history, ref in assistant_turns(a.data):
    prompt = tok.apply_chat_template(history, add_generation_prompt=True, tokenize=True)
    full = tok.apply_chat_template([*history, ref], tokenize=True)
    if full[: len(prompt)] != prompt:
        raise SystemExit("chat template is not prefix-stable; cannot isolate the assistant turn")
    ids = mx.array([full])
    logits = model(ids[:, :-1])[0].astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = ids[0, 1:]
    tok_lp = mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0]
    sel = tok_lp[len(prompt) - 1 :]  # positions predicting the assistant tokens (incl. end-of-turn)
    raw = generate(model, tok, prompt=prompt, max_tokens=a.max_new_tokens)
    # Same chat template as training (thinking on). Drop the thought channel; an unfinished one leaves an empty answer.
    pred = re.sub(r"<\|channel>.*?(<channel\|>|$)", "", raw, flags=re.S).replace("<turn|>", "").strip()
    rows.append(
        {
            "nll": -float(sel.sum()),
            "n_tokens": int(sel.size),
            "ref_bytes": len(ref["content"].encode()),
            "pred": pred,
            "ref": ref["content"],
        }
    )

res = {"name": a.name, "params_m": 4628.6, "metrics": summarize(rows), "samples": rows}
out = a.out or ROOT / f"results/{a.name}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(a.name, json.dumps(res["metrics"], indent=1))
