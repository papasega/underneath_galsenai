"""Evaluate a galsenai checkpoint on the validation conversations (run with the project venv).

python scripts/eval_galsenai.py --name galsenai-sft --checkpoint data/private/checkpoints/galsenai_sft.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).parent)]
from eval_metrics import assistant_turns, summarize  # noqa: E402

from galsenai.chat import ASSISTANT_END, ASSISTANT_START, encode_prompt, load_tokenizer, render_conversation  # noqa: E402
from galsenai.checkpoint import load_checkpoint  # noqa: E402
from galsenai.generate import generate  # noqa: E402
from galsenai.train import pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--checkpoint", type=Path, required=True)
ap.add_argument("--tokenizer", type=Path, default=ROOT / "data/private/galsenai_tok/galsenai.model")
ap.add_argument("--data", type=Path, default=ROOT / "data/private/mlx_sft/valid.jsonl")
ap.add_argument("--max-new-tokens", type=int, default=200)
ap.add_argument("--out", type=Path)
a = ap.parse_args()

device = pick_device()
model, tok = load_checkpoint(a.checkpoint, device).eval(), load_tokenizer(a.tokenizer)
start, end = tok.special_tokens[ASSISTANT_START], tok.special_tokens[ASSISTANT_END]
rows = []
for history, ref in assistant_turns(a.data):
    ids, mask = render_conversation(tok, [*history, ref])
    last = max(i for i, t in enumerate(ids) if t == start)
    mask = [0] * (last + 1) + mask[last + 1 :]  # score only the final assistant turn
    ids, mask = ids[: model.cfg.block_size], mask[: model.cfg.block_size]
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    with torch.no_grad():
        logits = model(x).logits
    m = torch.tensor(mask[1:], device=device).bool()
    nll = F.cross_entropy(logits[0][m], y[0][m], reduction="sum").item()
    prompt = torch.tensor([encode_prompt(tok, history)], device=device)
    out = generate(model, prompt, a.max_new_tokens, temperature=0, eos_ids={end})
    pred = tok.decode([t for t in out[0, prompt.size(1) :].tolist() if t != end]).strip()
    rows.append(
        {
            "nll": nll,
            "n_tokens": int(m.sum()),
            "ref_bytes": len(ref["content"].encode()),
            "pred": pred,
            "ref": ref["content"],
        }
    )

res = {
    "name": a.name,
    "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    "metrics": summarize(rows),
    "samples": rows,
}
out = a.out or ROOT / f"results/{a.name}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(a.name, json.dumps(res["metrics"], indent=1))
