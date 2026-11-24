"""Linear weight merge of the base model and the full fine-tuned one: W = (1 - w) * base + w * full.

Why: full fine-tuning on ~3k messages moves the weights toward the group's style but also forgets; interpolating back
toward the base (model soups / WiSE-FT) usually keeps most of the style with less forgetting. w is the share of the
fine-tuned model: 40 -> 40 % full + 60 % base. Only tensors that differ are interpolated (top layers when num_layers < all).
Run with the mlx-lm interpreter:
    .venv/bin/python scripts/merge_models.py --full data/private/models/gemma4-full-ft --weights 20 40 60 80
Output: data/private/models/gemma4-full-merge-w<w>/ (the webapp lists every gemma4-full-* directory).
"""

import argparse
import glob
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.utils import hf_repo_to_path, save

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="google/gemma-4-E2B-it", help="same bf16 checkpoint the full fine-tuning started from")
ap.add_argument("--full", type=Path, required=True, help="fused full fine-tuned model directory")
ap.add_argument("--weights", type=int, nargs="+", default=[20, 40, 60, 80], help="percent of the fine-tuned model")
ap.add_argument("--out-dir", type=Path, default=Path("data/private/models"))
a = ap.parse_args()

model, tok, config = load(str(a.full), return_config=True)
full = dict(tree_flatten(model.parameters()))
src = Path(a.base) if Path(a.base).exists() else hf_repo_to_path(a.base)
base = {}
for f in sorted(glob.glob(str(src / "*.safetensors"))):
    base.update(mx.load(f))  # lazy: nothing is read until used
base = model.sanitize(base) if hasattr(model, "sanitize") else base

changed = []
for k, v in full.items():
    b = base.get(k)
    if b is None or b.shape != v.shape or not mx.issubdtype(v.dtype, mx.floating):
        continue
    if not bool(mx.array_equal(b.astype(v.dtype), v)):
        changed.append(k)
if not changed:
    raise SystemExit("no tensor differs between base and full model: wrong --full or the fine-tuning changed nothing")
print(f"{len(changed)} of {len(full)} tensors differ from the base; merging those")

for pct in a.weights:
    w = pct / 100
    merged = []
    for k in changed:
        m = (1 - w) * base[k].astype(mx.float32) + w * full[k].astype(mx.float32)
        merged.append((k, m.astype(full[k].dtype)))
    mx.eval([m for _, m in merged])
    model.update(tree_unflatten(merged))  # untouched tensors keep their (identical) values
    dst = a.out_dir / f"gemma4-full-merge-w{pct}"
    save(dst, str(a.full), model, tok, config, donate_model=False)
    print("saved", dst)
    model.update(tree_unflatten(list(full.items())))  # restore the fine-tuned weights for the next w
