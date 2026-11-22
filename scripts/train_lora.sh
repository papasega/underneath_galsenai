#!/usr/bin/env bash
# LoRA SFT of Gemma 4 E2B (MLX). Hyper-parameters live in config_gemma.yml (incl. `epochs`); extra flags override it.
# Run scripts/prepare_sft_mlx.py first.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-config_gemma.yml}
# Without --iters, derive iters and per-epoch eval/save steps from `epochs`.
if [[ " $* " != *" --iters"* ]]; then
  read -r ITERS PER_EPOCH < <(python3 -I - "$CONFIG" <<'PY'
import math, re, sys
cfg = open(sys.argv[1]).read()
get = lambda k: float(re.search(rf"^{k}:\s*([0-9.eE+-]+)", cfg, re.M).group(1))
data = re.search(r"^data:\s*(\S+)", cfg, re.M).group(1)
n = sum(1 for _ in open(f"{data}/train.jsonl"))
per_epoch = math.ceil(n / get("batch_size"))
print(math.ceil(get("epochs") * per_epoch), per_epoch)
PY
  )
  set -- --iters "$ITERS" --steps-per-eval "$PER_EPOCH" --save-every "$PER_EPOCH" "$@"
  echo "epochs from config -> iters=$ITERS ($PER_EPOCH per epoch)"
fi
~/.local/bin/mlx_lm.lora --config "$CONFIG" "$@"
