#!/usr/bin/env bash
# LoRA SFT of Gemma 4 E2B (MLX). Hyper-parameters live in config/config_gemma.yml (incl. `epochs`); extra flags override it.
# Requires the frozen dataset selected by the configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
cat config/dataset_notice.txt >&2
CONFIG=${CONFIG:-config/config_gemma.yml}
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
  # Preserve evaluation/save cadence from config, falling back to once per epoch.
  if ! rg -q '^steps_per_eval:' "$CONFIG"; then
    set -- --steps-per-eval "$PER_EPOCH" "$@"
  fi
  if ! rg -q '^save_every:' "$CONFIG"; then
    set -- --save-every "$PER_EPOCH" "$@"
  fi
  set -- --iters "$ITERS" "$@"
  echo "epochs from config -> iters=$ITERS ($PER_EPOCH per epoch)"
fi
~/.local/bin/mlx_lm.lora --config "$CONFIG" "$@"
