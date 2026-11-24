#!/usr/bin/env bash
# LoRA SFT of Gemma 4 E2B (MLX). Hyper-parameters live in config/config_gemma.yml (incl. `epochs`); extra flags override it.
# Requires the frozen dataset selected by the configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
printf '\033[32m%s\033[0m\n' "$(cat config/dataset_notice.txt)" >&2
CONFIG=${CONFIG:-config/config_gemma.yml}
MLXPY=${MLXPY:-.venv/bin/python}
# Without --iters, derive iters and per-epoch eval/save steps from `epochs`.
HAS_ITERS=false
for arg in "$@"; do
  case "$arg" in --iters|--iters=*) HAS_ITERS=true ;; esac
done
if [ "$HAS_ITERS" = false ]; then
  SCHEDULE=$(python3 -I - "$CONFIG" "$@" <<'PY'
import math, re, sys
cfg = open(sys.argv[1]).read()
get = lambda k: float(re.search(rf"^{k}:\s*([0-9.eE+-]+)", cfg, re.M).group(1))
data = re.search(r"^data:\s*(\S+)", cfg, re.M).group(1)
args = sys.argv[2:]
def override(name, default):
    value = default
    for i, arg in enumerate(args):
        if arg == name:
            value = args[i + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value
data = override("--data", data)
batch = float(override("--batch-size", get("batch_size")))
if batch < 1:
    raise ValueError("batch size must be positive")
n = sum(1 for line in open(f"{data}/train.jsonl") if line.strip())
if not n:
    raise ValueError("Training dataset is empty")
per_epoch = math.ceil(n / batch)
print(math.ceil(get("epochs") * per_epoch), per_epoch)
PY
  )
  read -r ITERS PER_EPOCH <<< "$SCHEDULE"
  # Preserve evaluation/save cadence from config, falling back to once per epoch.
  if ! grep -q '^steps_per_eval:' "$CONFIG"; then
    set -- --steps-per-eval "$PER_EPOCH" "$@"
  fi
  if ! grep -q '^save_every:' "$CONFIG"; then
    set -- --save-every "$PER_EPOCH" "$@"
  fi
  set -- --iters "$ITERS" "$@"
  echo "epochs from config -> iters=$ITERS ($PER_EPOCH per epoch)"
fi
"$MLXPY" scripts/mlx_entry.py lora --config "$CONFIG" "$@"
