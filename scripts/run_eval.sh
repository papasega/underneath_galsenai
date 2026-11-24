#!/usr/bin/env bash
# Compare base, v2 LoRA and assistant v3 on the same assistant validation set.
set -euo pipefail
cd "$(dirname "$0")/.."
cat config/dataset_notice.txt >&2
MLXPY=${MLXPY:-"$HOME/Library/Application Support/pipx/venvs/mlx-lm/bin/python"}
DATA=data/training_data/pilot_v3_assistant/valid.jsonl
V2=data/private/adapters/pilot_v2/gemma4_e2b_lora_seed0
V3=data/private/adapters/pilot_v3_assistant/gemma4_e2b_lora_rank8_seed0
for f in "$DATA" config/assistant_system.txt "$V2/adapter_config.json" "$V2/adapters.safetensors" \
         "$V3/adapter_config.json" "$V3/adapters.safetensors"; do
  [ -f "$f" ] || { echo "Missing required file: $f" >&2; exit 1; }
done
mkdir -p results/local
if [ -n "${OUT_DIR:-}" ]; then mkdir "$OUT_DIR"; else OUT_DIR=$(mktemp -d results/local/assistant-eval-XXXXXX); fi
echo "Results: $OUT_DIR"
args=(--data "$DATA" --system-prompt-file config/assistant_system.txt --no-thinking
      --max-new-tokens 512 --repetition-penalty 1.1 --seed 0)
if [ -n "${LIMIT:-}" ]; then args+=(--limit "$LIMIT"); fi
"$MLXPY" scripts/eval_gemma.py "${args[@]}" --name gemma-base --out "$OUT_DIR/base.json"
"$MLXPY" scripts/eval_gemma.py "${args[@]}" --name gemma-lora-v2 --adapter "$V2" --out "$OUT_DIR/v2.json"
"$MLXPY" scripts/eval_gemma.py "${args[@]}" --name gemma-lora-v3 --adapter "$V3" --out "$OUT_DIR/v3.json"
.venv/bin/python scripts/eval_assistant_report.py "$OUT_DIR"
