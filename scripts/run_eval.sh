#!/usr/bin/env bash
# Compare Gemma 4 E2B (base / LoRA) with the galsenai nanochat-style model on the same validation conversations.
# Prerequisites: scripts/prepare_sft_mlx.py, train_lora.sh, train_tokenizer.py, galsenai.train (base, sft).
set -euo pipefail
cd "$(dirname "$0")/.."
MLXPY="$HOME/Library/Application Support/pipx/venvs/mlx-lm/bin/python"
A=data/private/adapters; C=data/private/checkpoints
"$MLXPY" -W ignore scripts/eval_gemma.py --name gemma-base
"$MLXPY" -W ignore scripts/eval_gemma.py --name gemma-lora --adapter $A/gemma4_e2b_galsenai
[ -d $A/ep1 ] && "$MLXPY" -W ignore scripts/eval_gemma.py --name gemma-lora-1ep --adapter $A/ep1
for n in sft-only sft; do
  [ -f $C/galsenai_$n.pt ] && .venv/bin/python -W ignore scripts/eval_galsenai.py --name galsenai-$n --checkpoint $C/galsenai_$n.pt
done
.venv/bin/python scripts/eval_report.py
