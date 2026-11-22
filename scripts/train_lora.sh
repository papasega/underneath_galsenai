#!/usr/bin/env bash
# LoRA SFT of Gemma 4 E2B (4-bit QAT, MLX) on the GalsenAI chat data. Run scripts/prepare_sft_mlx.py first.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-mlx-community/gemma-4-E2B-it-qat-4bit}
# 68 train convs / batch 2 = 34 it per epoch; 170 iters is about 5 epochs. Watch val loss, lower ITERS if it rises.
~/.local/bin/mlx_lm.lora --model "$MODEL" --train --data data/private/mlx_sft \
  --mask-prompt --num-layers 16 --batch-size 2 --iters "${ITERS:-170}" --learning-rate 1e-4 \
  --steps-per-eval 34 --save-every 34 --max-seq-length 1024 --grad-checkpoint \
  --adapter-path data/private/adapters/gemma4_e2b_galsenai
