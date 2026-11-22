#!/usr/bin/env bash
# LoRA SFT of Gemma 4 E2B (MLX). Hyper-parameters live in config_gemma.yml; extra flags override it.
# Run scripts/prepare_sft_mlx.py first.
set -euo pipefail
cd "$(dirname "$0")/.."
~/.local/bin/mlx_lm.lora --config "${CONFIG:-config_gemma.yml}" "$@"
