#!/usr/bin/env bash
# Gemma + RAG on the validation turns, every retriever x every adapter. Results: results/*-rag-*.json (git-ignored).
# Needs scripts/rag_contexts.py first (project venv), then the mlx-lm interpreter for generation.
set -euo pipefail
cd "$(dirname "$0")/.."
MLXPY="$HOME/Library/Application Support/pipx/venvs/mlx-lm/bin/python"
A=data/private/adapters
.venv/bin/python -W ignore scripts/rag_contexts.py --sweep
for m in bm25 dense hybrid oracle; do
  "$MLXPY" -W ignore scripts/eval_gemma.py --name gemma-base-rag-$m --rag-method $m
  "$MLXPY" -W ignore scripts/eval_gemma.py --name gemma-lora-rag-$m --rag-method $m --adapter $A/gemma4_e2b_galsenai
  [ -d $A/ep1 ] && "$MLXPY" -W ignore scripts/eval_gemma.py --name gemma-lora-1ep-rag-$m --rag-method $m --adapter $A/ep1
done
.venv/bin/python scripts/eval_report.py
