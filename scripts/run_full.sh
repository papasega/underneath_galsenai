#!/usr/bin/env bash
# Full fine-tuning chain for Gemma 4 E2B. Everything stays local: data/private/ and results/*full*, results/*merge* are git-ignored.
#   ./scripts/run_full.sh train [mlx_lm.lora flags]   train (config/pilotv1_config/config_gemma_full.yml), e.g. --learning-rate 1e-5 --num-layers -1
#   ./scripts/run_full.sh ckpt STEP [ADAPTER_DIR]     checkpoint at iteration STEP -> adapter dir <ADAPTER_DIR>_step<STEP>
#   ./scripts/run_full.sh fuse [ADAPTER_DIR] [NAME]   trained weights -> data/private/models/gemma4-full-NAME (default ft)
#   FULL=ep1 ./scripts/run_full.sh merge [20 40 60 80]  base + gemma4-full-$FULL (default ft) -> gemma4-full-merge-w<pct>
#   ./scripts/run_full.sh eval                        validation metrics of every gemma4-full-* model (+ bf16 base)
#   RAG=hybrid ./scripts/run_full.sh eval             also with retrieved passages (needs scripts/rag_contexts.py run before)
set -euo pipefail
cd "$(dirname "$0")/.."
printf '\033[32m%s\033[0m\n' "$(cat config/dataset_notice.txt)" >&2
MLXPY=${MLXPY:-.venv/bin/python}
MODELS=data/private/models
BASE=google/gemma-4-E2B-it
cmd=${1:-}; shift || true
case "$cmd" in
  train) CONFIG=config/pilotv1_config/config_gemma_full.yml ./scripts/train_lora.sh "$@" ;;
  ckpt) # mlx-lm saves NNNNNNN_adapters.safetensors every epoch; fuse wants adapters.safetensors in a directory
    SRC=${2:-data/private/adapters/gemma4_e2b_full}; DST=${SRC}_step$1
    mkdir -p "$DST"; cp "$SRC/adapter_config.json" "$DST/"
    cp "$SRC/$(printf %07d "$1")_adapters.safetensors" "$DST/adapters.safetensors"; echo "$DST" ;;
  fuse)
    ADAPTER=${1:-data/private/adapters/gemma4_e2b_full}
    "$MLXPY" scripts/mlx_entry.py fuse --model "$BASE" --adapter-path "$ADAPTER" --save-path "$MODELS/gemma4-full-${2:-ft}" ;;
  merge) "$MLXPY" -W ignore scripts/merge_models.py --full "$MODELS/gemma4-full-${FULL:-ft}" ${*:+--weights "$@"} ;;
  eval)
    # bf16 base: the fair reference, since the full models are bf16 and the other base row is 4-bit QAT
    "$MLXPY" -W ignore scripts/eval_gemma.py --name gemma-base-bf16 --model "$BASE"
    for d in $MODELS/gemma4-full-*/; do
      n=$(basename "$d"); n=${n#gemma4-}
      "$MLXPY" -W ignore scripts/eval_gemma.py --name "gemma-$n" --model "$d"
      [ -n "${RAG:-}" ] && "$MLXPY" -W ignore scripts/eval_gemma.py --name "gemma-$n-rag-$RAG" --model "$d" --rag-method "$RAG"
    done
    .venv/bin/python scripts/eval_report.py ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
