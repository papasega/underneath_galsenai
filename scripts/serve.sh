#!/usr/bin/env bash
# Start the Gemma inference server (mlx_lm.server + LoRA adapter) and the chat web app, which also offers the
# nanochat-style model. Ctrl-C stops both. vLLM does not run on Apple Silicon: on a GPU machine start
# `vllm serve MODEL --port 8000` and run `VLLM_BASE_URL=http://HOST:8000/v1 .venv/bin/python webapp/server.py`.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-mlx-community/gemma-4-E2B-it-qat-4bit}
MLXPY=${MLXPY:-.venv/bin/python}
ADAPTER=${ADAPTER:-data/private/adapters/pilot_v4_assistant/gemma4_e2b_lora_rank8_seed0}
IPORT=${IPORT:-8001}; WPORT=${WPORT:-8080}
args=(--model "$MODEL" --port "$IPORT")
if [ -n "${MODEL_REVISION:-}" ]; then args+=(--revision "$MODEL_REVISION"); fi
label="$MODEL"
if [ "$ADAPTER" != "none" ]; then
  for f in adapter_config.json adapters.safetensors; do
    [ -f "$ADAPTER/$f" ] || { echo "Missing adapter file: $ADAPTER/$f" >&2; exit 1; }
  done
  args+=(--adapter-path "$ADAPTER")
  label="$MODEL + LoRA ($(basename "$ADAPTER"))"
fi
"$MLXPY" scripts/mlx_entry.py server "${args[@]}" >/tmp/galsenai_mlx_server.log 2>&1 &
INFER=$!
GPORT=${GPORT:-8002}
GOOGLE_MODEL=google/gemma-4-E2B-it
GOOGLE_ADAPTER=data/private/adapters/pilot_v4_assistant/google_gemma4_rank8_run1
"$MLXPY" scripts/mlx_entry.py server --model "$GOOGLE_MODEL" --adapter-path "$GOOGLE_ADAPTER" --port "$GPORT" >/tmp/galsenai_google_server.log 2>&1 &
GOOGLE_INFER=$!
trap 'kill $INFER $GOOGLE_INFER 2>/dev/null; wait $GOOGLE_INFER 2>/dev/null || true; wait $INFER 2>/dev/null || true; echo "Serveurs arrêtés."' EXIT
trap 'exit 0' INT TERM
for _ in $(seq 60); do curl -sf "localhost:$IPORT/v1/models" >/dev/null && break; sleep 2; done
curl -sf "localhost:$IPORT/v1/models" >/dev/null || { echo "inference server failed, see /tmp/galsenai_mlx_server.log"; exit 1; }
for _ in $(seq 60); do curl -sf "localhost:$GPORT/v1/models" >/dev/null && break; sleep 2; done
curl -sf "localhost:$GPORT/v1/models" >/dev/null || { echo "Google inference failed, see /tmp/galsenai_google_server.log"; exit 1; }
echo "Gemma sur :$IPORT ($label). Chat : http://127.0.0.1:$WPORT"
# mlx_lm.server serves the preloaded model (with its adapter) only under the name "default_model".
GOOGLE_BASE_URL="http://localhost:$GPORT/v1" GOOGLE_ACTIVE_MODEL="$GOOGLE_MODEL + LoRA ($(basename "$GOOGLE_ADAPTER"))" MODEL=default_model ACTIVE_MODEL="$label" VLLM_BASE_URL="http://localhost:$IPORT/v1" .venv/bin/python webapp/server.py --port "$WPORT"
