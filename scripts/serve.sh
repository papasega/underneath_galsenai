#!/usr/bin/env bash
# Start the Gemma inference server (mlx_lm.server + LoRA adapter) and the chat web app, which also offers the
# nanochat-style model. Ctrl-C stops both. vLLM does not run on Apple Silicon: on a GPU machine start
# `vllm serve MODEL --port 8000` and run `VLLM_BASE_URL=http://HOST:8000/v1 .venv/bin/python webapp/server.py`.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-google/gemma-4-E2B-it}
ADAPTER=${ADAPTER:-data/private/adapters/pilot_v4_assistant/google_gemma4_rank8_run1}
IPORT=${IPORT:-8001}; WPORT=${WPORT:-8080}
args=(--model "$MODEL" --port "$IPORT")
label="$MODEL"
if [ "$ADAPTER" != "none" ]; then
  for f in adapter_config.json adapters.safetensors; do
    [ -f "$ADAPTER/$f" ] || { echo "Missing adapter file: $ADAPTER/$f" >&2; exit 1; }
  done
  args+=(--adapter-path "$ADAPTER")
  label="$MODEL + LoRA ($(basename "$ADAPTER"))"
fi
~/.local/bin/mlx_lm.server "${args[@]}" >/tmp/galsenai_mlx_server.log 2>&1 &
INFER=$!
trap 'kill $INFER 2>/dev/null; wait $INFER 2>/dev/null || true; echo "Serveurs arrêtés."' EXIT
trap 'exit 0' INT TERM
for _ in $(seq 60); do curl -sf "localhost:$IPORT/v1/models" >/dev/null && break; sleep 2; done
curl -sf "localhost:$IPORT/v1/models" >/dev/null || { echo "inference server failed, see /tmp/galsenai_mlx_server.log"; exit 1; }
echo "Gemma sur :$IPORT ($label). Chat : http://127.0.0.1:$WPORT"
# mlx_lm.server serves the preloaded model (with its adapter) only under the name "default_model".
MODEL=default_model ACTIVE_MODEL="$label" VLLM_BASE_URL="http://localhost:$IPORT/v1" .venv/bin/python webapp/server.py --port "$WPORT"
