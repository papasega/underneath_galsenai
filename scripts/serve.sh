#!/usr/bin/env bash
# Start the local inference server (mlx_lm.server, OpenAI-compatible) and the chat web app together.
# vLLM does not run on Apple Silicon: on a GPU machine start `vllm serve MODEL --port 8000` yourself and run
# `VLLM_BASE_URL=http://HOST:8000/v1 python3 webapp/server.py` instead of this script.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-mlx-community/gemma-4-E2B-it-qat-4bit}
ADAPTER=${ADAPTER:-data/private/adapters/gemma4_e2b_galsenai}
IPORT=${IPORT:-8001}; WPORT=${WPORT:-8080}
args=(--model "$MODEL" --port "$IPORT")
[ -d "$ADAPTER" ] && args+=(--adapter-path "$ADAPTER")
~/.local/bin/mlx_lm.server "${args[@]}" >/tmp/galsenai_mlx_server.log 2>&1 &
INFER=$!
trap 'kill $INFER 2>/dev/null' EXIT
for _ in $(seq 60); do curl -sf "localhost:$IPORT/v1/models" >/dev/null && break; sleep 2; done
curl -sf "localhost:$IPORT/v1/models" >/dev/null || { echo "inference server failed, see /tmp/galsenai_mlx_server.log"; exit 1; }
echo "inference up on :$IPORT, chat on http://127.0.0.1:$WPORT"
# mlx_lm.server serves the preloaded model (with its LoRA adapter) only for the name "default_model".
MODEL=default_model VLLM_BASE_URL="http://localhost:$IPORT/v1" python3 -I webapp/server.py --port "$WPORT"
