#!/bin/bash
# Start a private Ollama server for the CAE experiments on ONE GPU, with its own model directory.
# Does not touch the machine's default Ollama service.   usage: start_ollama.sh <gpu-uuid> [port]
GPU_UUID="${1:?gpu uuid}"
PORT="${2:-11500}"
BASE="$HOME/cae_exp"
mkdir -p "$BASE/ollama_models" "$BASE/logs"
if curl -s "127.0.0.1:$PORT/api/version" > /dev/null; then echo "already running on $PORT"; exit 0; fi
CUDA_VISIBLE_DEVICES="$GPU_UUID" OLLAMA_VULKAN=0 OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_MODELS="$BASE/ollama_models" \
OLLAMA_NUM_PARALLEL=8 OLLAMA_KEEP_ALIVE=24h setsid nohup ollama serve > "$BASE/logs/ollama_$PORT.log" 2>&1 < /dev/null &
echo $! > "$BASE/logs/ollama_$PORT.pid"
sleep 8
curl -s "127.0.0.1:$PORT/api/version"; echo
grep -E "inference compute|overrode|library=|pci" "$BASE/logs/ollama_$PORT.log" | cut -c1-230 | head -6
