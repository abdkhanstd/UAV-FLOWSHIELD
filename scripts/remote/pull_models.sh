#!/bin/bash
# Pull one model into MY private Ollama store (server on port 11500). usage: pull_models.sh <model>
export OLLAMA_HOST=127.0.0.1:11500
ollama pull "${1:?model}"
echo "PULL_DONE $1"
