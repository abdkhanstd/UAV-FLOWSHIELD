#!/bin/bash
# Build the fresh test benchmark T2 on the remote machine (authoring model on the private server 11500).
cd "$HOME/cae_exp/code" || exit 1
export FS_OLLAMA=http://127.0.0.1:11500
export FS_OLLAMA_CAP=8
rm -f results/cache_authoring.jsonl
python3 scripts/build_bench2.py --author-model granite4.1:3b --out data/bench_t2.json > results/logs_build_t2.txt 2>&1
tail -3 results/logs_build_t2.txt
sha256sum data/bench_t2.json
