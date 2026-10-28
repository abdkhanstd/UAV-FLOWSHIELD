#!/bin/bash
# T1: FlowShield with the mean05 scales (the default selected by the rule), both backbones; runs after run_t1_pc2.sh.
cd "$(dirname "$0")/.." || exit 1
export FS_OLLAMA=http://127.0.0.1:11435,http://127.0.0.1:11436
export FS_OLLAMA_CAP=8,8
R=results/eval
until grep -q "T1_PC2_DONE" results/logs/run_t1_pc2.log; do sleep 60; done
python scripts/run_llm.py --model gemma4:12b --split test --scales data/scales_mean05_t1.json --methods flowshield,fs_fallback --out $R/t1_gemma_fs_mean05.jsonl --threads 16
python scripts/run_llm.py --model qwen3:8b --split test --scales data/scales_mean05_t1.json --methods flowshield,fs_fallback --out $R/t1_qwen_fs_mean05.jsonl --threads 16
echo "T1_PC3_DONE $(date +%T)"
