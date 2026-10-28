#!/bin/bash
# T1, after the T2 runs on the local GPUs: qwen3:8b FlowShield family with the mean-rule scales, and FlowShield with the
# upper-limit scales (if data/scales_ucb.json exists) for both backbones.
cd "$(dirname "$0")/.." || exit 1
export FS_OLLAMA=http://127.0.0.1:11435,http://127.0.0.1:11436
export FS_OLLAMA_CAP=8,8
R=results/eval
FSALL=flowshield,fs_chan,fs_legacy,fs_noauthority,fs_fallback,fs_detx,fs_hybrid,fs_no_view,fs_no_view_euclid,fs_no_typed,fs_no_envelope,fs_no_verifier,fs_raw_noverifier
until grep -q "PC_DONE" results/logs/run_t2_pc.log; do sleep 60; done
echo "T1 qwen FS family $(date +%T)"
python scripts/run_llm.py --model qwen3:8b --split test --scales data/scales.json --methods $FSALL --out $R/t1_qwen_fs_mean10.jsonl --threads 16
if [ -f data/scales_ucb.json ]; then
  echo "T1 upper-limit scales $(date +%T)"
  python scripts/run_llm.py --model gemma4:12b --split test --scales data/scales_ucb.json --methods flowshield,fs_fallback --out $R/t1_gemma_fs_ucb.jsonl --threads 16
  python scripts/run_llm.py --model qwen3:8b --split test --scales data/scales_ucb.json --methods flowshield,fs_fallback --out $R/t1_qwen_fs_ucb.jsonl --threads 16
fi
echo "T1_PC2_DONE $(date +%T)"
