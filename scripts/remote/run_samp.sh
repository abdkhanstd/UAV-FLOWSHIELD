#!/bin/bash
# Sampled decoding (temperature 0.7), seeds given as arguments, headline methods on the ablation subset.
cd "$HOME/cae_exp/code" || exit 1
export FS_OLLAMA=http://127.0.0.1:11500,http://127.0.0.1:11501
export FS_OLLAMA_CAP=8,8
for SEED in "$@"; do
  python3 scripts/run_llm.py --model gemma4:12b --split test --methods undefended,prv,mcp_pep,flowshield \
    --ids data/ablation_ids.txt --temperature 0.7 --seed $SEED --out results/samp_gemma4-12b_s$SEED.jsonl --threads 16
done
echo "SAMP_DONE $(date +%T)"
