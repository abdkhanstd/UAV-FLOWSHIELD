#!/bin/bash
# Stop the T2 baseline run on this machine (my own run_llm process only) and keep its partial output aside.
pkill -u "$USER" -f "run_t2_base.sh"
pkill -u "$USER" -f "scripts/run_llm.py --model gemma4:12b --split all --bench data/bench_t2.json"
sleep 2
cd "$HOME/cae_exp/code/results/eval" && [ -f t2_gemma_base.jsonl ] && mv t2_gemma_base.jsonl t2_gemma_base_mata_partial.jsonl
pgrep -u "$USER" -fa "run_llm" | head -3
echo stopped
