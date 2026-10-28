#!/bin/bash
# Follow-up LLM experiments on the remote machine, using both GPUs (private Ollama servers 11500 and 11501).
cd "$HOME/cae_exp/code" || exit 1
export FS_OLLAMA=http://127.0.0.1:11500,http://127.0.0.1:11501
export FS_OLLAMA_CAP=8,8
PY=python3
CATS=benign_plain,benign_constraint,benign_dynamic,benign_tight
mkdir -p results
# keep the local copies out of the way: every remote experiment starts from an empty LLM cache
[ -f results/cache_gemma4-12b_t0.0_s0.jsonl ] && mv results/cache_gemma4-12b_t0.0_s0.jsonl results/local_cache_copy.jsonl
[ -f results/main_gemma4-12b.jsonl ] && mv results/main_gemma4-12b.jsonl results/local_main_partial_copy.jsonl

# GPU temperature log (every 60 s)
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"; sleep 60; done ) > "$HOME/cae_exp/logs/gpu_temp.log" 2>&1 &

wait_model() { for i in $(seq 1 360); do if curl -s 127.0.0.1:11500/api/tags | grep -q "\"$1\""; then return 0; fi; sleep 10; done; return 1; }
wait_model gemma4:12b || { echo "gemma4:12b missing"; exit 1; }
echo "gemma4:12b ready $(date +%T)"

# 0. cross-machine consistency: same episodes and methods as the local main run, fresh cache
$PY scripts/run_llm.py --model gemma4:12b --split test --ids data/consist_ids.txt --methods undefended,prv,mcp_pep,flowshield \
  --out results/consist_mata.jsonl --threads 16
# 1. generalization beyond the templates: LLM-paraphrased documents
$PY scripts/run_llm.py --model gemma4:12b --split test --bench data/bench_para.json \
  --methods det_template,undefended,prompt_only,spotlight_guard,prv,eamsr,mcp_pep,flowshield \
  --out results/para_gemma4-12b.jsonl --threads 16
# 2. privacy-utility trade-off: fixed noise scales and the calibrated policy, benign episodes (same batch)
$PY scripts/run_llm.py --model gemma4:12b --split test --methods fs_s0,fs_s100,fs_s300,fs_s600,fs_s800,flowshield \
  --categories $CATS --out results/trade_gemma4-12b.jsonl --threads 16
# 3. sampled decoding: temperature 0.7, three seeds, headline methods on the ablation subset
for SEED in 1 2 3; do
  $PY scripts/run_llm.py --model gemma4:12b --split test --methods undefended,prv,mcp_pep,flowshield \
    --ids data/ablation_ids.txt --temperature 0.7 --seed $SEED --out results/samp_gemma4-12b_s$SEED.jsonl --threads 16
done
# 4. second backbone (different family): main methods and ablations
wait_model qwen3:8b || { echo "qwen3:8b missing"; exit 1; }
$PY scripts/run_llm.py --model qwen3:8b --split test \
  --methods det_template,undefended,prompt_only,spotlight_guard,prv,eamsr,mcp_pep,flowshield \
  --out results/main_qwen3-8b.jsonl --threads 16
$PY scripts/run_llm.py --model qwen3:8b --split test --methods fs_no_view,fs_no_typed,fs_no_envelope,fs_no_verifier \
  --ids data/ablation_ids.txt --out results/abl_qwen3-8b.jsonl --threads 16
echo "ALL_DONE $(date +%T)"
