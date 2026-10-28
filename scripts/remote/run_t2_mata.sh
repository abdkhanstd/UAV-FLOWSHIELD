#!/bin/bash
# One pass over the fresh benchmark T2 with the frozen pipeline (remote machine, both GPUs).
# Order: main model, second model, mean-calibrated scales, edge extractors, network loss, small planners.
cd "$HOME/cae_exp/code" || exit 1
export FS_OLLAMA=http://127.0.0.1:11500,http://127.0.0.1:11501
export FS_OLLAMA_CAP=8,8
PY=python3
B=data/bench_t2.json
S=data/scales_t2.json
R=results/eval
mkdir -p $R
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"; sleep 60; done ) > "$HOME/cae_exp/logs/gpu_temp_t2.log" 2>&1 &
TEMP_PID=$!
ALL=det_template,det_plus,undefended,prompt_only,spotlight_guard,prv,eamsr,mcp_pep,undefended_rep,prompt_only_rep,spotlight_guard_rep,prv_rep,eamsr_rep,eamsr_repdel,mcp_pep_rep,mcp_pep_repdel,flowshield,fs_chan,fs_legacy,fs_noauthority,fs_fallback,fs_detx,fs_hybrid,fs_no_view,fs_no_view_euclid,fs_no_typed,fs_no_envelope,fs_no_verifier,fs_raw_noverifier
FSALL=flowshield,fs_chan,fs_legacy,fs_noauthority,fs_fallback,fs_detx,fs_hybrid,fs_no_view,fs_no_view_euclid,fs_no_typed,fs_no_envelope,fs_no_verifier,fs_raw_noverifier
SMALL=undefended,prv,mcp_pep,mcp_pep_repdel,flowshield,fs_fallback,det_plus
wait_model() { for i in $(seq 1 360); do if curl -s 127.0.0.1:11500/api/tags | grep -q "\"$1\""; then return 0; fi; sleep 10; done; return 1; }

wait_model gemma4:12b || exit 1
echo "T2 gemma4:12b $(date +%T)"
$PY scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales $S --methods $FSALL --out $R/t2_gemma_fs.jsonl --threads 16
echo "T2 qwen3:8b $(date +%T)"
wait_model qwen3:8b || exit 1
$PY scripts/run_llm.py --model qwen3:8b --split all --bench $B --scales $S --methods $ALL --out $R/t2_qwen.jsonl --threads 16
if [ -f data/scales_t2_mean10.json ]; then
  echo "T2 mean-calibrated scales $(date +%T)"
  $PY scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales data/scales_t2_mean10.json --methods flowshield --out $R/t2_gemma_mean10.jsonl --threads 16
fi
for X in granite4.1:3b qwen3:1.7b nemotron-3-nano:4b qwen3:8b; do
  wait_model $X || continue
  echo "T2 edge extractor $X $(date +%T)"
  $PY scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales $S --methods flowshield --edge-model $X --out $R/t2_gemma_edge_$(echo $X | tr ':/' '__').jsonl --threads 16
done
echo "T2 fixed noise scales $(date +%T)"
$PY scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales $S --methods fs_s0,fs_s100,fs_s300,fs_s800   --categories benign_plain,benign_constraint,benign_dynamic,benign_tight,benign_multi --out $R/t2_gemma_fixed.jsonl --threads 16
for P in 0.1 0.3 0.5 1.0; do
  echo "T2 network loss $P $(date +%T)"
  $PY scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales $S --methods flowshield,fs_fallback \
    --categories benign_plain,benign_constraint,benign_dynamic,benign_tight,benign_multi --opts "{\"drop_p\": $P}" --out $R/t2_gemma_net$P.jsonl --threads 16
done
for M in granite4.1:3b qwen3:1.7b nemotron-3-nano:4b; do
  wait_model $M || continue
  echo "T2 small planner $M $(date +%T)"
  $PY scripts/run_llm.py --model $M --split all --bench $B --scales $S --methods $SMALL --out $R/t2_$(echo $M | tr ':/' '__').jsonl --threads 16
done
kill $TEMP_PID
echo "ALL_DONE $(date +%T)"
