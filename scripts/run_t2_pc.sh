#!/bin/bash
# T2 with the main backbone on the local GPUs, after the baseline pass: waits for the scale file, then runs the FlowShield
# family, the mean-calibrated variant, fixed noise scales, edge extractors, network loss and the optimization-driven attacker.
cd "$(dirname "$0")/.." || exit 1
python scripts/remote/verify_frozen.py || exit 1
export FS_OLLAMA=http://127.0.0.1:11435,http://127.0.0.1:11436
export FS_OLLAMA_CAP=8,8
B=data/bench_t2.json
R=results/eval
FSALL=flowshield,fs_chan,fs_legacy,fs_noauthority,fs_fallback,fs_detx,fs_hybrid,fs_no_view,fs_no_view_euclid,fs_no_typed,fs_no_envelope,fs_no_verifier,fs_raw_noverifier
BEN=benign_plain,benign_constraint,benign_dynamic,benign_tight,benign_multi
until grep -q "^finished" results/logs/t2_gemma_base.log; do sleep 30; done
until [ -f data/scales_t2.json ]; do sleep 30; done
echo "FS family $(date +%T)"
python scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales data/scales_t2.json --methods $FSALL --out $R/t2_gemma_fs.jsonl --threads 16
[ -f data/scales_t2_alt.json ] && python scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales data/scales_t2_alt.json --methods flowshield --out $R/t2_gemma_alt.jsonl --threads 16
echo "fixed scales $(date +%T)"
python scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales data/scales_t2.json --methods fs_s0,fs_s100,fs_s300,fs_s800 --categories $BEN --out $R/t2_gemma_fixed.jsonl --threads 16
for P in 0.1 0.3 0.5 1.0; do
  echo "network loss $P $(date +%T)"
  python scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales data/scales_t2.json --methods flowshield,fs_fallback --categories $BEN --opts "{\"drop_p\": $P}" --out $R/t2_gemma_net$P.jsonl --threads 16
done
for X in granite4.1:3b qwen3:1.7b nemotron-3-nano:4b qwen3:8b; do
  echo "edge extractor $X $(date +%T)"
  python scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales data/scales_t2.json --methods flowshield --edge-model $X --out $R/t2_gemma_edge_$(echo $X | tr ':/' '__').jsonl --threads 16
done
echo "optimization-driven attacker $(date +%T)"
python scripts/exp_optattack.py --bench $B --target-model gemma4:12b --attacker-model gemma4:12b --scales data/scales_t2.json --rounds 10 --per-goal 5 --out $R/optattack_gemma.jsonl --threads 16
echo "PC_DONE $(date +%T)"
