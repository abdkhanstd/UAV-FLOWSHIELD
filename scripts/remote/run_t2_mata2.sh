#!/bin/bash
# T2 on the remote machine, second part: waits for the qwen baseline pass and for the scale file, then runs the
# FlowShield family with qwen3:8b, the smaller planners and the sampled-decoding runs on the stratified subset.
cd "$HOME/cae_exp/code" || exit 1
python3 scripts/remote/verify_frozen.py || exit 1
export FS_OLLAMA=http://127.0.0.1:11500,http://127.0.0.1:11501
export FS_OLLAMA_CAP=8,8
B=data/bench_t2.json
R=results/eval
FSALL=flowshield,fs_chan,fs_legacy,fs_noauthority,fs_fallback,fs_detx,fs_hybrid,fs_no_view,fs_no_view_euclid,fs_no_typed,fs_no_envelope,fs_no_verifier,fs_raw_noverifier
SMALL=undefended,prv,mcp_pep,mcp_pep_repdel,flowshield,fs_fallback,det_plus
wait_model() { for i in $(seq 1 360); do if curl -s 127.0.0.1:11500/api/tags | grep -q "\"$1\""; then return 0; fi; sleep 10; done; return 1; }
until grep -q "T2_QWEN_BASE_DONE" "$HOME/cae_exp/logs/run_t2_qwen_base.out"; do sleep 30; done
until [ -f data/scales_t2.json ]; do sleep 30; done
echo "qwen FS family $(date +%T)"
python3 scripts/run_llm.py --model qwen3:8b --split all --bench $B --scales data/scales_t2.json --methods $FSALL --out $R/t2_qwen_fs.jsonl --threads 16
for M in granite4.1:3b qwen3:1.7b nemotron-3-nano:4b; do
  wait_model $M || continue
  echo "small planner $M $(date +%T)"
  python3 scripts/run_llm.py --model $M --split all --bench $B --scales data/scales_t2.json --methods $SMALL --out $R/t2_$(echo $M | tr ':/' '__').jsonl --threads 16
done
wait_model gemma4:12b
echo "sampled decoding $(date +%T)"
python3 scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales data/scales_t2.json --ids data/t2_subset_ids.txt --methods undefended,prv,mcp_pep_repdel,flowshield --out $R/t2_samp_s0.jsonl --threads 16
for SEED in 1 2 3; do
  python3 scripts/run_llm.py --model gemma4:12b --split all --bench $B --scales data/scales_t2.json --ids data/t2_subset_ids.txt --methods undefended,prv,mcp_pep_repdel,flowshield --temperature 0.7 --seed $SEED --out $R/t2_samp_s$SEED.jsonl --threads 16
done
echo "MATA2_DONE $(date +%T)"
