#!/bin/bash
# T2, second backbone (qwen3:8b), methods that do not depend on the noise scales.
cd "$HOME/cae_exp/code" || exit 1
python3 scripts/remote/verify_frozen.py || exit 1
export FS_OLLAMA=http://127.0.0.1:11500,http://127.0.0.1:11501
export FS_OLLAMA_CAP=8,8
mkdir -p results/eval
for i in $(seq 1 120); do curl -s 127.0.0.1:11500/api/tags | grep -q qwen3:8b && break; sleep 5; done
python3 scripts/run_llm.py --model qwen3:8b --split all --bench data/bench_t2.json \
  --methods det_template,det_plus,undefended,prompt_only,spotlight_guard,prv,eamsr,mcp_pep,undefended_rep,prompt_only_rep,spotlight_guard_rep,prv_rep,eamsr_rep,eamsr_repdel,mcp_pep_rep,mcp_pep_repdel \
  --out results/eval/t2_qwen_base.jsonl --threads 16
echo "T2_QWEN_BASE_DONE $(date +%T)"
