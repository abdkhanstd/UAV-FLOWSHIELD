#!/bin/sh
# Follow-up LLM experiments, run sequentially after the main gemma run (single GPU: nvidia-smi GPU 1 only).
cd "$(dirname "$0")/.."
set -e
CATS=benign_plain,benign_constraint,benign_dynamic,benign_tight

# 1. generalization beyond the templates: LLM-paraphrased documents (numbers in words, new structure)
python scripts/run_llm.py --model gemma4:12b --split test --bench data/bench_para.json \
  --methods det_template,undefended,prompt_only,spotlight_guard,prv,eamsr,mcp_pep,flowshield \
  --out results/para_gemma4-12b.jsonl --threads 8

# 2. privacy-utility trade-off: fixed noise scales on benign episodes
python scripts/run_llm.py --model gemma4:12b --split test --methods fs_s0,fs_s100,fs_s300,fs_s600,fs_s800 \
  --categories $CATS --out results/trade_gemma4-12b.jsonl --threads 8

# 3. second backbone (different model family): main methods and ablations
python scripts/run_llm.py --model qwen3:8b --split test \
  --methods det_template,undefended,prompt_only,spotlight_guard,prv,eamsr,mcp_pep,flowshield \
  --out results/main_qwen3-8b.jsonl --threads 8
python scripts/run_llm.py --model qwen3:8b --split test --methods fs_no_view,fs_no_typed,fs_no_envelope,fs_no_verifier \
  --ids data/ablation_ids.txt --out results/abl_qwen3-8b.jsonl --threads 8

# 4. sampled decoding: temperature 0.7, three seeds, headline methods on the ablation subset
for SEED in 1 2 3; do
  python scripts/run_llm.py --model gemma4:12b --split test --methods undefended,prv,mcp_pep,flowshield \
    --ids data/ablation_ids.txt --temperature 0.7 --seed $SEED --out results/samp_gemma4-12b_s$SEED.jsonl --threads 8
done
