#!/bin/bash
# Regenerate every table, figure and macro of the paper from the raw results (no LLM calls; cached calls are not needed).
cd "$(dirname "$0")/.." || exit 1
set -e
python scripts/analyze_eval.py --tag tbg --name T2 --files results/eval/t2_gemma_base.jsonl results/eval/t2_gemma_fs.jsonl results/eval/t2_gemma_fixed.jsonl
python scripts/analyze_eval.py --tag tbq --name T2 --files results/eval/t2_qwen_base.jsonl results/eval/t2_qwen_fs.jsonl
python scripts/analyze_eval.py --tag tag --name T1 --files results/eval/t1_gemma_base.jsonl results/eval/t1_gemma_fs_mean10.jsonl
python scripts/analyze_eval.py --tag taq --name T1 --files results/eval/t1_qwen_base.jsonl results/eval/t1_qwen_fs_mean10.jsonl
python scripts/analyze_goal.py tbg; python scripts/analyze_goal.py tag; python scripts/fig_eval.py tbg
python scripts/analyze_priv.py --tag ta --priv results/privacy3_mata.json --extra results/privacy3_fixedx.json --attackers results/privacy4_mata.json
python scripts/analyze_priv.py --tag tb --priv results/privacy3_t2_heldout.json --regret results/regret_t2.json --extra results/privacy3_t2_mean05.json results/privacy3_cert100_t2.json
python scripts/analyze_cert.py
python scripts/analyze_calib_coverage.py results/calib_cov_t1.json ta
python scripts/analyze_p5.py; python scripts/fig_frontier.py results/privacy5_C_bonf.json b
python scripts/analyze_misc.py; python scripts/analyze_regret.py; python scripts/analyze_trade.py
python scripts/analyze_net_edge.py; python scripts/analyze_samp.py; python scripts/analyze_opt.py results/eval/optattack_gemma.jsonl tb; python scripts/analyze_opt.py results/eval/optattack_qwen_attacker.jsonl tq
python scripts/analyze_scalepol.py; python scripts/analyze_cost.py
python scripts/analyze_cb.py; python scripts/analyze_cluster.py
python scripts/analyze_t3.py results/eval/t3_gemma.jsonl tb; python scripts/analyze_t3.py results/eval/t3_qwen.jsonl tq; python scripts/analyze_t3.py results/eval/t3h_gemma.jsonl hb bench_t3h.json; python scripts/analyze_t3.py results/eval/t3_gemma.jsonl tr bench_t3h.json screened; python scripts/analyze_t3.py results/eval/t3h_gemma.jsonl hr bench_t3h.json screened; python scripts/analyze_t3.py results/eval/t3h_qwen.jsonl hq bench_t3h.json screened; python scripts/analyze_t3.py results/eval/t3_qwen.jsonl tw bench_t3h.json screened
python scripts/analyze_runtime.py; python scripts/exp_confirm.py; python scripts/analyze_authority.py; python scripts/analyze_calib_tab.py; python scripts/analyze_geo4.py; python scripts/make_chronology.py; python scripts/analyze_final.py; python scripts/analyze_cert_split.py
