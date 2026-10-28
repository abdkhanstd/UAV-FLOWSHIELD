"""Run the T3 pipelines (flowshield/pipeline_t3.py): identical to run_llm.py with the specification text, judge and pipelines
registered. usage: python scripts/run_t3.py --model gemma4:12b --bench data/bench_t3.json --scales data/scales_t3.json
--methods flowshield,fs_no_view,undefended,det_plus,t3_regex,t3_dsl_cov,t3_dsl_all,t3_llm_dslv --out results/eval/t3_gemma.jsonl"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import flowshield.pipeline_t3  # noqa: F401  registers the T3 pipelines
import run_llm

if __name__ == "__main__":
    run_llm.main()
