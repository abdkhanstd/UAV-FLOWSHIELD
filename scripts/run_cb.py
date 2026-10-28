"""Run the common-boundary variants (<baseline>_cb, flowshield/pipeline_cb.py): identical to run_llm.py, with the variants
registered. usage: python scripts/run_cb.py --model gemma4:12b --split all --bench data/bench_t2.json
--methods undefended_cb,prompt_only_cb,spotlight_guard_cb,prv_cb,eamsr_cb,mcp_pep_cb --out results/eval/t2_gemma_cb.jsonl"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import flowshield.pipeline_cb  # noqa: F401  registers the _cb methods
import run_llm

if __name__ == "__main__":
    run_llm.main()
