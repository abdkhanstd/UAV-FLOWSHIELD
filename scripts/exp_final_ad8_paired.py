"""Paired check of AD8 in the final configuration: the same episodes without the attacker's document, same operator draws."""
import copy, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import flowshield.pipeline_final as PF
from flowshield.data import load_bench
from flowshield.llm import LLM
B = load_bench("data/bench_t2.json")
llm = LLM("gemma4:12b", os.path.join(ROOT, "results", "cache_gemma4-12b_t0.0_s0.jsonl"), temperature=0.0, seed=0)
out = []
for ep in B["episodes"]:
    if (ep["attack"] or {}).get("goal") != "AD8":
        continue
    e2 = copy.deepcopy(ep)
    e2["docs"] = [d for d in e2["docs"] if d.get("attack") is None]
    e2["attack"] = None
    for m in PF.METHODS:
        r = PF.run_episode(e2, m, llm)
        out.append({"id": ep["id"], "method": m, "success": r["success"], "fail": r["fail_reason"], "ticket": r.get("fin_ticket"), "n_calls_new": None})
json.dump(out, open(os.path.join(ROOT, "results", "eval", "t2_final_ad8_paired.json"), "w"), indent=1)
for o in out:
    print(o)
