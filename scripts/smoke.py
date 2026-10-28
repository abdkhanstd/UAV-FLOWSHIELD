import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield.llm import LLM
from flowshield.pipeline import run_episode, METHODS

root = os.path.join(os.path.dirname(__file__), "..")
B = json.load(open(os.path.join(root, "data", "bench.json")))
eps = [e for e in B["episodes"] if e["split"] == "dev"]
model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:12b"
llm = LLM(model, os.path.join(root, "results", "smoke_cache.jsonl"))
pick = {}
for e in eps:
    key = e["category"] if e["category"] != "attack" else "attack_" + e["attack"]["goal"]
    pick.setdefault(key, e)
methods = sys.argv[2].split(",") if len(sys.argv) > 2 else METHODS
jobs = [(e, m) for e in pick.values() for m in methods]
t = time.time()


def go(a):
    e, m = a
    return run_episode(e, m, llm, scale=150.0)


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(go, jobs))
print("elapsed", time.time() - t, "calls", sum(r["n_calls"] for r in res))
for r in res:
    print(f"{r['id']} {r['category']:<17} {str(r['goal']):<5} {r['method']:<16} dec={r['decision']:<9} succ={int(r['success'])} "
          f"atk={int(r['attack_success'])} att={int(r['attempted'])} {r['fail_reason'] or ''} {r['reason'] or ''} "
          f"leakP={r['leak_abs_prompt']} calls={r['n_calls']}")
