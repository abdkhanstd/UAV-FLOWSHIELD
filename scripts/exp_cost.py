"""Experiment 6: system cost. Sequential (one request in flight) latency, tokens, cloud bytes, and edge compute.

Run on a quiet GPU server: FS_OLLAMA=http://127.0.0.1:11436 python scripts/exp_cost.py"""
import json
import os
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield.llm import LLM
from flowshield.pipeline import run_episode
from privacy_common import bench, matcher, episode, seed, GRID, ROOT
from flowshield.reid import reid_risk

METHODS = ["undefended", "prompt_only", "spotlight_guard", "prv", "eamsr", "mcp_pep", "flowshield"]
N_PER = 12


def pick():
    eps = [e for e in bench()["episodes"] if e["split"] == "test"]
    out = []
    for cat in (["benign_plain", "benign_constraint", "benign_dynamic", "benign_tight"], ["attack"], ["adaptive"]):
        sub = [e for e in eps if e["category"] in cat and not ((e["attack"] or {}).get("goal") == "AD4")]
        rng = np.random.default_rng(1)
        out += [sub[i] for i in rng.choice(len(sub), N_PER, replace=False)]
    return out


def main(model):
    scales = json.load(open(os.path.join(ROOT, "data", "scales.json")))["scales"]
    eps = pick()
    scratch = os.path.join(ROOT, "results", f"cost_cache_{int(time.time())}.jsonl")
    llm = LLM(model, scratch)
    # warm-up so that model load time is excluded
    run_episode(eps[0], "undefended", llm)
    llm.cache.clear()
    rows = []
    for e in eps:
        for m in METHODS:
            llm.cache.clear()
            t0 = time.perf_counter()
            r = run_episode(e, m, llm, scale=scales[e["id"]]["scale"])
            r["e2e"] = time.perf_counter() - t0
            r["trusted_s"] = r["e2e"] - r["llm_wall"]
            rows.append({k: r[k] for k in ("id", "method", "category", "e2e", "llm_wall", "trusted_s", "n_calls", "cloud_calls",
                                           "edge_calls", "pt", "ot", "cloud_up", "cloud_down", "call_walls")})
        print("done", e["id"], flush=True)
    # edge preflight: attack-calibration time per mission (single core) and peak memory
    proc = psutil.Process()
    cal = []
    for cls in ("dense", "medium", "sparse"):
        sub = [e for e in bench()["episodes"] if e["split"] == "test" and e["area"].startswith(cls)][:10]
        for e in sub:
            C, M = matcher(e["area"])
            idx = [s["cand_idx"] for s in e["sites"]]
            H = np.array(e["home"]["xy"])
            rng = np.random.default_rng(seed(e["id"], "cost"))
            t0 = time.perf_counter()
            for s in GRID:
                est, _ = reid_risk(M, C, idx, H, s, rng, 10, frame=True)
                if est <= 0.10:
                    break
            cal.append({"cls": cls, "seconds": time.perf_counter() - t0, "scale": s})
    mem = proc.memory_info()
    out = {"model": model, "rows": rows, "calibration": cal,
           "peak_rss_mb": getattr(mem, "peak_wset", mem.rss) / 1e6, "rss_mb": mem.rss / 1e6}
    json.dump(out, open(os.path.join(ROOT, "results", f"cost_{model.replace(':', '-')}.json"), "w"))
    for m in METHODS:
        x = [r for r in rows if r["method"] == m]
        e2e = [r["e2e"] for r in x]
        print(f"{m:16s} p50 {np.percentile(e2e, 50):6.2f}s p95 {np.percentile(e2e, 95):6.2f}s calls {np.mean([r['n_calls'] for r in x]):.2f} "
              f"up {np.mean([r['cloud_up'] for r in x]):.0f}B down {np.mean([r['cloud_down'] for r in x]):.0f}B trusted {np.mean([r['trusted_s'] for r in x]) * 1000:.1f}ms")
    for cls in ("dense", "medium", "sparse"):
        v = [c["seconds"] for c in cal if c["cls"] == cls]
        print("calibration", cls, "median", np.median(v), "p95", np.percentile(v, 95))
    print("peak rss MB", out["peak_rss_mb"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gemma4:12b")
