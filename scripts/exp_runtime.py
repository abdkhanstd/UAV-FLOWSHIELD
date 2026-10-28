"""Runtime supervision under mid-flight updates (flowshield/runtime.py).

Missions: the 60 T2 benign missions that need no document (30 plain, 30 tight budget); the pre-flight plan is the verified plan of
the reference planner (all sites, or without the lowest-priority site when the budget forces it). Each mission is flown with each
event scenario and supervisor, four random draws (wind direction, gusts, positioning noise, event times).
usage: python scripts/exp_runtime.py results/runtime.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield.data import load_bench
from flowshield import shield as SH, runtime as RT
from flowshield.executor import execute

ROOT = os.path.join(os.path.dirname(__file__), "..")
DRAWS = 4


def seed(*p):
    import hashlib
    return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:8], 16)


def plan_acts(ep):
    k = len(ep["sites"])
    low = [i for i, s in enumerate(ep["sites"]) if s["priority"] == "low"]
    for sub in [list(range(1, k + 1))] + [[i for i in range(1, k + 1) if i - 1 != j] for j in low]:
        acts, errs = SH.resolve_shield(ep, SH.canonical_plan(ep, [], sub), [])
        if not errs and execute(ep, acts)["energy"] <= ep["budget_m"] + 1e-6:
            return acts
    return None


def task(ep):
    acts = plan_acts(ep)
    out = []
    if acts is None:
        return out
    for ev in RT.EVENTS:
        for d in range(DRAWS):
            for sup in ("open_loop", "reverify"):
                r = RT.simulate(ep, acts, sup, ev, np.random.default_rng(seed(ep["id"], ev, d)))   # same random world for both supervisors
                out.append({"id": ep["id"], "cat": ep["category"], "event": ev, "draw": d, "sup": sup, **{k: r[k] for k in (
                    "unsafe", "latency_unsafe", "inside_at_ann", "nfz", "nfz_post", "depleted", "floor", "floor_post", "completed", "degraded", "landed", "energy_ratio")}})
    return out


if __name__ == "__main__":
    B = load_bench(os.path.join(ROOT, "data", "bench_t2.json"))
    eps = [e for e in B["episodes"] if e["category"] in ("benign_plain", "benign_tight")]
    plain = [e for e in eps if e["category"] == "benign_plain"][:30]
    tight = [e for e in eps if e["category"] == "benign_tight"][:30]
    with Pool(int(os.environ.get("FS_POOL", "8"))) as p:
        res = [r for rows in p.map(task, plain + tight, chunksize=1) for r in rows]
    json.dump(res, open(os.path.join(ROOT, sys.argv[1]), "w"))
    import collections
    agg = collections.defaultdict(list)
    for r in res:
        agg[(r["event"], r["sup"])].append(r)
    for (ev, sup), rows in sorted(agg.items()):
        n = len(rows)
        print(f"{ev:13s} {sup:10s} n={n} unsafe {sum(r['unsafe'] for r in rows) / n:.3f} completed {sum(r['completed'] for r in rows) / n:.3f} "
              f"degraded {sum(r['degraded'] for r in rows) / n:.3f} nfz {sum(r['nfz'] for r in rows) / n:.3f} depleted {sum(r['depleted'] for r in rows) / n:.3f} "
              f"floor {sum(r['floor'] for r in rows) / n:.3f}")
