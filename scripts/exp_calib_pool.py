"""Pools of simulated attack outcomes for the coverage study of the calibration rules.

For a sample of missions and every scale of the grid, N independent simulated registration attacks are run and the
per-site identification matrix is stored (as bit masks). The pools are the 'population' against which the calibration
procedures are evaluated in exp_calib_coverage.py, so the statistical behaviour of a rule can be measured without
rerunning the attack for every replication.

usage: FS_BENCH=data/bench.json FS_KD_WORKERS=1 FS_POOL=48 python scripts/exp_calib_pool.py results/calib_pool_t1.jsonl [N] [per_area] [split]
Dense registries are pooled only up to 300 m (no development or test mission needed more; the cap of the ladder is stated
in the paper), medium and sparse registries over the whole grid.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa
from flowshield.certify import reid_draws_ok

CAP = {"dense": 300, "medium": 800, "sparse": 800}


def pool_task(args):
    eid, N = args
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    cls = ep["area"].split("_")[0]
    out = {"id": eid, "area": ep["area"], "k": len(idx), "N": N, "scales": {}}
    for s in GRID:
        if s > CAP[cls]:
            continue
        rng = np.random.default_rng(seed(eid, "pool", s))
        ok = reid_draws_ok(M, C, idx, H, s, rng, N)
        out["scales"][str(s)] = [int(sum(int(b) << j for j, b in enumerate(row))) for row in ok]
    return out


def pick(split, per_area):
    cnt, ids = {}, []
    for e in sorted(bench()["episodes"], key=lambda x: (x["area"], x["id"])):
        if split != "all" and e["split"] != split:
            continue
        if not (e["category"].startswith("benign") or e["category"] in ("attack", "adaptive")):
            continue
        if cnt.get(e["area"], 0) < per_area:
            ids.append(e["id"])
            cnt[e["area"]] = cnt.get(e["area"], 0) + 1
    return ids, cnt


if __name__ == "__main__":
    out = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    per_area = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    split = sys.argv[4] if len(sys.argv) > 4 else "test"
    ids, cnt = pick(split, per_area)
    done = set()
    if os.path.exists(os.path.join(ROOT, out)):
        for line in open(os.path.join(ROOT, out)):
            done.add(json.loads(line)["id"])
    todo = [i for i in ids if i not in done]
    # expensive (dense) missions first
    todo.sort(key=lambda i: 0 if episode(i)["area"].startswith("dense") else 1 if episode(i)["area"].startswith("medium") else 2)
    print("missions", len(ids), cnt, "todo", len(todo), flush=True)
    with Pool(int(os.environ.get("FS_POOL", "24"))) as p, open(os.path.join(ROOT, out), "a") as f:
        for n, r in enumerate(p.imap_unordered(pool_task, [(i, N) for i in todo], chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if n % 10 == 0:
                print(n + 1, "/", len(todo), flush=True)
    print("done", flush=True)
