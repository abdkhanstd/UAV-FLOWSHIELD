"""Held-out evaluation of the certified rules on a third synthetic geography (data/bench_geo4.json: 12 fresh areas).

For each mission: the fixed-sequence ladder (cp_fs) and the table-first rule (cp_tab) are run at T draws (alpha 5 %, rho 10 %), and
the released scales, together with the public per-class table scale, are evaluated on fresh draws (exp_privacy3.held_task protocol).
usage: FS_BENCH=data/bench_geo4.json FS_POOL=24 python scripts/exp_geo4.py <T> results/geo4.json"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa
from exp_privacy3 import held_task
from flowshield.certify import reid_draws_ok, select_scale

CAP = {"dense": 300, "medium": 800, "sparse": 800}
TABLE = {"dense": 50, "medium": 150, "sparse": 600}
T = int(sys.argv[1]) if len(sys.argv) > 1 else 100          # read at import time (spawned workers re-import this module)


def calib(eid):
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    cls = ep["area"].split("_")[0]
    grid = [s for s in GRID if s <= CAP[cls]]
    out = {"id": eid, "area": ep["area"], "cls": cls, "k": len(idx)}
    for rule in ("cp_fs", "cp_tab"):
        rng = np.random.default_rng(seed(eid, "geo4", rule, T))
        r = select_scale(lambda s, n: reid_draws_ok(M, C, idx, H, s, rng, n), grid, 0.10, rule, rng, T=T, pilot=10, start=TABLE[cls])
        out[rule] = {"scale": r["scale"], "passed": bool(r["passed"]), "draws": r["draws"], "table": bool(r.get("table", False))}
    return out


if __name__ == "__main__":
    out = sys.argv[2]
    n_max = int(os.environ.get("FS_MAX", "0"))
    order = sorted(bench()["episodes"], key=lambda e: (e["area"], e["id"]))
    if n_max:
        order = order[::max(1, len(order) // n_max)][:n_max]
    ids = sorted((e["id"] for e in order), key=lambda i: 0 if episode(i)["area"].startswith("dense") else 1 if episode(i)["area"].startswith("medium") else 2)
    with Pool(int(os.environ.get("FS_POOL", "24"))) as p:
        cal = p.map(calib, ids, chunksize=1)
        cm = {c["id"]: c for c in cal}
        held = p.map(held_task, [(i, {"ladder": cm[i]["cp_fs"]["scale"], "tabfirst": cm[i]["cp_tab"]["scale"], "table": TABLE[cm[i]["cls"]]}) for i in ids], chunksize=3)
    json.dump({"T": T, "calibration": cal, "heldout": held}, open(os.path.join(ROOT, out), "w"))
    for cls in ("dense", "medium", "sparse"):
        c = [x for x in cal if x["cls"] == cls]
        h = [x for x in held if x["area"].startswith(cls)]
        print(cls, len(c), {k: round(float(np.mean([x["pol"][k]["site"] for x in h])), 4) for k in ("ladder", "tabfirst", "table")},
              "scale", {r: round(float(np.mean([x[r]["scale"] for x in c]))) for r in ("cp_fs", "cp_tab")})
