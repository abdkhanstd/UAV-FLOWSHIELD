"""Numerical check of the triangle inequality of the router metric with one to three no-fly circles (the proof of the diversion bound needs
a shortest-path metric; the sequential-detour router is exact for one circle). usage: python scripts/exp_router_triangle.py results/router_triangle.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield.geo import route

ROOT = os.path.join(os.path.dirname(__file__), "..")


def run(rng, nz, overlap, n):
    viol = tot = 0
    worst = 0.0
    for _ in range(n):
        c0 = rng.uniform(1500, 4500, 2)
        nf = [(c0 + rng.uniform(-1, 1, 2) * (500 if overlap else 3000), rng.uniform(250, 450)) for _ in range(nz)]
        pts = rng.uniform(0, 6000, (3, 2))
        if any(np.linalg.norm(p - c) < r + 35 for p in pts for c, r in nf):
            continue
        a, b, c = pts
        ex = route(a, c, nf)[1] - route(a, b, nf)[1] - route(b, c, nf)[1]
        tot += 1
        if ex > 1e-6:
            viol += 1
            worst = max(worst, float(ex))
    return {"zones": nz, "overlapping": overlap, "triples": tot, "violations": viol, "worst_excess_m": worst}


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    res = [run(rng, nz, ov, 6000) for nz, ov in ((1, False), (2, False), (3, False), (2, True), (3, True))]
    json.dump(res, open(os.path.join(ROOT, sys.argv[1]), "w"), indent=1)
    for r in res:
        print(r)
