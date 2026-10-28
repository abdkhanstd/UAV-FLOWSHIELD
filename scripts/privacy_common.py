import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield.reid import ShapeMatcher, release_view, tau_for_scale, correct_ids, reid_risk
from flowshield.geo import planar_laplace

ROOT = os.path.join(os.path.dirname(__file__), "..")
GRID = [10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800]
_BENCH = None
_MATCH = {}


def bench():
    global _BENCH
    if _BENCH is None:
        _BENCH = json.load(open(os.path.join(ROOT, os.environ.get("FS_BENCH", "data/bench.json"))))
        from flowshield.data import sign_docs      # client-portal tickets carry their origin tag
        for e in _BENCH["episodes"]:
            sign_docs(e)
    return _BENCH


def matcher(area):
    if area not in _MATCH:
        while len(_MATCH) >= 2:          # bound the memory of long-running workers
            _MATCH.pop(next(iter(_MATCH)))
        C = np.array(bench()["areas"][area]["C"])
        _MATCH[area] = (C, ShapeMatcher(C))
    return _MATCH[area]


def episode(eid):
    for e in bench()["episodes"]:
        if e["id"] == eid:
            return e


def seed(*parts):
    import hashlib
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def aligned_nearest(M, C, idx, scale, rng):
    """Frame-compromise bound: attacker knows the exact alignment and picks the nearest registry point."""
    S = C[idx]
    Vn = S + planar_laplace(rng, len(S), 2.0 / scale if scale > 0 else 0)
    _, a = M.tree.query(Vn)
    return correct_ids(a, idx, C).mean()
