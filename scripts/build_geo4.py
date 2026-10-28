"""A third synthetic geography for the privacy evaluation: 12 fresh areas (4 per registry class, new seed, new registry
draws) with 10 plain missions each (3 to 6 sites). No language model is involved; the file has the layout of the other
benchmarks so that the privacy scripts (FS_BENCH=data/bench_geo4.json) run on it unchanged.

usage: python scripts/build_geo4.py --out data/bench_geo4.json"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield import data as D

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bench_geo4.json")
    ap.add_argument("--seed", type=int, default=4141)
    ap.add_argument("--per-area", type=int, default=10)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    areas = D.make_areas(seed=a.seed + 1, per_class=4, tag="g")
    eps = []
    for name in areas:
        for j in range(a.per_area):
            k = int(rng.choice([3, 4, 5, 6]))
            m = D.make_mission(rng, areas, name, k)
            eps.append(D.to_episode(rng, m, f"G{len(eps):04d}", "plain", [], None, "test"))
    out = {"areas": {k: {"cls": v["cls"], "C": v["C"].tolist(), "offset": v["offset"].tolist()} for k, v in areas.items()}, "episodes": eps}
    json.dump(out, open(os.path.join(ROOT, a.out), "w"))
    print(len(eps), "episodes in", len(areas), "areas")


if __name__ == "__main__":
    main()
