"""Build FlowShield-Bench: synthetic mission areas, benign episodes, injection episodes.

Splits are made by mission area and mission family before attack payloads are attached:
areas ending in _1 and _2 form the development split (threshold/prompt tuning, calibration
sanity checks); areas ending in _3 form the held-out test split.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield import data as D
from flowshield import attacks as A


def build(seed, counts, out_path):
    rng = np.random.default_rng(seed)
    areas = D.make_areas(seed=7)
    names = sorted(areas)
    test_areas = [a for a in names if a.endswith("_3")]
    dev_areas = [a for a in names if not a.endswith("_3")]
    eps = []

    def add(split, category, cfg=None):
        pool = test_areas if split == "test" else dev_areas
        area = pool[int(rng.integers(len(pool)))]
        k = int(rng.choice([3, 4, 4, 5]))
        m = D.make_mission(rng, areas, area, k, tight=(category == "benign_tight"))
        nm = [f"S{i + 1}" for i in range(k)]
        docs, attack = D.gen_docs(rng, m, category, nm, cfg)
        eid = f"E{len(eps):04d}"
        eps.append(D.to_episode(rng, m, eid, category, docs, attack, split))

    for split in ("test", "dev"):
        n = counts[split]
        for cat in ("benign_plain", "benign_constraint", "benign_dynamic", "benign_tight"):
            for _ in range(n["benign"]):
                add(split, cat)
        for g in A.GOALS:
            for st in A.STYLES:
                for ch in A.CHANNELS:
                    for _ in range(n["attack_rep"]):
                        add(split, "attack", (g, st, ch))
        for g, st in (("AD1", "blended"), ("AD2", "authority"), ("AD3", "authority"), ("AD4", "blended")):
            for i in range(n["adaptive"]):
                add(split, "adaptive", (g, st, A.CHANNELS[i % 3]))
    json.dump({"areas": {k: {"cls": v["cls"], "C": v["C"].tolist(), "offset": v["offset"].tolist()}
                         for k, v in areas.items()}, "episodes": eps}, open(out_path, "w"))
    return eps


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "bench.json")
    counts = {"test": {"benign": 30, "attack_rep": 2, "adaptive": 15}, "dev": {"benign": 4, "attack_rep": 1, "adaptive": 3}}
    eps = build(11, counts, out)
    from collections import Counter
    print(len(eps), Counter((e["split"], e["category"]) for e in eps))
