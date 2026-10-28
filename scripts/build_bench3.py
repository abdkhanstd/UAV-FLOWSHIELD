"""Build FlowShield-Bench T3: missions with open-ended route specifications from the operator (no external documents).

Question answered: on which kinds of specification does an LLM planner add anything to an LLM extractor followed by an exact
solver behind the same trusted checks? Nine specification types (specs.py): five expressible in a small constraint language
(COVERED), four not (UNCOVERED). Single-spec episodes (12 per type) and two-spec combinations (12 each of
covered+covered, covered+uncovered, uncovered+uncovered). Larger assignments than T1/T2 (4-6 sites). Half of the
specification texts are rewritten by a language model of another family (granite4.1:3b) from the template wording, with a
check that every site name and number survives; the rest use the two template wordings.

usage: python scripts/build_bench3.py [--author-model granite4.1:3b] --out data/bench_t3.json
"""
import argparse
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield import data as D
from flowshield import specs as SP
from flowshield.llm import LLM, parse_json

ROOT = os.path.join(os.path.dirname(__file__), "..")
CLS_SCALE = {"dense": 50, "medium": 150, "sparse": 600}     # per-class table of the paper (noise scale of the private view)


def feasible_margins(ep, types):
    S = {s["name"]: np.asarray(s["xy"]) for s in ep["sites"]}
    home = np.asarray(ep["home"]["xy"])
    if "by_distance" in types:
        d = sorted(np.linalg.norm(v - home) for v in S.values())
        if min(np.diff(d)) < 250:
            return False
    if "closest_pair" in types:
        pr = sorted(np.linalg.norm(a - b) for a, b in itertools.combinations(S.values(), 2))
        if not pr[0] < 0.7 * pr[1]:
            return False
    if "geo_first" in types:
        y = sorted((v[1] for v in S.values()), reverse=True)
        if y[0] - y[1] < 300:
            return False
    if "alt_by_camera" in types:
        modes = {s["mode"] for s in ep["sites"]}
        if modes != {"rgb", "multispectral"}:
            return False
    return True


def sample_spec(rng, t, names):
    if t == "precedence":
        a, b = rng.choice(names, 2, replace=False)
        return {"type": t, "a": str(a), "b": str(b)}
    if t == "altitude":
        return {"type": t, "site": str(rng.choice(names)), "meters": int(rng.choice([40, 50, 70, 80, 90, 100]))}
    if t in ("first", "last", "revisit"):
        return {"type": t, "site": str(rng.choice(names))}
    if t == "alt_by_camera":
        a, b = rng.choice([40, 50, 70, 80, 90, 100], 2, replace=False)
        return {"type": t, "rgb": int(a), "multi": int(b)}
    return {"type": t}


CONFLICT = [{"altitude", "alt_by_camera"}, {"first", "geo_first"}, {"first", "by_distance"}, {"geo_first", "by_distance"},
            {"first", "last"}, {"last", "revisit"}, {"first", "revisit"}, {"geo_first", "revisit"}, {"by_distance", "revisit"},
            {"by_distance", "closest_pair"}, {"closest_pair", "geo_first"}]


def compatible(types):
    return not any(set(types) == c for c in CONFLICT)


KW = {"altitude": r"altitude|above ground|height|high", "alt_by_camera": r"altitude|above ground|height|high",
      "geo_first": r"north", "by_distance": r"distance|nearest|closest|outward", "closest_pair": r"closest|nearest|neighbour",
      "revisit": r"twice|second|again|once more|two", "first": r"first|begin|start", "last": r"last|final|end",
      "precedence": r"before|until|prior|ahead|precede|first"}


def survives(text, specs, names):
    if not all(re.search(KW[s["type"]], text, re.I) for s in specs):
        return False
    toks = set(re.findall(r"\bS\d\b", " ".join(SP.render(s) for s in specs))) | set(re.findall(r"\d+", " ".join(SP.render(s) for s in specs)))
    got = set(re.findall(r"\bS\d\b", text)) | set(re.findall(r"\d+", text))
    return toks <= got and got <= toks | {"1", "2"} and 15 <= len(text) <= 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author-model", default=None)
    ap.add_argument("--out", default="data/bench_t3.json")
    ap.add_argument("--seed", type=int, default=303)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    areas = D.make_areas(seed=a.seed + 1, per_class=1, tag="t3")
    llm = LLM(a.author_model, os.path.join(ROOT, "results", "cache_author_t3.jsonl"), temperature=0.7, seed=1, num_predict=200) if a.author_model else None
    singles = [[t] for t in SP.ALL_TYPES for _ in range(12)]
    cc, cu, uu = [], [], []
    while len(cc) < 12:
        t = [str(x) for x in rng.choice(SP.COVERED, 2, replace=False)]
        if compatible(t):
            cc.append(t)
    while len(cu) < 12:
        t = [str(rng.choice(SP.COVERED)), str(rng.choice(SP.UNCOVERED))]
        if compatible(t):
            cu.append(t)
    while len(uu) < 12:
        t = [str(x) for x in rng.choice(SP.UNCOVERED, 2, replace=False)]
        if compatible(t):
            uu.append(t)
    plan = [("single", t) for t in singles] + [("cc", t) for t in cc] + [("cu", t) for t in cu] + [("uu", t) for t in uu]
    order = rng.permutation(len(plan))
    episodes, stats = [], {"llm": 0, "tmpl": 0}
    area_names = list(areas)
    for n, ix in enumerate(order):
        kind, types = plan[int(ix)]
        area = area_names[n % len(area_names)]
        for attempt in range(400):
            k = int(rng.choice([4, 5, 6]))
            m = D.make_mission(rng, areas, area, k)
            ep = D.to_episode(rng, m, f"T{n:04d}", f"spec_{'+'.join(types)}", [], None, "test")
            names = [s["name"] for s in ep["sites"]]
            if not feasible_margins(ep, types):
                continue
            specs = [sample_spec(rng, t, names) for t in types]
            e_best = SP.best_energy(ep, specs)
            if e_best is None:
                continue
            ep["specs"] = specs
            ep["budget_m"] = float(1.4 * e_best)
            break
        else:
            raise RuntimeError("no feasible mission for " + str(types))
        fam = n % 2
        text = SP.render_all(specs, fam)
        author = "template%d" % fam
        if llm is not None and n % 2 == 0:
            tmpl = SP.render_all(specs, 0)
            for tries in range(3):
                try:
                    r = llm.chat("You edit operations manuals for a drone inspection company. Reply with JSON {\"text\": \"...\"}.",
                                 "Rewrite the following operator requirement in your own words, in one or two plain sentences. "
                                 "Keep every site name (S1, S2, ...) and every number exactly as written; do not add new requirements.\n"
                                 f"Requirement: {tmpl}" + (f"\n(variant {tries})" if tries else ""), tag=f"{ep['id']}:author:{tries}")
                    d = parse_json(r["content"])
                    t2 = d.get("text") if isinstance(d, dict) else None
                    if isinstance(t2, str) and survives(t2, specs, names):
                        text, author = t2, "llm"
                        break
                except Exception:
                    pass
        stats["llm" if author == "llm" else "tmpl"] += 1
        ep["spec_text"], ep["spec_author"], ep["spec_kind"] = text, author, kind
        ep["noise_scale"] = CLS_SCALE[area.split("_")[0]]
        episodes.append(ep)
    out = {"areas": {k: {"cls": v["cls"], "C": v["C"].tolist(), "offset": v["offset"].tolist()} for k, v in areas.items()},
           "episodes": episodes, "stats": stats}
    json.dump(out, open(os.path.join(ROOT, a.out), "w"))
    scales = {e["id"]: {"scale": e["noise_scale"]} for e in episodes}
    json.dump({"scales": scales}, open(os.path.join(ROOT, "data", "scales_t3.json"), "w"))
    print(len(episodes), "episodes", stats)


if __name__ == "__main__":
    main()
