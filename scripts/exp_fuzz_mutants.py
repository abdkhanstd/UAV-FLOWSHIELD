"""Power check for the fuzz test: each safeguard is disabled in turn (a mutant) and the same fuzz
procedure must then report violations. A fuzz test that cannot detect a broken safeguard would
say nothing about the intact one."""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import exp_fuzz as F
import flowshield.shield as SH
from flowshield import auth as AUTH

ROOT = F.ROOT
N = 4000


def euclid_distance(ep, xy, metric="route"):
    d = [float(np.linalg.norm(np.asarray(xy) - np.asarray(s["xy"]))) for s in ep["sites"]]
    j = int(np.argmin(d))
    return j, d[j]


def floors_last_wins(obs):
    return {o["site_name"]: o["value"] for o in obs if o["kind"] == "min_altitude"}


def run(label, patches=(), allow_points=False):
    old = []
    for obj, name, val in patches:
        old.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)
    rng = np.random.default_rng(5)
    EP, ST = F.pool(rng)
    stats, viol = Counter(), []
    for t in range(N):
        F.one_trial(rng, EP, ST, stats, viol, t, allow_points=allow_points)
    for obj, name, val in old:
        setattr(obj, name, val)
    detail = Counter(v[1] for v in viol)
    print(f"{label:38s} accepted {stats['accepted']:5d}  violations {len(viol):5d}  {dict(detail)}", flush=True)
    return {"label": label, "accepted": stats["accepted"], "violations": len(viol), "detail": dict(detail)}


if __name__ == "__main__":
    auth_all = dict(SH.AUTHORITY)
    out = [run("intact")]
    out.append(run("no spatial envelope", [(SH, "ENVELOPE_R", 1e9)]))
    out.append(run("straight-line envelope", [(SH, "envelope_distance", euclid_distance)]))
    out.append(run("no origin check", [(AUTH, "verify", lambda ep, doc: True)]))
    out.append(run("no loiter cap", [(SH, "MAX_LOITER", 1e9)]))
    out.append(run("no altitude bound", [(SH, "ALT_MAX", 1e4), (SH, "ALT_MIN", -1e4)]))
    out.append(run("no number redaction", [(SH, "redact_numbers", lambda t: t)]))
    out.append(run("coordinate targets allowed", [], allow_points=True))
    out.append(run("any channel may add spatial task", [(SH, "AUTH_SPATIAL_CHANNELS", {"portal", "feed", "peer"}),
                                                         (SH, "AUTHORITY", {**auth_all, "inspection_request": {"portal", "feed", "peer"}})]))
    out.append(run("peer may impose constraints", [(SH, "AUTHORITY", {**auth_all, "min_altitude": {"feed", "portal", "peer"},
                                                                       "crop_stress": {"portal", "peer"}})]))
    out.append(run("conflicting floors: last wins", [(SH, "floors_of", floors_last_wins)]))
    json.dump(out, open(os.path.join(ROOT, "results", "fuzz_mutants.json"), "w"), indent=1)
