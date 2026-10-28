"""Coverage study of the calibration rules on pools of simulated attack outcomes (exp_calib_pool.py).

For every mission the pool defines a population of attack outcomes per scale, hence the true per-site risk
mu(s) (mean identification rate over the pool). Each rule is run R times, drawing its calibration samples from the
pool with replacement, and the selected scale is compared with the truth:
  violation = the released scale has mu(s*) > rho
Reported per rule: violation probability (per mission and averaged), mean released scale, share of missions for which
the rule certifies nothing (uncertified), and the number of simulated attacks used.

usage: python scripts/exp_calib_coverage.py results/calib_pool_t1.jsonl results/calib_cov_t1.json [R]
"""
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield.certify import select_scale, cp_max_hits

ROOT = os.path.join(os.path.dirname(__file__), "..")
RHO = 0.10
CONFIGS = [  # name, rule, T, rho, mission_level
    ("mean10", "mean", 10, 0.10, False),
    ("wilson30", "wilson", 30, 0.10, False),
    ("cp_asc30", "cp_asc", 30, 0.10, False),
    ("cp_bonf60", "cp_bonf", 60, 0.10, False),
    ("cp_fs30", "cp_fs", 30, 0.10, False),
    ("cp_fs60", "cp_fs", 60, 0.10, False),
    ("cp_fs100", "cp_fs", 100, 0.10, False),
    ("cp_fs200", "cp_fs", 200, 0.10, False),
    ("cp_fs30m", "cp_fs", 30, 0.10, True),
    ("cp_fs200_05", "cp_fs", 200, 0.05, False),
    ("cp_tab60", "cp_tab", 60, 0.10, False),
    ("cp_tab100", "cp_tab", 100, 0.10, False),
    ("cp_tab200", "cp_tab", 200, 0.10, False),
]
if os.environ.get("FS_ONLY"):
    CONFIGS = [c for c in CONFIGS if c[0] in os.environ["FS_ONLY"].split(",")]
TABLE = {"dense": 50, "medium": 150, "sparse": 600}        # per-class table scale of the paper (public, mission independent)


def decode(masks, k):
    a = np.array(masks, dtype=np.int64)
    return ((a[:, None] >> np.arange(k)[None, :]) & 1).astype(bool)


def task(args):
    rec, R = args
    k, N = rec["k"], rec["N"]
    pool = {int(s): decode(m, k) for s, m in rec["scales"].items()}
    grid = sorted(pool)
    truth = {s: float(pool[s].mean()) for s in grid}
    truth_m = {s: float(pool[s].any(1).mean()) for s in grid}
    orc = next((s for s in grid if truth[s] <= RHO), grid[-1])
    out = {"id": rec["id"], "area": rec["area"], "k": k, "truth": truth, "truth_mission": truth_m, "rules": {},
           "oracle": orc, "oracle_ok": bool(truth[orc] <= RHO)}
    for name, rule, T, rho, ml in CONFIGS:
        rng = np.random.default_rng(int(hashlib.sha256(f"{rec['id']}|{name}".encode()).hexdigest()[:8], 16))
        viol = unc = 0
        scales, draws, tabs = [], [], []
        for _ in range(R):
            def draw(s, n):
                return pool[s][rng.integers(N, size=n)]
            r = select_scale(draw, grid, rho, rule, rng, T=T, mission_level=ml, start=TABLE[rec["area"].split("_")[0]])
            tr = (truth_m if ml else truth)[r["scale"]]
            viol += bool(r["passed"] and tr > rho)         # a claim of safety at a scale whose true risk is too high
            unc += not r["passed"]                          # no scale passed: no claim (the largest scale is released)
            scales.append(r["scale"])
            draws.append(r["draws"])
            if "table" in r:
                tabs.append(r["table"])
        out["rules"][name] = {"viol": viol / R, "unc": unc / R, "scale": float(np.mean(scales)), "draws": float(np.mean(draws)),
                              "scales": [int(x) for x in np.percentile(scales, [10, 50, 90])],
                              "table_share": float(np.mean(tabs)) if tabs else None}
    return out


if __name__ == "__main__":
    pool_file, out_file = sys.argv[1], sys.argv[2]
    R = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    recs = [json.loads(l) for l in open(os.path.join(ROOT, pool_file))]
    print(len(recs), "missions", flush=True)
    with Pool(int(os.environ.get("FS_POOL", "16"))) as p:
        res = p.map(task, [(r, R) for r in recs], chunksize=2)
    json.dump({"R": R, "rho": RHO, "missions": res, "configs": [c[0] for c in CONFIGS]}, open(os.path.join(ROOT, out_file), "w"))
    for name, *_ in CONFIGS:
        for cls in ("dense", "medium", "sparse", "all"):
            m = [x for x in res if cls == "all" or x["area"].startswith(cls)]
            v = np.array([x["rules"][name]["viol"] for x in m])
            print(f"{name:12s} {cls:7s} viol mean {v.mean():.3f} max {v.max():.3f} n>0.05 {int((v > 0.05).sum())}/{len(v)}  "
                  f"scale {np.mean([x['rules'][name]['scale'] for x in m]):.0f}  unc {np.mean([x['rules'][name]['unc'] for x in m]):.2f}")
