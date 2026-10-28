"""Denser grid of fixed noise scales for the held-out comparison (same protocol and draws as exp_privacy3.py):
fixed 75, 200 and 400 m, so that the privacy-planning curve of the fixed policies is not interpolated over a wide gap.
usage: FS_KD_WORKERS=1 FS_POOL=16 python scripts/exp_privacy3b.py  -> results/privacy3_fixedx.json"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from exp_privacy3 import held_task, bench, ROOT

EXTRA = {"fixed75": 75, "fixed200": 200, "fixed400": 400}

if __name__ == "__main__":
    order = sorted([e for e in bench()["episodes"] if e["split"] == "test"], key=lambda e: (e["area"], e["id"]))
    with Pool(int(os.environ.get("FS_POOL", "16"))) as p:
        res = p.map(held_task, [(e["id"], dict(EXTRA)) for e in order], chunksize=3)
    json.dump({"heldout": res}, open(os.path.join(ROOT, "results", "privacy3_fixedx.json"), "w"))
    print("done", len(res))
