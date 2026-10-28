"""Write the per-episode noise-scale file used by run_llm.py from a calibration run.
usage: python scripts/make_scales.py results/privacy3.json ucb10 data/scales_ucb.json"""
import json
import sys

src, policy, dst = sys.argv[1:4]
D = json.load(open(src))
scales = {c["id"]: {"scale": c[policy]["scale"], "est": c[policy].get("est"), "ucb": c[policy].get("ucb")} for c in D["calibration"]}
json.dump({"policy": policy, "source": src, "grid": D["grid"], "scales": scales}, open(dst, "w"))
import numpy as np
print(len(scales), "episodes; mean scale", round(float(np.mean([v["scale"] for v in scales.values()])), 1))
