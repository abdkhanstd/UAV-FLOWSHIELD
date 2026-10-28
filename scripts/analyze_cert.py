"""Macros for the certified calibration (exp_privacy3d.py) and the T2 scale file for the pipeline runs.
usage: python scripts/analyze_cert.py   -> ../paper/auto/numbers_cert.tex, data/scales_t2_cert.json"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
macros = []


def mac(n, v):
    macros.append(f"\\newcommand{{\\{n}}}{{{v}}}")


for tag, f in (("Ta", "privacy3_cert100_t1.json"), ("Tb", "privacy3_cert100_t2.json")):
    p = os.path.join(ROOT, "results", f)
    if not os.path.exists(p):
        continue
    C = json.load(open(p))["calibration"]
    mac(f"certUnc{tag}", f"{100 * np.mean([not c['certified'] for c in C]):.1f}")
    mac(f"certDraws{tag}", f"{np.mean([c['draws'] for c in C]):.0f}")
    mac(f"certN{tag}", len(C))
    for cls in ("dense", "medium", "sparse"):
        h = [c for c in C if c["area"].startswith(cls)]
        mac(f"certUnc{cls.capitalize()}{tag}", f"{100 * np.mean([not c['certified'] for c in h]):.1f}")
        mac(f"certScale{cls.capitalize()}{tag}", f"{np.mean([c['scale'] for c in h]):.0f}")
    if tag == "Tb":
        json.dump({"scales": {c["id"]: {"scale": c["scale"], "certified": c["certified"]} for c in C}}, open(os.path.join(ROOT, "data", "scales_t2_cert.json"), "w"))
open(os.path.join(AUTO, "numbers_cert.tex"), "w").write("\n".join(macros) + "\n")
print(len(macros), "macros")
