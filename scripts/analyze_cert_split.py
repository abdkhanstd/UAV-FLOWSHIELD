"""Held-out identification on T2 split into certified and uncertified missions (results/privacy3_cert100_t2.json), and the paired
AD8 check of the final configuration (results/eval/t2_final_ad8_paired.json).
usage: python scripts/analyze_cert_split.py -> ../paper/auto/numbers_certsplit.tex"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
NC = "\\newcommand"


def main():
    d = json.load(open(os.path.join(ROOT, "results", "privacy3_cert100_t2.json")))
    cal = {c["id"]: c for c in d["calibration"]}
    m = []
    for cls, code in (("dense", "Dense"), ("medium", "Medium"), ("sparse", "Sparse"), ("all", "All")):
        h = [x for x in d["heldout"] if cls == "all" or x["area"].startswith(cls)]
        for name, sel in (("Cert", True), ("Unc", False)):
            s = [x for x in h if cal[x["id"]]["certified"] == sel]
            v = f"{100 * np.mean([x['pol']['cert100']['site'] for x in s]):.1f}" if s else "--"
            m += [f"{NC}{{\\split{name}Site{code}}}{{{v}}}", f"{NC}{{\\split{name}N{code}}}{{{len(s)}}}"]
    p = os.path.join(ROOT, "results", "eval", "t2_final_ad8_paired.json")
    if os.path.exists(p):
        rows = json.load(open(p))
        fin = [json.loads(l) for l in open(os.path.join(ROOT, "results", "eval", "t2_gemma_final.jsonl"))]
        lost = {(r["id"], r["method"]) for r in fin if r["goal"] == "AD8" and r["attack_success"]}
        same = sum(1 for r in rows if (r["id"], r["method"]) in lost and not r["success"])
        m += [f"{NC}{{\\pairLost}}{{{len(lost)}}}", f"{NC}{{\\pairSame}}{{{same}}}"]
    open(os.path.join(AUTO, "numbers_certsplit.tex"), "w").write("\n".join(m) + "\n")
    print("\n".join(m))


if __name__ == "__main__":
    main()
