"""Sampled-decoding table for T2 (temperature 0.7, three seeds, stratified subset) next to the deterministic run on the same subset.
usage: python scripts/analyze_samp.py"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_common import load, rate

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "eval")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
METH = [("undefended", "Undefended", "Undef"), ("prv", "Prompt-RAG-Validator", "Prv"), ("mcp_pep_repdel", "MCP-PEP + repair + delegation", "PepRepDel"),
        ("flowshield", "UAV-FlowShield", "Fs")]


def stats(df, m):
    d = df[df.method == m]
    a = d[d.category == "attack"]
    ad = d[(d.category == "adaptive") & (d.goal != "AD4")]
    b = d[d.category.str.startswith("benign")]
    return [100 * a.attack_success.mean(), 100 * ad.attack_success.mean(), 100 * b.success.mean()], (len(a), len(ad), len(b))


def main():
    det = os.path.join(RES, "t2_samp_s0.jsonl")
    seeds = [os.path.join(RES, f"t2_samp_s{s}.jsonl") for s in (1, 2, 3)]
    if not all(os.path.exists(f) for f in [det] + seeds):
        print("sampling runs missing")
        return
    D = load([det])
    S = [load([f]) for f in seeds]
    macros = []
    L = ["\\begin{tabular}{lcccc}", "\\toprule", "Method & Deterministic ASR / adaptive / MCR (\\%) & ASR (\\%) & Adaptive ASR (\\%) & MCR (\\%) \\\\", "\\midrule"]
    for m, lab, code in METH:
        d0, n = stats(D, m)
        v = np.array([stats(s, m)[0] for s in S])
        mu, sd = v.mean(0), v.std(0, ddof=1)
        L.append(f"{lab} & {d0[0]:.1f} / {d0[1]:.1f} / {d0[2]:.1f} & {mu[0]:.1f} $\\pm$ {sd[0]:.1f} & {mu[1]:.1f} $\\pm$ {sd[1]:.1f} & {mu[2]:.1f} $\\pm$ {sd[2]:.1f} \\\\")
        for k, nm in enumerate(("Asr", "Adapt", "Mcr")):
            macros.append(f"\\newcommand{{\\samp{nm}{code}}}{{{mu[k]:.1f}}}")
            macros.append(f"\\newcommand{{\\samp{nm}{code}Sd}}{{{sd[k]:.1f}}}")
            macros.append(f"\\newcommand{{\\sampDet{nm}{code}}}{{{d0[k]:.1f}}}")
    macros.append(f"\\newcommand{{\\sampN}}{{{len(D[D.method == 'flowshield'])}}}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_samp_tb.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, "numbers_samp_tb.tex"), "w").write("\n".join(macros) + "\n")
    print(open(os.path.join(AUTO, "table_samp_tb.tex")).read())


if __name__ == "__main__":
    main()
