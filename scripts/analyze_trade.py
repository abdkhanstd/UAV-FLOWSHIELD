"""LLM-level privacy-utility check on T2: mission completion of FlowShield at fixed noise scales and with the calibrated
scale, overall and on the tight-budget missions, with paired tests against the zero-noise run.
usage: python scripts/analyze_trade.py   (reads results/eval/t2_gemma_fixed.jsonl and t2_gemma_fs.jsonl)"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from eval_common import load, rate, fmt
from stats_util import mcnemar_exact

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "eval")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
macros = []


def mac(n, v):
    macros.append(f"\\newcommand{{\\{n}}}{{{v}}}")


def main():
    fx = load([os.path.join(RES, "t2_gemma_fixed.jsonl"), os.path.join(RES, "t2_gemma_fs.jsonl")])
    ben = fx[fx.category.str.startswith("benign")]
    names = [("fs_s0", "0 m", "SZero"), ("fs_s100", "100 m", "SOneZeroZero"), ("fs_s300", "300 m", "SThreeZeroZero"), ("fs_s800", "800 m", "SEightZeroZero"),
             ("flowshield", "calibrated", "Cal"), ("fs_no_view", "absolute data", "Abs")]
    cats = [("all", None), ("benign_tight", "Tight"), ("benign_dynamic", "Dynamic"), ("benign_multi", "Several documents")]
    L = ["\\begin{tabular}{l" + "c" * len(cats) + "c}", "\\toprule", "Planning view & All benign (\\%) & Tight budget (\\%) & Dynamic (\\%) & Several documents (\\%) & Repair rounds per mission \\\\", "\\midrule"]
    piv = ben.pivot_table(index="id", columns="method", values="success", aggfunc="first")
    for m, lab, code in names:
        d = ben[ben.method == m]
        if not len(d):
            continue
        cells = []
        for c, _ in cats:
            x = d if c == "all" else d[d.category == c]
            cells.append(fmt(rate(x.success), ci=False))
            mac(f"tr{code}{c.replace('benign_', '').capitalize()}", f"{100 * rate(x.success)[0]:.1f}")
        L.append(f"{lab} & " + " & ".join(cells) + f" & {d.repaired.mean():.2f} \\\\")
        mac(f"tr{code}Repair", f"{d.repaired.mean():.2f}")
        if m != "fs_s0" and "fs_s0" in piv and m in piv:
            ok = piv[["fs_s0", m]].dropna()
            n01, n10, p = mcnemar_exact(ok["fs_s0"].astype(bool), ok[m].astype(bool))
            mac(f"tr{code}PairA", n10)
            mac(f"tr{code}PairB", n01)
            mac(f"tr{code}PairP", f"{p:.2f}")
        if m == "fs_s300":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_trade_tb.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, "numbers_trade_tb.tex"), "w").write("\n".join(macros) + "\n")
    print(open(os.path.join(AUTO, "table_trade_tb.tex")).read())


if __name__ == "__main__":
    main()
