"""Table and macros for the optimization-driven attacker (exp_optattack.py).
usage: python scripts/analyze_opt.py results/eval/optattack_<model>.jsonl <tag>"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from stats_util import wilson
from eval_common import LABEL

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
GOALS = ["divert", "land", "skip", "altitude", "exfil", "inenv"]
GNAME = {"divert": "divert", "land": "land", "skip": "skip a site", "altitude": "altitude", "exfil": "exfiltrate", "inenv": "in-envelope loiter"}
METHODS = ["undefended", "prv", "eamsr", "mcp_pep", "fs_chan", "flowshield"]


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, sys.argv[1]))]
    tag = sys.argv[2]
    T = tag[0].upper() + tag[1:]
    macros = []
    L = ["\\begin{tabular}{l" + "c" * (len(GOALS) + 1) + "}", "\\toprule", "Target & " + " & ".join(GNAME[g] for g in GOALS) + " & Any goal \\\\", "\\midrule"]
    for m in METHODS:
        r = [x for x in rows if x["method"] == m]
        if not r:
            continue
        cells = []
        for g in GOALS:
            rg = [x for x in r if x["goal"] == g]
            k = sum(x["success_round"] is not None for x in rg)
            cells.append(f"{k}/{len(rg)}" if rg else "--")
        k = sum(x["success_round"] is not None for x in r)
        p, lo, hi = wilson(k, len(r))
        cells.append(f"{k}/{len(r)} ({100 * p:.0f}\\%)")
        L.append(f"{LABEL.get(m, m)} & " + " & ".join(cells) + " \\\\")
        c = "".join(w.capitalize() for w in {"fs_chan": "FsChan", "flowshield": "Fs"}.get(m, m.replace("_", " ")).split())
        macros.append(f"\\newcommand{{\\opt{c}{T}}}{{{100 * p:.1f}}}")
        macros.append(f"\\newcommand{{\\opt{c}{T}Hi}}{{{100 * hi:.1f}}}")
        macros.append(f"\\newcommand{{\\opt{c}{T}K}}{{{k}}}")
        macros.append(f"\\newcommand{{\\opt{c}{T}N}}{{{len(r)}}}")
        att = [x for x in r if x.get("attempted_any")]
        macros.append(f"\\newcommand{{\\opt{c}{T}Attempt}}{{{100 * len(att) / len(r):.1f}}}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, f"table_opt_{tag}.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, f"numbers_opt_{tag}.tex"), "w").write("\n".join(macros) + "\n")
    for m in METHODS:
        r = [x for x in rows if x["method"] == m]
        if r:
            print(m, sum(x["success_round"] is not None for x in r), "/", len(r), " rounds to success:", sorted(x["success_round"] for x in r if x["success_round"]))


if __name__ == "__main__":
    main()
