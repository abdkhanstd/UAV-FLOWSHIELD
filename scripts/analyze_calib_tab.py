"""Table-first certification versus the full ladder (results/calib_cov_t1_tab.json, same pools as the coverage study).
usage: python scripts/analyze_calib_tab.py   -> ../paper/auto/table_calib_tab.tex, numbers_calibtab.tex"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
NC = "\\newcommand"
ROWS = [("cp_fs60", "Ladder", 60, "FsSixty"), ("cp_fs100", "Ladder", 100, "FsHundred"), ("cp_fs200", "Ladder", 200, "FsTwoHundred"),
        ("cp_tab60", "Table first", 60, "TabSixty"), ("cp_tab100", "Table first", 100, "TabHundred"), ("cp_tab200", "Table first", 200, "TabTwoHundred")]


def main():
    d = json.load(open(os.path.join(ROOT, "results", "calib_cov_t1_tab.json")))
    L = ["\\begin{tabular}{lrcccccc}", "\\toprule",
         "Rule & $T$ & Violation, mean (max) & Mean scale (m) & Attacks & At table scale (\\%) & No claim (\\%) \\\\", "\\midrule"]
    m = []
    for name, lab, T, code in ROWS:
        ms = d["missions"]
        v = np.array([x["rules"][name]["viol"] for x in ms]) * 100
        sc = np.mean([x["rules"][name]["scale"] for x in ms])
        dr = np.mean([x["rules"][name]["draws"] for x in ms])
        un = 100 * np.mean([x["rules"][name]["unc"] for x in ms])
        ts = [x["rules"][name]["table_share"] for x in ms if x["rules"][name]["table_share"] is not None]
        tab = f"{100 * np.mean(ts):.0f}" if ts else "--"
        L.append(f"{lab} & {T} & {v.mean():.2f}\\,({v.max():.1f})\\,\\% & {sc:.0f} & {dr:.0f} & {tab} & {un:.1f} \\\\")
        m += [f"{NC}{{\\ct{code}Viol}}{{{v.mean():.2f}}}", f"{NC}{{\\ct{code}Max}}{{{v.max():.1f}}}", f"{NC}{{\\ct{code}Scale}}{{{sc:.0f}}}",
              f"{NC}{{\\ct{code}Draws}}{{{dr:.0f}}}", f"{NC}{{\\ct{code}Unc}}{{{un:.1f}}}"]
        if ts:
            m.append(f"{NC}{{\\ct{code}Share}}{{{100 * np.mean(ts):.0f}}}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_calib_tab.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, "numbers_calibtab.tex"), "w").write("\n".join(m) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
