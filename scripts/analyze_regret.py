"""Planning-regret table and macros (T1 and T2 geometry). usage: python scripts/analyze_regret.py"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
SC = [0, 100, 150, 300, 600, 800]
NAME = {0: "Zero", 100: "OneZeroZero", 150: "OneFiveZero", 300: "ThreeZeroZero", 600: "SixZeroZero", 800: "EightZeroZero"}
macros = []


def mac(n, v):
    macros.append(f"\\newcommand{{\\{n}}}{{{v}}}")


def main():
    L = ["\\begin{tabular}{lcccccc}", "\\toprule", "Scale (m) & \\multicolumn{2}{c}{Energy regret (\\%)} & \\multicolumn{2}{c}{Suboptimal tour (\\%)} & \\multicolumn{2}{c}{Tight: believes full fits (\\%)} \\\\",
         " & T1 & T2 & T1 & T2 & T1 & T2 \\\\", "\\midrule"]
    data = {}
    for tag, f in (("Ta", "regret.json"), ("Tb", "regret_t2.json")):
        R = json.load(open(os.path.join(ROOT, "results", f)))["missions"]
        data[tag] = R
    for s in SC:
        cells = []
        base = {}
        for tag in ("Ta", "Tb"):
            R = data[tag]
            use = [m for m in R if m["category"] in ("benign_plain", "benign_tight", "attack", "adaptive", "benign_constraint", "benign_dynamic", "benign_multi")]
            reg = np.mean([m["curve"][str(s)]["regret"] for m in use]) * 100
            sub = np.mean([m["curve"][str(s)]["subopt"] for m in use]) * 100
            sub0 = np.mean([m["curve"]["0"]["subopt"] for m in use]) * 100
            tight = [m for m in R if m["category"] == "benign_tight"]
            bel = np.mean([m["curve"][str(s)]["believes_fits"] for m in tight]) * 100
            base[tag] = (reg, sub, bel, sub - sub0)
            for k, v in (("Regret", reg), ("Subopt", sub), ("Fits", bel), ("SuboptNet", sub - sub0)):
                mac(f"reg{k}{NAME[s]}{tag}", f"{v:.2f}" if k == "Regret" else f"{v:.1f}")
        cells = [f"{base['Ta'][0]:.2f}", f"{base['Tb'][0]:.2f}", f"{base['Ta'][1]:.1f}", f"{base['Tb'][1]:.1f}", f"{base['Ta'][2]:.1f}", f"{base['Tb'][2]:.1f}"]
        L.append(f"{s} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_regret.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, "numbers_regret.tex"), "w").write("\n".join(macros) + "\n")
    print(open(os.path.join(AUTO, "table_regret.tex")).read())
    print({m: len([x for x in data[m]]) for m in data})


if __name__ == "__main__":
    main()
