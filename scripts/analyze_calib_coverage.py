"""Tables, macros and figure for the coverage study of the calibration rules (exp_calib_coverage.py).
usage: python scripts/analyze_calib_coverage.py results/calib_cov_t1.json <tag>     -> ../paper/auto/{numbers,table}_covcal_<tag>.tex, figs/calib_coverage_<tag>.pdf"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
FIG = os.path.join(ROOT, "..", "paper", "figs")
LABEL = {
    "mean10": ("mean of 10 draws", "10 draws, mean $\\le \\rho$; first passing scale"),
    "wilson30": ("Wilson limit, fractional mean (previous rule)", "30 draws, Wilson limit $\\le \\rho$; first passing scale"),
    "cp_asc30": ("exact test, ascending scan", "30 draws, exact test at 5 %; first passing scale, no correction"),
    "cp_bonf60": ("exact test, Bonferroni", "60 draws, exact test at 5 %/$|G|$; first passing scale"),
    "cp_fs30": ("certified, fixed sequence, T=30", "30 draws, exact test at 5 %; fixed sequence from the largest scale"),
    "cp_fs60": ("certified, fixed sequence, T=60", "60 draws"),
    "cp_fs100": ("certified, fixed sequence, T=100", "100 draws"),
    "cp_fs30m": ("certified, mission-level, T=30", "30 draws, any site identified"),
    "cp_fs200": ("certified, fixed sequence, T=200", "200 draws"),
    "cp_fs200_05": ("certified, $\\rho$=5\\,\\%, T=200", "200 draws, target 5 %"),
}
CODE = {"mean10": "MeanTen", "wilson30": "WilsonThirty", "cp_asc30": "CpAscThirty", "cp_bonf60": "CpBonfSixty", "cp_fs30": "CpFsThirty",
        "cp_fs60": "CpFsSixty", "cp_fs100": "CpFsHundred", "cp_fs200": "CpFsTwoHundred", "cp_fs30m": "CpFsThirtyM", "cp_fs200_05": "CpFsTwoHundredFive"}


def main(path, tag):
    R = json.load(open(os.path.join(ROOT, path)))
    M = R["missions"]
    T = tag[0].upper() + tag[1:]
    macros = []
    rows = []
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for name in R["configs"]:
        rho = 0.05 if name.endswith("_05") else 0.10
        vm = {}
        for cls in ("dense", "medium", "sparse", "all"):
            m = [x for x in M if cls == "all" or x["area"].startswith(cls)]
            v = np.array([x["rules"][name]["viol"] for x in m])
            vm[cls] = (100 * v.mean(), 100 * v.max(), int((v > 0.05).sum()), len(v),
                       float(np.mean([x["rules"][name]["scale"] for x in m])), 100 * float(np.mean([x["rules"][name]["unc"] for x in m])),
                       float(np.mean([x["rules"][name]["draws"] for x in m])))
        for cls in ("dense", "medium", "sparse", "all"):
            macros += [f"\\newcommand{{\\cov{CODE[name]}Viol{cls.capitalize()}{T}}}{{{vm[cls][0]:.1f}}}",
                       f"\\newcommand{{\\cov{CODE[name]}Scale{cls.capitalize()}{T}}}{{{vm[cls][4]:.0f}}}"]
        macros += [f"\\newcommand{{\\cov{CODE[name]}Max{T}}}{{{vm['all'][1]:.1f}}}", f"\\newcommand{{\\cov{CODE[name]}Over{T}}}{{{vm['all'][2]}}}",
                   f"\\newcommand{{\\cov{CODE[name]}Unc{T}}}{{{vm['all'][5]:.1f}}}", f"\\newcommand{{\\cov{CODE[name]}Draws{T}}}{{{vm['all'][6]:.0f}}}",
                   f"\\newcommand{{\\cov{CODE[name]}N{T}}}{{{vm['all'][3]}}}"]
        rows.append((name, rho, vm))
        ax.scatter(vm["all"][4], vm["all"][0], s=28, label=LABEL[name][0], marker="o" if name.startswith("cp_fs") else "s")
    ax.axhline(5, color="#999", lw=0.8, ls="--")
    ax.text(ax.get_xlim()[1] * 0.98, 5.3, "5 %", ha="right", fontsize=6)
    ax.set_xlabel("mean released scale (m)")
    ax.set_ylabel("missions with true risk above $\\rho$ (%)")
    ax.legend(fontsize=5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"calib_coverage_{tag}.pdf"))
    plt.close(fig)
    L = ["\\begin{tabular}{lcccccc}", "\\toprule",
         "Rule & \\multicolumn{3}{c}{Claim at a scale with true risk above $\\rho$ (\\%)} & Mean scale (m) & Without a claim (\\%) & Attacks used \\\\",
         " & dense / medium / sparse & all & max over missions &  &  &  \\\\", "\\midrule"]
    for name, rho, vm in rows:
        a = vm["all"]
        L.append(f"{LABEL[name][0]} & {vm['dense'][0]:.1f} / {vm['medium'][0]:.1f} / {vm['sparse'][0]:.1f} & {a[0]:.1f} & {a[1]:.0f} & {a[4]:.0f} & "
                 f"{a[5]:.0f} & {a[6]:.0f} \\\\")
        if name in ("mean10", "cp_bonf60"):
            L.append("\\midrule")
    orc = {cls: float(np.mean([x["oracle"] for x in M if cls == "all" or x["area"].startswith(cls)])) for cls in ("dense", "medium", "sparse", "all")}
    for cls in orc:
        macros.append(f"\\newcommand{{\\covOracleScale{cls.capitalize()}{T}}}{{{orc[cls]:.0f}}}")
    macros.append(f"\\newcommand{{\\covOracleNone{T}}}{{{100 * np.mean([not x['oracle_ok'] for x in M]):.1f}}}")
    L.append("\\midrule")
    L.append(f"reference: smallest scale whose true risk is at most $\\rho$ (needs the true risk) & -- & -- & -- & {orc['all']:.0f} & "
             f"{100 * np.mean([not x['oracle_ok'] for x in M]):.0f} & -- \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, f"table_covcal_{tag}.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, f"numbers_covcal_{tag}.tex"), "w").write("\n".join(macros) + "\n")
    for name, rho, vm in rows:
        print(f"{name:12s} viol all {vm['all'][0]:5.1f}% max {vm['all'][1]:5.1f}% n>5% {vm['all'][2]}/{vm['all'][3]} scale {vm['all'][4]:.0f} unc {vm['all'][5]:.0f}% draws {vm['all'][6]:.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
