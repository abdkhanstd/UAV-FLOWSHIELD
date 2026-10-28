"""Frontier of the disclosure decision: how much of the planning can be sent to the cloud, at what noise, at what planning cost.

For each declared threat set and cap on the noise, a mission is released to the planner when every declared threat is certified at a scale
within the cap (the released scale is the largest certified scale of the set) and is planned on the edge otherwise. The planning cost of a released
mission is read from the regret curve of its registry class (results/regret_t2.json, an ideal planner on the noisy view; tour excess over the
optimum and share of suboptimal tours). Edge planning has no cloud exposure and no regret.
usage: python scripts/fig_frontier.py [results/privacy5_C_bonf.json] [tag]  -> ../paper/figs/frontier_<tag>.pdf, ../paper/auto/table_frontier_<tag>.tex, numbers_frontier_<tag>.tex"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "..", "paper", "figs")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
SETS = [("blind", ["blind"], "#1f77b4", "Blind"), ("blind, 4 releases", ["blind", "x4"], "#2ca02c", "Four"),
        ("+ portfolio of 100", ["blind", "x4", "pf100"], "#ff7f0e", "PfH"), ("+ portfolio of 25", ["blind", "x4", "pf100", "pf25"], "#d62728", "PfQ")]
CAPS = [100, 200, 300, 400, 600, 800]
NC = "\\newcommand"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results/privacy5_C.json"
    tag = sys.argv[2] if len(sys.argv) > 2 else "a"
    M = json.load(open(os.path.join(ROOT, src)))["missions"]
    R = json.load(open(os.path.join(ROOT, "results", "regret_t2.json")))
    curve = {}
    for cls in ("dense", "medium", "sparse"):
        ms = [x for x in R["missions"] if x["area"].startswith(cls)]
        curve[cls] = {int(s): (np.mean([x["curve"][s]["regret"] for x in ms]), np.mean([x["curve"][s]["subopt"] for x in ms])) for s in ms[0]["curve"]}
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
    L = ["\\begin{tabular}{llcccc}", "\\toprule", "Declared threats & Cap (m) & Planned in the cloud (\\%) & Mean released scale (m) & Excess tour energy (\\%) & Suboptimal tours (\\%) \\\\", "\\midrule"]
    m = []
    for name, ths, col, code in SETS:
        share, xs, ys = [], [], []
        for cap in CAPS:
            rel = [all(x["cert"][t]["certified"] for t in ths) and max(x["cert"][t]["scale"] for t in ths) <= cap for x in M]
            sc = [max(x["cert"][t]["scale"] for t in ths) for x, r in zip(M, rel) if r]
            reg = [curve[x["area"].split("_")[0]][max(x["cert"][t]["scale"] for t in ths)] for x, r in zip(M, rel) if r]
            sh = 100 * np.mean(rel)
            share.append(sh)
            e = 100 * np.mean([r[0] for r in reg]) if reg else float("nan")
            s_ = 100 * np.mean([r[1] for r in reg]) if reg else float("nan")
            xs.append(sh)
            ys.append(e)
            if cap in (200, 400, 800):
                dash = lambda v, f: f.format(v) if reg else "--"
                L.append(f"{name if cap == 200 else ''} & {cap} & {sh:.0f} & {dash(np.mean(sc) if sc else 0, '{:.0f}')} & {dash(e, '{:.2f}')} & {dash(s_, '{:.1f}')} \\\\")
                m += [f"{NC}{{\\fr{code}Cap{ {200: 'Two', 400: 'Four', 800: 'Eight'}[cap] }Share}}{{{sh:.0f}}}", f"{NC}{{\\fr{code}Cap{ {200: 'Two', 400: 'Four', 800: 'Eight'}[cap] }Reg}}{{{e:.2f}}}"]
        ax[0].plot(CAPS, share, "o-", color=col, ms=3.5, lw=1.2, label=name)
        ax[1].plot(xs, ys, "o-", color=col, ms=3.5, lw=1.2)
        L.append("\\midrule") if name != SETS[-1][0] else None
    ax[0].set_xlabel("cap on the noise scale (m)")
    ax[0].set_ylabel("missions planned in the cloud (%)")
    ax[0].set_ylim(-3, 103)
    ax[1].set_xlabel("missions planned in the cloud (%)")
    ax[1].set_ylabel("excess tour energy of released missions (%)")
    ax[0].legend(fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.52))
    ax[0].set_title("(a) availability of the cloud planner", fontsize=8)
    ax[1].set_title("(b) availability against planning cost", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"frontier_{tag}.pdf"))
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, f"table_frontier_{tag}.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, f"numbers_frontier_{tag}.tex"), "w").write("\n".join(m) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
