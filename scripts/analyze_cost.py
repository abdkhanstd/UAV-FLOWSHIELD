"""System-cost table and figure from results/cost_<model>.json (written by exp_cost.py)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_main import LABEL

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
FIG = os.path.join(ROOT, "..", "paper", "figs")
METHODS = ["undefended", "prompt_only", "spotlight_guard", "prv", "eamsr", "mcp_pep", "flowshield"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
macros = []
CODE = {"undefended": "Undef", "prompt_only": "Prompt", "spotlight_guard": "Spot", "prv": "Prv", "eamsr": "Eamsr", "mcp_pep": "Pep",
        "flowshield": "Fs"}


def mac(n, v):
    macros.append(f"\\newcommand{{\\{n}}}{{{v}}}")


def main(model="gemma4:12b"):
    p = os.path.join(ROOT, "results", f"cost_{model.replace(':', '-')}.json")
    if not os.path.exists(p):
        print("no cost file")
        return
    D = json.load(open(p))
    rows = D["rows"]
    L = ["\\begin{tabular}{lccccccc}", "\\toprule",
         "Method & p50 (s) & p95 (s) & Cloud calls & Edge calls & Up (kB) & Down (kB) & Edge time (ms) \\\\", "\\midrule"]
    stat = {}
    for m in METHODS:
        x = [r for r in rows if r["method"] == m]
        e2e = np.array([r["e2e"] for r in x])
        s = {"p50": np.percentile(e2e, 50), "p95": np.percentile(e2e, 95), "cc": np.mean([r["cloud_calls"] for r in x]),
             "ec": np.mean([r["edge_calls"] for r in x]), "up": np.mean([r["cloud_up"] for r in x]) / 1000,
             "down": np.mean([r["cloud_down"] for r in x]) / 1000, "tr": np.mean([r["trusted_s"] for r in x]) * 1000}
        stat[m] = s
        L.append(f"{LABEL[m]} & {s['p50']:.1f} & {s['p95']:.1f} & {s['cc']:.2f} & {s['ec']:.2f} & {s['up']:.2f} & {s['down']:.2f} & {s['tr']:.1f} \\\\")
        for k, v in s.items():
            name = {"p50": "Med", "p95": "Tail"}.get(k, k.capitalize())   # macro names must not contain digits
            mac(f"cost{name}{CODE[m]}", f"{v:.1f}" if k in ("p50", "p95", "tr") else f"{v:.2f}")
        if m == "mcp_pep":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_cost_seq.tex"), "w").write("\n".join(L) + "\n")
    cal = D["calibration"]
    for c in ("dense", "medium", "sparse"):
        v = [x["seconds"] for x in cal if x["cls"] == c]
        mac(f"calMed{c.capitalize()}", f"{np.median(v):.1f}")
        mac(f"calTail{c.capitalize()}", f"{np.percentile(v, 95):.1f}")
    mac("peakRss", f"{D['peak_rss_mb']:.0f}")
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    x = np.arange(len(METHODS))
    ax.bar(x - 0.18, [stat[m]["p50"] for m in METHODS], 0.34, color="#1f5fa8", label="p50")
    ax.bar(x + 0.18, [stat[m]["p95"] for m in METHODS], 0.34, color="#9ecae1", label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m].replace("Prompt-RAG-Validator", "PRV").replace("UAV-", "") for m in METHODS], fontsize=6.5, rotation=15)
    ax.set_ylabel("End-to-end latency per mission (s)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cost.pdf"))
    plt.close(fig)
    open(os.path.join(AUTO, "numbers_cost.tex"), "w").write("\n".join(macros) + "\n")
    print("cost analysis written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gemma4:12b")
