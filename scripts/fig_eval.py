"""Figures of the evaluation: security versus completion (matched baselines) and per-goal heatmap.
usage: python scripts/fig_eval.py <tag>   (reads results/eval/summary_<tag>.json written by analyze_eval.py)"""
import json
import os
import sys

import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "..", "paper", "figs")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
SHORT = {"undefended": "Undefended", "prompt_only": "Prompt-only", "spotlight_guard": "Spotlight-Guard", "prv": "PRV", "eamsr": "EAMSR",
         "mcp_pep": "MCP-PEP", "flowshield": "UAV-FlowShield", "det_template": "Basic parser", "det_plus": "Extended parser",
         "fs_chan": "FlowShield, channel-only", "fs_fallback": "FlowShield + fallback"}
FAM = {"undefended": "#7f7f7f", "prompt_only": "#8c564b", "spotlight_guard": "#9467bd", "prv": "#bcbd22", "eamsr": "#17becf", "mcp_pep": "#ff7f0e"}


def load(tag):
    return json.load(open(os.path.join(ROOT, "results", "eval", f"summary_{tag}.json")))["summary"]


def scatter(S, tag):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    handles = []
    for b, col in FAM.items():
        for suffix, mk in (("", "o"), ("_rep", "s"), ("_repdel", "^")):
            m = b + suffix
            if m in S and "asr" in S[m] and "mcr" in S[m]:
                x, y = 100 * S[m]["asr"][0], 100 * S[m]["mcr"][0]
                ax.plot(x, y, mk, color=col, ms=5.5, mfc=col if suffix else "white", mew=1.1)
                if suffix == "":
                    ax.annotate(SHORT[b], (x, y), fontsize=5.8, xytext=(4, -9), textcolor=col) if False else ax.annotate(SHORT[b], (x, y), fontsize=5.8, xytext=(5, -9), textcoords="offset points", color=col)
        handles.append(plt.Line2D([], [], marker="o", color=col, ls="", ms=5, label=SHORT[b]))
    for m, col, mk, off in (("flowshield", "#d62728", "*", (10, 9)), ("det_plus", "#2ca02c", "P", (9, -13)), ("det_template", "#98df8a", "P", None)):
        if m in S and "asr" in S[m] and "mcr" in S[m]:
            x, y = 100 * S[m]["asr"][0], 100 * S[m]["mcr"][0]
            ax.errorbar(x, y, xerr=[[max(0.0, 100 * (S[m]["asr"][0] - S[m]["asr"][1]))], [max(0.0, 100 * (S[m]["asr"][2] - S[m]["asr"][0]))]], fmt=mk, color=col, ms=9, lw=0.8)
            if off:
                ax.annotate(SHORT[m], (x, y), fontsize=6, xytext=off, textcoords="offset points", color=col, arrowprops=dict(arrowstyle="-", color=col, lw=0.5))
    handles += [plt.Line2D([], [], marker="o", mfc="white", color="#555", ls="", ms=5, label="baseline"),
                plt.Line2D([], [], marker="s", color="#555", ls="", ms=5, label="+ repair feedback"),
                plt.Line2D([], [], marker="^", color="#555", ls="", ms=5, label="+ repair + delegation")]
    ax.set_xlabel("Executed attack success, injection suite (%)  [lower is better]")
    ax.set_ylabel("Mission completion (%)  [higher is better]")
    ax.legend(handles=handles, fontsize=5.8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"matched_{tag}.pdf"))
    plt.close(fig)


def heat(S, tag, methods):
    ms = [m for m in methods if m in S]
    goals = ["G1", "G2", "G3", "G4", "G5", "G6"]
    ads = sorted({g for m in ms for g in S[m].get("adapt_goal", {})})
    cols = goals + ads
    M = np.full((len(ms), len(cols)), np.nan)
    for i, m in enumerate(ms):
        for j, c in enumerate(cols):
            d = S[m].get("asr_goal", {}) if c.startswith("G") else S[m].get("adapt_goal", {})
            if c in d:
                M[i, j] = 100 * d[c][0]
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(ms) + 1.0))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    for i in range(len(ms)):
        for j in range(len(cols)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.8, color="#888")
            else:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.2, color="white" if M[i, j] > 55 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6.5)
    ax.set_yticks(range(len(ms)))
    from eval_common import LABEL
    lab = lambda m: (LABEL.get(m, m).replace("Prompt-RAG-Validator", "PRV").replace("UAV-FlowShield", "UAV-FlowShield").replace(" + repair + delegation", " +rep+del").replace(" + repair", " +rep"))
    ax.set_yticklabels([lab(m) for m in ms], fontsize=6.5)
    ax.axvline(len(goals) - 0.5, color="k", lw=0.8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Executed attack success (%)", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"heat_{tag}.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    tag = sys.argv[1]
    S = load(tag)
    scatter(S, tag)
    heat(S, tag, ["undefended", "prompt_only", "spotlight_guard", "prv", "eamsr", "mcp_pep", "spotlight_guard_rep", "mcp_pep_rep", "mcp_pep_repdel",
                  "eamsr_repdel", "det_plus", "flowshield", "fs_chan"])
    print("figures for", tag)
