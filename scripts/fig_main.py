"""Figures for the main LLM results: ASR bars, per-goal heatmap, utility. Reads results/summary_<tag>.json."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "..", "paper", "figs")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
MAIN = ["undefended", "prompt_only", "spotlight_guard", "prv", "eamsr", "mcp_pep", "flowshield", "det_template"]
SHORT = {"det_template": "Determ.\n(no LLM)", "undefended": "Undefended", "prompt_only": "Prompt-only", "spotlight_guard": "Spotlight-\nGuard", "prv": "PRV",
         "eamsr": "EAMSR", "mcp_pep": "MCP-PEP", "flowshield": "UAV-\nFlowShield"}
COL = {"std": "#1f5fa8", "ad": "#e08a1e", "ok": "#2c7a3f", "bad": "#c0392b"}


def load(tag):
    return json.load(open(os.path.join(ROOT, "results", f"summary_{tag}.json")))["summary"]


def err(p):
    return [[100 * (p[0] - p[1])], [100 * (p[2] - p[0])]]


def asr_fig(S, tag):
    ms = [m for m in MAIN if m in S]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(ms))
    w = 0.36
    for off, key, lab, col in ((-w / 2, "asr", "Injection suite (G1-G6, 108 episodes)", COL["std"]),
                               (w / 2, "asr_adapt", "Adaptive attacks (AD1-AD3, 45 episodes)", COL["ad"])):
        v = [S[m][key] for m in ms]
        ax.bar(x + off, [100 * a[0] for a in v], w, color=col, label=lab, yerr=np.clip(np.array([[100 * (a[0] - a[1]) for a in v],
               [100 * (a[2] - a[0]) for a in v]]), 0, None), error_kw=dict(lw=0.8, capsize=2))
        for xi, a in zip(x + off, v):
            ax.text(xi, 100 * a[2] + 1.5, f"{100 * a[0]:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in ms], fontsize=7)
    ax.set_ylabel("Executed attack success (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6.8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"asr_{tag}.pdf"))
    plt.close(fig)


def heat_fig(S, tag):
    ms = [m for m in MAIN if m in S]
    cols = ["G1", "G2", "G3", "G4", "G5", "G6", "AD1", "AD2", "AD3", "AD4"]
    M = np.full((len(ms), len(cols)), np.nan)
    for i, m in enumerate(ms):
        for j, c in enumerate(cols):
            d = S[m].get("asr_goal", {}) if c.startswith("G") else S[m].get("asr_ad", {})
            if c in d:
                M[i, j] = 100 * d[c][0]
            if c == "AD4" and "asr_ad4" in S[m]:
                M[i, j] = 100 * S[m]["asr_ad4"][0]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    for i in range(len(ms)):
        for j in range(len(cols)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color="#888")
            else:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 55 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([SHORT[m].replace("\n", "") for m in ms], fontsize=7)
    ax.axvline(5.5, color="k", lw=0.8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label("Executed attack success (%)", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"heatmap_{tag}.pdf"))
    plt.close(fig)


def util_fig(S, tag):
    ms = [m for m in MAIN if m in S]
    cats = ["benign_plain", "benign_constraint", "benign_dynamic", "benign_tight"]
    names = ["Plain", "Constraint", "Dynamic task", "Tight budget"]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(ms))
    w = 0.2
    cols = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    for k, (c, nme) in enumerate(zip(cats, names)):
        v = [S[m]["mcr_cat"][c][0] * 100 if c in S[m]["mcr_cat"] else np.nan for m in ms]
        ax.bar(x + (k - 1.5) * w, v, w, color=cols[k], label=nme)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in ms], fontsize=7)
    ax.set_ylabel("Mission completion (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"utility_{tag}.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "gemma"
    S = load(tag)
    asr_fig(S, tag)
    heat_fig(S, tag)
    util_fig(S, tag)
    print("figures written for", tag)
