"""Privacy figures and tables from results/privacy*.json and results/attacker.json."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from stats_util import boot_ci

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "..", "paper", "figs")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
os.makedirs(FIG, exist_ok=True)
os.makedirs(AUTO, exist_ok=True)
CLS = ["dense", "medium", "sparse"]
CL = {"dense": "Dense registry", "medium": "Medium registry", "sparse": "Sparse registry"}
COL = {"blind": "#1f5fa8", "aligned": "#c0392b", "weak": "#7f8c8d", "strong": "#1f5fa8"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
cls_of = lambda a: a.split("_")[0]
macros = []


WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def W(n):
    """Spell digits out: LaTeX control words contain letters only."""
    return "".join(WORDS[int(ch)] for ch in str(n))


def mac(name, val):
    macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def load(name):
    return json.load(open(os.path.join(ROOT, "results", name)))


def curves():
    P = load("privacy.json")
    grid = P["curve_grid"]
    X = [max(s, 6) for s in grid]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, c in zip(axs, CLS):
        cur = [x for x in P["results"] if cls_of(x["area"]) == c and "curve" in x]
        med = np.median([x["heldout"]["scale"] for x in P["results"] if cls_of(x["area"]) == c])
        for key, lab, ls in (("blind", "Blind registration (secret frame)", "-"), ("aligned", "Frame compromised (aligned)", "--")):
            m, lo, hi = [], [], []
            for s in grid:
                v = [x["curve"][str(s)][key] for x in cur]
                a, l, h = boot_ci(v, B=500)
                m.append(a); lo.append(l); hi.append(h)
            ax.plot(X, m, ls, color=COL[key], lw=1.4, marker="o", ms=2.8, label=lab)
            ax.fill_between(X, lo, hi, color=COL[key], alpha=0.15, lw=0)
        ax.axhline(0.10, color="k", lw=0.7, ls=":")
        ax.axvline(med, color="#2c7a3f", lw=1.0, ls="-.")
        ax.text(med * 0.95, 1.0, f"{med:.0f} m", color="#2c7a3f", fontsize=6.5, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xticks([6, 10, 25, 50, 100, 200, 400, 800])
        ax.set_xticklabels(["0", "10", "25", "50", "100", "200", "400", "800"], fontsize=6.5)
        ax.minorticks_off()
        ax.set_title(CL[c], fontsize=8)
        ax.set_xlabel("Noise scale (mean displacement, m)")
        ax.set_ylim(-0.02, 1.03)
    axs[0].set_ylabel("Sites re-identified")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "privacy_curves.pdf"))
    plt.close(fig)
    # numbers
    for c in CLS:
        cur = [x for x in P["results"] if cls_of(x["area"]) == c and "curve" in x]
        for s in (0, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800):
            for key in ("blind", "aligned"):
                mac(f"curve{key.capitalize()}{c.capitalize()}S{W(s)}", f"{100 * np.mean([x['curve'][str(s)][key] for x in cur]):.0f}")


def heldout():
    P = load("privacy.json")
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
    ax = axs[0]
    for i, c in enumerate(CLS):
        sc = np.array([x["heldout"]["scale"] for x in P["results"] if cls_of(x["area"]) == c], float)
        jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(sc))
        ax.scatter(i + jit, sc, s=7, alpha=0.5, color="#1f5fa8", lw=0)
        ax.hlines(np.median(sc), i - 0.3, i + 0.3, color="k", lw=1.5)
    ax.axhline(300, color="#c0392b", lw=1.0, ls="--")
    ax.text(-0.45, 340, "single fixed scale that\nsuffices in every registry", color="#c0392b", fontsize=6.3, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels([c.capitalize() for c in CLS])
    ax.set_ylabel("Calibrated noise scale (m)")
    ax.set_title("(a) Scale chosen per mission", fontsize=8)
    ax.set_ylim(7, 1200)
    ax = axs[1]
    data_b = [[x["heldout"]["blind"] for x in P["results"] if cls_of(x["area"]) == c] for c in CLS]
    data_a = [[x["heldout"]["aligned"] for x in P["results"] if cls_of(x["area"]) == c] for c in CLS]
    pos = np.arange(3)
    bp1 = ax.boxplot(data_b, positions=pos - 0.18, widths=0.3, patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data_a, positions=pos + 0.18, widths=0.3, patch_artist=True, showfliers=False)
    for b in bp1["boxes"]:
        b.set(facecolor=COL["blind"], alpha=0.6, lw=0.7)
    for b in bp2["boxes"]:
        b.set(facecolor=COL["aligned"], alpha=0.6, lw=0.7)
    for bp in (bp1, bp2):
        for k in ("medians",):
            for l in bp[k]:
                l.set(color="k", lw=1.0)
    ax.axhline(0.10, color="k", ls=":", lw=0.7)
    ax.set_xticks(pos)
    ax.set_xticklabels([c.capitalize() for c in CLS])
    ax.set_ylabel("Sites re-identified (held-out draws)")
    ax.set_title("(b) Risk at the calibrated scale", fontsize=8)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Blind registration", "Frame compromised"], fontsize=6.5, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "privacy_calibration.pdf"))
    plt.close(fig)

    F = load("privacy_fixed.json") if os.path.exists(os.path.join(ROOT, "results", "privacy_fixed.json")) else None
    L = ["\\begin{tabular}{lcccccc}", "\\toprule",
         "Registry & Missions & Median $\\bar s^\\star$ (m) & Blind re-id & $\\Pr[\\text{re-id}>0.2]$ & Aligned re-id & Home error (km) \\\\", "\\midrule"]
    for c in CLS:
        r = [x["heldout"] for x in P["results"] if cls_of(x["area"]) == c]
        b = [h["blind"] for h in r]
        a, lo, hi = boot_ci(b, B=1000)
        tail = float(np.mean(np.array(b) > 0.2))
        al = float(np.mean([h["aligned"] for h in r]))
        he = float(np.median([h["home_err"] for h in r])) / 1000
        med = float(np.median([h["scale"] for h in r]))
        L.append(f"{c.capitalize()} & {len(r)} & {med:.0f} & {100 * a:.1f}\\,\\scriptsize[{100 * lo:.1f}, {100 * hi:.1f}]\\normalsize & {100 * tail:.1f} & {100 * al:.1f} & {he:.1f} \\\\")
        mac(f"heldBlind{c.capitalize()}", f"{100 * a:.1f}")
        mac(f"heldBlindLo{c.capitalize()}", f"{100 * lo:.1f}")
        mac(f"heldBlindHi{c.capitalize()}", f"{100 * hi:.1f}")
        mac(f"heldTail{c.capitalize()}", f"{100 * tail:.0f}")
        mac(f"heldAligned{c.capitalize()}", f"{100 * al:.0f}")
        mac(f"heldScale{c.capitalize()}", f"{med:.0f}")
        mac(f"heldHome{c.capitalize()}", f"{he:.1f}")
        mac(f"nMissions{c.capitalize()}", f"{len(r)}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_privacy.tex"), "w").write("\n".join(L) + "\n")

    if F:
        ids = {x["id"]: x for x in F["results"]}
        L = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Policy & \\multicolumn{3}{c}{Mean re-id (\\%) by registry} & Mean noise (m) \\\\",
             " & Dense & Medium & Sparse & \\\\", "\\midrule"]
        scale_of = {x["id"]: x["heldout"]["scale"] for x in P["results"]}
        # calibrated (same missions)
        mc = {c: [x["heldout"]["blind"] for x in P["results"] if cls_of(x["area"]) == c and x["id"] in ids] for c in CLS}
        sc = [scale_of[i] for i in ids]
        L.append("Attack-calibrated (per mission) & " + " & ".join(f"{100 * np.mean(mc[c]):.1f}" for c in CLS) + f" & {np.mean(sc):.0f} \\\\")
        for s in F["fixed"]:
            vals = {c: [x["fixed"][str(s)] for x in F["results"] if cls_of(x["area"]) == c] for c in CLS}
            L.append(f"Fixed {s}\\,m & " + " & ".join(f"{100 * np.mean(vals[c]):.1f}" for c in CLS) + f" & {s} \\\\")
            for c in CLS:
                mac(f"fixed{c.capitalize()}S{W(s)}", f"{100 * np.mean(vals[c]):.1f}")
            tails = np.mean([v > 0.2 for c in CLS for v in vals[c]])
            mac(f"fixedTailS{W(s)}", f"{100 * tails:.0f}")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(AUTO, "table_policy.tex"), "w").write("\n".join(L) + "\n")
        mac("calMeanScale", f"{np.mean(sc):.0f}")
        mac("calTailAll", f"{100 * np.mean([v > 0.2 for c in CLS for v in mc[c]]):.0f}")
        for c in CLS:
            vv = np.array([scale_of[i] for i in ids if cls_of(ids[i]["area"]) == c])
            mac(f"calMeanScale{c.capitalize()}", f"{vv.mean():.0f}")


def attacker():
    A = load("attacker.json")
    S = A["scales"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True)
    for ax, c in zip(axs, CLS):
        r = [x for x in A["results"] if cls_of(x["area"]) == c]
        for key, lab, ls in (("strong", "Strong attacker (ICP refinement)", "-"), ("weak", "Weak attacker (short base, no refinement)", "--")):
            m = [np.mean([x["res"][str(s)][key] for x in r]) for s in S]
            ax.plot(S, m, ls, color=COL[key], marker="o", ms=3, lw=1.3, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(S)
        ax.set_xticklabels([str(s) for s in S], fontsize=6.5)
        ax.minorticks_off()
        ax.set_title(CL[c], fontsize=8)
        ax.set_xlabel("Noise scale (m)")
        for s in S:
            mac(f"att{c.capitalize()}S{W(s)}Weak", f"{100 * np.mean([x['res'][str(s)]['weak'] for x in r]):.0f}")
            mac(f"att{c.capitalize()}S{W(s)}Strong", f"{100 * np.mean([x['res'][str(s)]['strong'] for x in r]):.0f}")
    axs[0].set_ylabel("Sites re-identified")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "attacker_strength.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    curves()
    heldout()
    attacker()
    open(os.path.join(AUTO, "numbers_privacy.tex"), "w").write("\n".join(macros) + "\n")
    print("privacy figures and tables written")
