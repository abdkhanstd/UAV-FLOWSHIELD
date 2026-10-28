"""Small single-column figures for the double-column paper: each panel is a 84 x 68 pt canvas with 7 pt Times text, three
panels per column row (paper/figs/c_*.pdf).
usage: python scripts/fig_compact.py"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "..", "paper", "figs")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
W, H = 84 / 72, 68 / 72
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7,
                     "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 5.8,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "xtick.major.pad": 1.2, "ytick.major.pad": 1.2, "axes.labelpad": 1.0, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "legend.handlelength": 1.1, "legend.handletextpad": 0.3, "legend.borderaxespad": 0.1,
                     "legend.labelspacing": 0.15, "lines.linewidth": 1.0, "lines.markersize": 2.6, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.4, "pdf.fonttype": 42})
C = {"dense": "#1f5fa8", "medium": "#d9832b", "sparse": "#3f8f3f", "red": "#c0392b", "grey": "#7f7f7f", "blue": "#1f5fa8", "orange": "#d9832b",
     "green": "#3f8f3f", "purple": "#7a4ea3"}


def new():
    fig, ax = plt.subplots(figsize=(W, H))
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def macros(*files):
    m = {}
    for f in files:
        for name, val in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}", open(os.path.join(AUTO, f), encoding="utf-8").read()):
            m.setdefault(name, val)
    return m


def load(*p):
    return json.load(open(os.path.join(ROOT, *p)))


# ------------------------------------------------------------------ privacy row
def curves():
    P = load("results", "privacy.json")
    grid = P["curve_grid"]
    X = [max(s, 6) for s in grid]
    fig, ax = new()
    for cls in ("dense", "medium", "sparse"):
        cur = [x for x in P["results"] if x["area"].startswith(cls) and "curve" in x]
        ax.plot(X, [100 * np.mean([x["curve"][str(s)]["blind"] for x in cur]) for s in grid], "-o", color=C[cls], label=cls)
        ax.text(*{"dense": (7.5, 20), "medium": (26, 70), "sparse": (190, 62)}[cls], cls, color=C[cls], fontsize=5.8)
    cur = [x for x in P["results"] if "curve" in x]
    ax.plot(X, [100 * np.mean([x["curve"][str(s)]["aligned"] for x in cur]) for s in grid], "--", color=C["grey"], label="frame known")
    ax.text(330, 90, "frame\nknown", color=C["grey"], fontsize=5.6, ha="center")
    ax.axhline(10, color="k", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([6, 25, 100, 400])
    ax.set_xticklabels(["0", "25", "100", "400"])
    ax.minorticks_off()
    ax.set_xlabel("noise scale (m)")
    ax.set_ylabel("sites re-identified (%)")
    ax.set_ylim(-3, 103)
    save(fig, "c_curves.pdf")


def calib():
    M = macros("numbers_covcal_ta.tex", "numbers_calibtab.tex")
    pts = [("mean", float(M["covMeanTenScaleAllTa"]), float(M["covMeanTenViolAllTa"]), C["red"], "o"),
           ("Wilson", float(M["covWilsonThirtyScaleAllTa"]), float(M["covWilsonThirtyViolAllTa"]), C["orange"], "s"),
           ("ladder", float(M["covCpFsHundredScaleAllTa"]), float(M["covCpFsHundredViolAllTa"]), C["blue"], "D"),
           ("table first", float(M["ctTabHundredScale"]), max(float(M["ctTabHundredViol"]), 0.01), C["green"], "^")]
    fig, ax = new()
    for lab, x, y, col, mk in pts:
        ax.plot([x], [y], mk, color=col, ms=4)
        ax.annotate(lab, (x, y), xytext=(3, 2), textcoords="offset points", fontsize=5.8, color=col)
    ax.axvline(float(M["covOracleScaleAllTa"]), color=C["grey"], lw=0.6, ls="--")
    ax.text(float(M["covOracleScaleAllTa"]) + 4, 0.02, "oracle", fontsize=5.5, color=C["grey"], rotation=90, va="bottom")
    ax.axhline(5, color="k", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(0.007, 80)
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    ax.minorticks_off()
    ax.set_xlim(100, 320)
    ax.set_xlabel("mean scale (m)")
    ax.set_ylabel("unsafe release (%)")
    save(fig, "c_calib.pdf")


def frontier():
    M = load("results", "privacy5_C_bonf.json")["missions"]
    sets = [("blind", ["blind"], C["blue"]), ("+4 releases", ["blind", "x4"], C["green"]), ("+portf. 100", ["blind", "x4", "pf100"], C["orange"]),
            ("+portf. 25", ["blind", "x4", "pf100", "pf25"], C["red"])]
    caps = [100, 200, 300, 400, 600, 800]
    fig, ax = new()
    for lab, ths, col in sets:
        share = [100 * np.mean([all(x["cert"][t]["certified"] for t in ths) and max(x["cert"][t]["scale"] for t in ths) <= c for x in M]) for c in caps]
        ax.plot(caps, share, "-o", color=col, label=lab)
        pos = {"blind": (520, 96), "+4 releases": (540, 58), "+portf. 100": (530, 36), "+portf. 25": (200, 7)}[lab]
        ax.text(*pos, lab, color=col, fontsize=5.6, ha="center")
    ax.set_xlabel("cap on noise (m)")
    ax.set_ylabel("planned in cloud (%)")
    ax.set_ylim(-3, 103)
    ax.set_xticks([200, 400, 600, 800])
    save(fig, "c_frontier.pdf")


# ------------------------------------------------------------------ security row
def secmcr():
    S = load("results", "eval", "summary_tbg.json")["summary"]
    lab = {"undefended": "undef.", "prompt_only": "prompt", "spotlight_guard": "spotl.", "prv": "PRV", "eamsr": "EAMSR", "mcp_pep": "PEP",
           "det_plus": "parser", "flowshield": "FS"}
    fig, ax = new()
    for m, l in lab.items():
        x, y = 100 * S[m]["asr"][0], 100 * S[m]["mcr"][0]
        col = C["red"] if m == "flowshield" else (C["green"] if m == "det_plus" else C["grey"])
        ax.plot([x], [y], "*" if m == "flowshield" else "o", color=col, ms=5 if m == "flowshield" else 3)
        off = {"eamsr": (2, -2), "mcp_pep": (2, -1), "prv": (-14, 2), "undefended": (-13, 3), "prompt_only": (5, -1), "spotlight_guard": (-3, -9),
               "flowshield": (4, -1), "det_plus": (4, -3)}[m]
        ax.annotate(l, (x, y), xytext=off, textcoords="offset points", fontsize=5.5, color=col)
    ax.set_xlabel("injections executed (%)")
    ax.set_ylabel("benign completed (%)")
    ax.set_xlim(-4, 70)
    ax.set_ylim(25, 102)
    save(fig, "c_secmcr.pdf")


def common_boundary():
    M = macros("numbers_tbg.tex", "numbers_cb_tbg.tex")
    names = [("Undef", "undef."), ("Prompt", "prompt"), ("Spot", "spotl."), ("Prv", "PRV"), ("Eamsr", "EAMSR"), ("Pep", "PEP")]
    pub = [float(M[f"adaptTbg{k}"]) for k, _ in names]
    cb = [float(M[f"adaptCbTbg{k}"]) for k, _ in names]
    fs = float(M["adaptTbgFs"])
    x = np.arange(len(names))
    fig, ax = new()
    ax.bar(x - 0.2, pub, 0.4, color=C["grey"], label="as published")
    ax.bar(x + 0.2, cb, 0.4, color=C["blue"], label="behind boundary")
    ax.axhline(fs, color=C["red"], lw=0.7, ls="--")
    ax.text(len(names) - 0.6, fs + 3, "FS", color=C["red"], fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in names], rotation=60, fontsize=5.3)
    ax.set_ylabel("adaptive executed (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.07, 1.08))
    save(fig, "c_cb.pdf")


def final():
    M = macros("numbers_final.tex")
    prof = [("Hab", "approve\nall"), ("Fin", "atten-\ntive"), ("FinExp", "expert")]
    prof = [("FinHab", "approve\nall"), ("Fin", "atten-\ntive"), ("FinExp", "expert")]
    att = [10 * float(M[f"fin{p}AdOneT"]) for p, _ in prof]
    lost = [100 * (40 - float(M[f"fin{p}Dyn"])) / 40 for p, _ in prof]
    x = np.arange(len(prof))
    fig, ax = new()
    ax.bar(x - 0.2, att, 0.4, color=C["red"], label="attack executed")
    ax.bar(x + 0.2, lost, 0.4, color=C["orange"], label="legit. request lost")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in prof], fontsize=5.6)
    ax.set_ylabel("requests (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.07, 1.1))
    save(fig, "c_final.pdf")


# ------------------------------------------------------------------ execution row
def wind():
    D = load("results", "eval", "dynamics.json")
    miss = [m for m in D["missions"] if "nominal" in m and f"{D['winds'][0]}_{D['gps'][0]}" in m["nominal"]]
    winds = D["winds"]
    ex = [100 * np.mean([m["nominal"][f"{w}_{D['gps'][0]}"]["exceed"] for m in miss]) for w in winds]
    comp = []
    for w in winds:
        if w == 0:
            comp.append(100.0)
            continue
        recs = [m["aware"].get(str(w), {}) for m in miss]
        comp.append(100 * np.mean([bool(r.get("success")) for r in recs]))
    fig, ax = new()
    ax.plot(winds, ex, "-o", color=C["red"], label="nominal: over budget")
    ax.plot(winds, comp, "-s", color=C["blue"], label="wind-aware: completed")
    ax.set_xlabel("mean wind (m/s)")
    ax.set_ylabel("missions (%)")
    ax.set_ylim(-3, 103)
    ax.set_xticks(winds)
    ax.legend(loc="center left", bbox_to_anchor=(-0.03, 0.55))
    save(fig, "c_wind.pdf")


def runtime():
    R = load("results", "runtime.json")
    ev = [("wind_step", "wind"), ("nfz_grow", "zone"), ("floor_update", "floor"), ("gps_degraded", "GPS"), ("combined", "all")]
    o = [100 * np.mean([r["unsafe"] for r in R if r["event"] == e and r["sup"] == "open_loop"]) for e, _ in ev]
    s = [100 * np.mean([r["unsafe"] for r in R if r["event"] == e and r["sup"] == "reverify"]) for e, _ in ev]
    x = np.arange(len(ev))
    fig, ax = new()
    ax.bar(x - 0.2, o, 0.4, color=C["grey"], label="open loop")
    ax.bar(x + 0.2, s, 0.4, color=C["blue"], label="supervisor")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in ev], fontsize=5.6)
    ax.set_ylabel("unsafe flights (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.03, 1.08))
    save(fig, "c_runtime.pdf")


def ttl():
    A = load("results", "authority.json")
    t = sorted(int(k) for k in A["availability"])
    fig, ax = new()
    ax.plot(t, [100 * A["availability"][str(k)] for k in t], "-o", color=C["blue"], label="legitimate accepted")
    ax.plot(t, [0 for _ in t], "-s", color=C["red"], label="replays accepted")
    ax.set_xscale("log")
    ax.set_xticks([30, 60, 120, 300, 900])
    ax.set_xticklabels(["30", "60", "120", "300", "900"])
    ax.minorticks_off()
    ax.set_xlabel("ticket lifetime (s)")
    ax.set_ylabel("tickets (%)")
    ax.set_ylim(-4, 104)
    ax.legend(loc="center right", bbox_to_anchor=(1.05, 0.45))
    save(fig, "c_ttl.pdf")


if __name__ == "__main__":
    for f in (curves, calib, frontier, secmcr, common_boundary, final, wind, runtime, ttl):
        f()
    print("ok")
