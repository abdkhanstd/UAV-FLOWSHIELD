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


# ------------------------------------------------------------------ attacker and cost row
def attacker():
    A = load("results", "attacker.json")
    S = A["scales"]
    fig, ax = new()
    for key, lab, ls in (("strong", "with ICP", "-o"), ("weak", "short base, no ICP", "--s")):
        ax.plot(S, [100 * np.mean([x["res"][str(s)][key] for x in A["results"]]) for s in S], ls, color=C["blue"] if key == "strong" else C["grey"],
                label=lab, ms=2.4)
    ax.set_xscale("log")
    ax.set_xticks(S)
    ax.set_xticklabels([str(s) for s in S])
    ax.minorticks_off()
    ax.set_xlabel("noise scale (m)")
    ax.set_ylabel("sites re-identified (%)")
    ax.legend(loc="upper right")
    save(fig, "c_attacker.pdf")


def pareto():
    R = load("results", "eval", "summary_priv_ta.json")["rows"]
    fx = [p for p in R if p.startswith("fixed")]
    fig, ax = new()
    ax.plot([100 * R[p]["site"]["all"][0] for p in fx], [100 * R[p]["subopt"][0] for p in fx], "-o", color=C["grey"], ms=2.2, label="fixed")
    for p, lab, col, mk in (("ucb10", "Wilson", C["green"], "D"), ("pc05", "table", C["blue"], "s"), ("mean05", "mean", C["red"], "^")):
        ax.plot([100 * R[p]["site"]["all"][0]], [100 * R[p]["subopt"][0]], mk, color=col, ms=3.2, label=lab)
    ax.legend(loc="upper right")
    ax.set_xlabel("sites re-identified (%)")
    ax.set_ylabel("suboptimal tours (%)")
    ax.set_xlim(0, 26)
    save(fig, "c_pareto.pdf")


def region():
    M = macros("numbers_p5.tex")
    cats = [("100", "FixA"), ("300", "FixB"), ("600", "FixC"), ("cal", "Ucb"), ("tab", "Tab"), ("none", "Chance")]
    x = np.arange(len(cats))
    one = [float(M[f"reg{k}OneKm"]) for _, k in cats]
    five = [float(M[f"reg{k}FiveKm"]) for _, k in cats]
    fig, ax = new()
    ax.bar(x - 0.2, one, 0.4, color=C["blue"], label="within 1 km")
    ax.bar(x + 0.2, five, 0.4, color=C["orange"], label="within 5 km")
    ax.set_xticks(x)
    ax.set_xticklabels([c for c, _ in cats], fontsize=5.4)
    ax.set_ylabel("centroid located (%)")
    ax.set_ylim(0, 40)
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "c_region.pdf")


# ------------------------------------------------------------------ robustness row
def edge():
    M = macros("numbers_netedge_tb.tex")
    cats = [("1.7B", "QwenSmall"), ("3B", "Granite"), ("4B", "Nemo"), ("8B", "QwenEight"), ("12B", "Same")]
    x = np.arange(len(cats))
    fig, ax = new()
    ax.bar(x - 0.2, [float(M[f"EdgeRecall{k}"]) for _, k in cats], 0.4, color=C["blue"], label="effects recovered")
    ax.bar(x + 0.2, [float(M[f"EdgeMcr{k}"]) for _, k in cats], 0.4, color=C["green"], label="completed")
    ax.plot(x, [float(M[f"EdgeAsr{k}"]) for _, k in cats], "x", color=C["red"], ms=3.5, label="injections executed")
    ax.set_xticks(x)
    ax.set_xticklabels([c for c, _ in cats])
    ax.set_xlabel("edge extractor")
    ax.set_ylabel("share (%)")
    ax.set_ylim(-4, 168)
    ax.set_yticks([0, 50, 100])
    ax.legend(loc="upper left", ncol=1, fontsize=5.2)
    ax.grid(axis="x", visible=False)
    save(fig, "c_edge.pdf")


def netloss():
    M = macros("numbers_netedge_tb.tex")
    keys = [(0, "Zero"), (10, "OnePct"), (30, "ThreePct"), (50, "FivePct"), (100, "All")]
    xs = [k for k, _ in keys]
    fig, ax = new()
    ax.plot(xs, [float(M[f"netMcrHold{k}"]) for _, k in keys], "-o", color=C["orange"], label="hold")
    ax.plot(xs, [float(M[f"netMcrFall{k}"]) for _, k in keys], "-s", color=C["blue"], label="verified fallback")
    ax.set_xlabel("requests lost (%)")
    ax.set_ylabel("missions completed (%)")
    ax.set_ylim(-4, 104)
    ax.legend(loc="lower left")
    save(fig, "c_netloss.pdf")


def sampling():
    M = macros("numbers_samp_tb.tex")
    meths = [("undef.", "Undef"), ("PRV", "Prv"), ("PEP+", "PepRepDel"), ("FS", "Fs")]
    x = np.arange(len(meths))
    fig, ax = new()
    for off, key, col, lab in ((-0.2, "Asr", C["grey"], "injections"), (0.2, "Adapt", C["red"], "adaptive")):
        ax.bar(x + off, [float(M[f"samp{key}{k}"]) for _, k in meths], 0.4, yerr=[float(M[f"samp{key}{k}Sd"]) for _, k in meths],
               color=col, label=lab, error_kw={"elinewidth": 0.6, "capsize": 1.2})
        ax.plot(x + off, [float(M[f"sampDet{key}{k}"]) for _, k in meths], "o", mfc="white", mec="k", ms=2.4, mew=0.6,
                label="greedy" if key == "Adapt" else None)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in meths])
    ax.set_ylabel("executed (%)")
    ax.set_ylim(0, 104)
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "c_sampling.pdf")


if __name__ == "__main__":
    for f in (curves, calib, frontier, secmcr, common_boundary, final, wind, runtime, ttl, attacker, pareto, region, edge, netloss, sampling):
        f()
    print("ok")
