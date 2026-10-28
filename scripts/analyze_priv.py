"""Tables, figure and macros for the privacy analysis.

usage: python scripts/analyze_priv.py --tag ta --priv results/privacy3_mata.json --regret results/regret.json [--attackers results/privacy4.json]
Reads the held-out evaluation of the disclosure policies (exp_privacy3), the planning-regret curves (exp_regret) and the
side-information attackers (exp_privacy4). Writes ../paper/auto/numbers_priv_<tag>.tex, table_policy_<tag>.tex,
table_attackers_<tag>.tex and ../paper/figs/pareto_<tag>.pdf."""
import argparse
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
AUTO = os.path.join(ROOT, "..", "paper", "auto")
FIG = os.path.join(ROOT, "..", "paper", "figs")
CLS = ["dense", "medium", "sparse"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
NAME = {"fixed75": "fixed 75 m", "fixed200": "fixed 200 m", "fixed400": "fixed 400 m", "fixed25": "fixed 25 m", "fixed50": "fixed 50 m", "fixed100": "fixed 100 m", "fixed150": "fixed 150 m", "fixed300": "fixed 300 m",
        "fixed600": "fixed 600 m", "fixed800": "fixed 800 m", "pc10": "per-class fixed (mean $\\leq$10\\,\\%)", "pc05": "per-class fixed (mean $\\leq$5\\,\\%)",
        "mean10": "calibrated, mean rule ($\\rho$=10\\,\\%)", "mean05": "calibrated, mean rule ($\\rho$=5\\,\\%)",
        "ucb10": "calibrated, upper limit ($\\rho$=10\\,\\%)", "ucb10m": "calibrated, upper limit, mission-level", "ucb05": "calibrated, upper limit ($\\rho$=5\\,\\%)",
        "cert30": "certified, exact test, $T$=30", "cert60": "certified, exact test, $T$=60", "cert100": "certified, exact test, $T$=100"}
ORDER = ["fixed75", "fixed100", "fixed150", "fixed200", "fixed300", "fixed400", "fixed600", "pc10", "pc05", "mean10", "mean05", "ucb10", "ucb10m", "ucb05", "cert30", "cert60", "cert100"]
macros = []


def mac(n, v):
    macros.append(f"\\newcommand{{\\{n}}}{{{v}}}")


def code(p):
    return p.replace("fixed", "Fixed").replace("mean", "Mean").replace("ucb", "Ucb").replace("cert", "Cert").replace("pc", "Pc").translate(str.maketrans("0123456789", "ZOTHFVSEIN")).replace("m", "M").replace("_", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--priv", required=True)
    ap.add_argument("--regret", default="results/regret.json")
    ap.add_argument("--attackers", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="held-out files with additional policies")
    a = ap.parse_args()
    T = a.tag[0].upper() + a.tag[1:]
    P = json.load(open(os.path.join(ROOT, a.priv)))
    H = P["heldout"]
    for ex in a.extra:
        X = {x["id"]: x["pol"] for x in json.load(open(os.path.join(ROOT, ex)))["heldout"]}
        for x in H:
            x["pol"].update(X.get(x["id"], {}))
    R = {m["id"]: m for m in json.load(open(os.path.join(ROOT, a.regret)))["missions"]}
    pols = [p for p in ORDER if p in H[0]["pol"]]
    rows = {}
    for p in pols:
        rec = {"scale": {}, "site": {}, "any": 0, "over20": {}, "regret": 0, "subopt": 0}
        per_m = []
        for c in CLS:
            h = [x for x in H if x["area"].startswith(c)]
            rec["scale"][c] = float(np.mean([x["pol"][p]["scale"] for x in h]))
            rec["site"][c] = boot_ci([x["pol"][p]["site"] for x in h])
            rec["over20"][c] = float(np.mean([x["pol"][p]["site"] > 0.20 for x in h]))
        allm = H
        rec["scale"]["all"] = float(np.mean([x["pol"][p]["scale"] for x in allm]))
        rec["site"]["all"] = boot_ci([x["pol"][p]["site"] for x in allm])
        rec["any"] = boot_ci([x["pol"][p]["any"] for x in allm])
        rec["over20"]["all"] = float(np.mean([x["pol"][p]["site"] > 0.20 for x in allm]))
        reg = [R[x["id"]]["curve"][str(int(x["pol"][p]["scale"]))]["regret"] for x in allm if x["id"] in R]
        sub = [R[x["id"]]["curve"][str(int(x["pol"][p]["scale"]))]["subopt"] for x in allm if x["id"] in R]
        rec["regret"] = boot_ci(reg)
        rec["subopt"] = boot_ci(sub)
        rows[p] = rec
        c = code(p)
        for k in ("dense", "medium", "sparse", "all"):
            mac(f"scale{c}{k.capitalize()}{T}", f"{rec['scale'][k]:.0f}")
            mac(f"site{c}{k.capitalize()}{T}", f"{100 * rec['site'][k][0]:.1f}")
            mac(f"over{c}{k.capitalize()}{T}", f"{100 * rec['over20'][k]:.1f}")
        mac(f"any{c}{T}", f"{100 * rec['any'][0]:.1f}")
        mac(f"regret{c}{T}", f"{100 * rec['regret'][0]:.2f}")
        mac(f"subopt{c}{T}", f"{100 * rec['subopt'][0]:.1f}")
    L = ["\\begin{tabular}{lccccccc}", "\\toprule",
         "Policy & Mean scale (m) & \\multicolumn{3}{c}{Per-site re-id (\\%) by class} & Any site (\\%) & $>$20\\,\\% of missions & Suboptimal tour (\\%) \\\\",
         " & & dense & medium & sparse & & (\\%) & \\\\", "\\midrule"]
    for p in pols:
        r = rows[p]
        L.append(f"{NAME[p]} & {r['scale']['all']:.0f} & " + " & ".join(f"{100 * r['site'][c][0]:.1f}" for c in CLS) +
                 f" & {100 * r['any'][0]:.1f} & {100 * r['over20']['all']:.1f} & {100 * r['subopt'][0]:.1f} \\\\")
        if p in ("fixed800", "pc05"):
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, f"table_policy_{a.tag}.tex"), "w").write("\n".join(L) + "\n")
    # Pareto figure
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    fx = [p for p in pols if p.startswith("fixed")]
    for k, (ykey, yl) in enumerate((("subopt", "Probability that the planned tour is not optimal (\\%)"), ("scale", "Mean noise scale (m)"))):
        yv = lambda p: 100 * rows[p]["subopt"][0] if ykey == "subopt" else rows[p]["scale"]["all"]
        xs = [100 * rows[p]["site"]["all"][0] for p in fx]
        ax[k].plot(xs, [yv(p) for p in fx], "-o", color="#888", ms=3, lw=1, label="fixed scales (%s-%s m)" % (min(int(p[5:]) for p in fx), max(int(p[5:]) for p in fx)))
        for p, col, mk in (("pc10", "#1f77b4", "s"), ("pc05", "#6baed6", "s"), ("mean10", "#d62728", "^"), ("mean05", "#ff9896", "^"), ("ucb10", "#2ca02c", "D"), ("ucb10m", "#98df8a", "D"), ("cert60", "#9467bd", "P"), ("cert30", "#c5b0d5", "P")):
            if p in rows:
                ax[k].plot(100 * rows[p]["site"]["all"][0], yv(p), mk, color=col, ms=5, label=NAME[p].replace("\\,", "").replace("\\%", "%").replace("$\\leq$", "≤").replace("$\\rho$", "ρ"))
        ax[k].set_xlabel("Held-out per-site re-identification (%)")
        ax[k].set_ylabel(yl.replace("\\%", "%"), fontsize=7)
    ax[0].legend(fontsize=5.6, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"pareto_{a.tag}.pdf"))
    plt.close(fig)
    # paired differences: ucb10 versus the fixed policy with the closest overall re-identification
    out = {"rows": rows}
    # distance of each calibrated policy from the curve of fixed scales at equal held-out re-identification (linear interpolation)
    fxs = sorted(fx, key=lambda q: rows[q]["site"]["all"][0])
    xs = np.array([rows[q]["site"]["all"][0] for q in fxs])
    for q in rows:
        if q.startswith("fixed") or not fxs:
            continue
        x = rows[q]["site"]["all"][0]
        if x < xs.min() or x > xs.max():
            continue
        gs = np.interp(x, xs, [rows[f]["subopt"][0] for f in fxs]) - rows[q]["subopt"][0]
        gn = np.interp(x, xs, [rows[f]["scale"]["all"] for f in fxs]) - rows[q]["scale"]["all"]
        gr = np.interp(x, xs, [rows[f]["regret"][0] for f in fxs]) - rows[q]["regret"][0]
        mac(f"gapSubopt{code(q)}{T}", f"{100 * gs:.1f}")
        mac(f"gapScale{code(q)}{T}", f"{gn:.0f}")
        mac(f"gapRegret{code(q)}{T}", f"{100 * gr:.2f}")
        print("gap vs fixed curve", q, "subopt %.1f pts, scale %.0f m, regret %.2f pts" % (100 * gs, gn, 100 * gr))
    # paired comparisons over missions: privacy and planning cost of a calibrated policy against a fixed scale
    def pair(pa, pb):
        ids = [x["id"] for x in H if x["id"] in R]
        ra = np.array([[x["pol"][pa]["site"] for x in H if x["id"] == i][0] for i in ids])
        rb = np.array([[x["pol"][pb]["site"] for x in H if x["id"] == i][0] for i in ids])
        sa = np.array([R[i]["curve"][str(int([x["pol"][pa]["scale"] for x in H if x["id"] == i][0]))]["subopt"] for i in ids])
        sb = np.array([R[i]["curve"][str(int([x["pol"][pb]["scale"] for x in H if x["id"] == i][0]))]["subopt"] for i in ids])
        ga = np.array([R[i]["curve"][str(int([x["pol"][pa]["scale"] for x in H if x["id"] == i][0]))]["regret"] for i in ids])
        gb = np.array([R[i]["curve"][str(int([x["pol"][pb]["scale"] for x in H if x["id"] == i][0]))]["regret"] for i in ids])
        return {"reid": boot_ci(ra - rb), "subopt": boot_ci(sa - sb), "regret": boot_ci(ga - gb)}
    out["pairs"] = {}
    for pa, pb in (("ucb10", "fixed300"), ("pc05", "fixed300"), ("ucb10", "pc05"), ("mean05", "fixed150"), ("pc10", "fixed150")):
        if pa in rows and pb in rows:
            out["pairs"][f"{pa}_vs_{pb}"] = pair(pa, pb)
            for k, v in out["pairs"][f"{pa}_vs_{pb}"].items():
                c = code(pa) + "Vs" + code(pb)
                mac(f"pair{k.capitalize()}{c}{T}", f"{100 * v[0]:.2f}")
                mac(f"pair{k.capitalize()}{c}{T}Lo", f"{100 * v[1]:.2f}")
                mac(f"pair{k.capitalize()}{c}{T}Hi", f"{100 * v[2]:.2f}")
    if "ucb10" in rows and fx:
        close = min(fx, key=lambda p: abs(rows[p]["site"]["all"][0] - rows["ucb10"]["site"]["all"][0]))
        d = [R[x["id"]]["curve"][str(int(x["pol"]["ucb10"]["scale"]))]["subopt"] - R[x["id"]]["curve"][str(int(x["pol"][close]["scale"]))]["subopt"]
             for x in H if x["id"] in R]
        out["ucb10_vs_close_fixed"] = {"fixed": close, "diff_subopt": boot_ci(d)}
    if a.attackers:
        A = json.load(open(os.path.join(ROOT, a.attackers)))["missions"]
        pol_a = list(A[0]["res"])
        att = ["blind", "fuse2", "fuse4", "portfolio25", "portfolio100", "anchor", "budget"]
        att_name = {"blind": "blind", "fuse2": "two releases fused", "fuse4": "four releases fused", "portfolio25": "portfolio of 25", "portfolio100": "portfolio of 100",
                    "anchor": "one site known", "budget": "budget-aware"}
        L = ["\\begin{tabular}{l" + "c" * len(pol_a) + "}", "\\toprule", "Attacker & " + " & ".join(
            {"fixed300": "fixed 300 m", "mean10": "mean rule", "ucb10": "upper limit", "ucb10_pf100": "upper limit, portfolio prior", "ucb10_x4": "upper limit, 4 releases"}.get(p, p) for p in pol_a) + " \\\\", "\\midrule"]
        L.append("Mean scale (m) & " + " & ".join(f"{np.mean([m['res'][p]['scale'] for m in A]):.0f}" for p in pol_a) + " \\\\")
        for p in pol_a:
            mac(f"attScale{code(p)}{T}", f"{np.mean([m['res'][p]['scale'] for m in A]):.0f}")
        for t in att:
            cells = []
            for p in pol_a:
                v = [m["res"][p][t] for m in A if t in m["res"][p]]
                cells.append(f"{100 * np.mean(v):.1f}" if v else "--")
                if v:
                    mac(f"att{t.capitalize().translate(str.maketrans('0123456789', 'ZOTHFVSEIN'))}{code(p)}{T}", f"{100 * np.mean(v):.1f}")
            L.append(f"{att_name[t]} & " + " & ".join(cells) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(AUTO, f"table_attackers_{a.tag}.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, f"numbers_priv_{a.tag}.tex"), "w").write("\n".join(macros) + "\n")
    json.dump(out, open(os.path.join(ROOT, "results", "eval", f"summary_priv_{a.tag}.json"), "w"), indent=1, default=float)
    for p in pols:
        r = rows[p]
        print(f"{NAME[p]:48s} scale {r['scale']['all']:5.0f}  site " + " ".join(f"{100 * r['site'][c][0]:5.1f}" for c in CLS) +
              f"  any {100 * r['any'][0]:5.1f}  >20% {100 * r['over20']['all']:5.1f}  subopt {100 * r['subopt'][0]:5.1f}  regret {100 * r['regret'][0]:.2f}")
    if "ucb10_vs_close_fixed" in out:
        print(out["ucb10_vs_close_fixed"])
    for k, v in out.get("pairs", {}).items():
        print(k, {a: tuple(round(100 * x, 2) for x in b) for a, b in v.items()})


if __name__ == "__main__":
    main()
