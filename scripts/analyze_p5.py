"""Tables and macros for exp_privacy5.py (region-level disclosure, attacker saturation, disclosure decision).
usage: python scripts/analyze_p5.py   -> ../paper/auto/table_region.tex, table_saturation.tex, table_decision.tex, table_misspec.tex, numbers_p5.tex"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
CLS = ["dense", "medium", "sparse"]
macros = []
KW = {0: "Zero", 1: "One", 2: "Two", 3: "Three"}
CW = {200: "Two", 400: "Four", 800: "Eight"}


def mac(n, v):
    macros.append(f"\\newcommand{{\\{n}}}{{{v}}}")


def load(name):
    p = os.path.join(ROOT, "results", name)
    return json.load(open(p))["missions"] if os.path.exists(p) else None


def part_A():
    M = load("privacy5_A.json")
    if not M:
        return
    pols = [("ucb10", "calibrated, upper limit (Wilson rule)", "Ucb"), ("table", "per-class table", "Tab"), ("fixed100", "fixed 100 m", "FixA"),
            ("fixed300", "fixed 300 m", "FixB"), ("fixed600", "fixed 600 m", "FixC")]
    L = ["\\begin{tabular}{lcccccc}", "\\toprule", "Policy & Mean scale (m) & Sites identified (\\%) & Centroid within 1\\,km (\\%) & within 2\\,km (\\%) & within 5\\,km (\\%) & Median error (m) \\\\", "\\midrule"]
    for p, name, c in pols:
        g = lambda k: float(np.mean([x["pol"][p][k] for x in M]))
        med = float(np.median([x["pol"][p]["median_err"] for x in M]))
        L.append(f"{name} & {g('scale'):.0f} & {100 * g('site'):.1f} & {100 * g('within1000'):.1f} & {100 * g('within2000'):.1f} & {100 * g('within5000'):.1f} & {med:.0f} \\\\")
        mac(f"reg{c}Scale", f"{g('scale'):.0f}")
        mac(f"reg{c}Site", f"{100 * g('site'):.1f}")
        mac(f"reg{c}OneKm", f"{100 * g('within1000'):.1f}")
        mac(f"reg{c}TwoKm", f"{100 * g('within2000'):.1f}")
        mac(f"reg{c}FiveKm", f"{100 * g('within5000'):.1f}")
        mac(f"reg{c}HalfKm", f"{100 * g('within500'):.1f}")
        mac(f"reg{c}Med", f"{med:.0f}")
        for cls in CLS:
            h = [x for x in M if x["area"].startswith(cls)]
            mac(f"reg{c}{cls.capitalize()}OneKm", f"{100 * np.mean([x['pol'][p]['within1000'] for x in h]):.1f}")
            mac(f"reg{c}{cls.capitalize()}Site", f"{100 * np.mean([x['pol'][p]['site'] for x in h]):.1f}")
    # no-information baseline: the mean of k registry points drawn at random (what an attacker that matched nothing would output)
    B = json.load(open(os.path.join(ROOT, "data", "bench.json")))
    eps = {e["id"]: e for e in B["episodes"]}
    rng = np.random.default_rng(0)
    within = {500: [], 1000: [], 2000: [], 5000: []}
    meds = []
    for m in M:
        ep = eps[m["id"]]
        C = np.array(B["areas"][ep["area"]]["C"])
        k = len(ep["sites"])
        truth = C[[s["cand_idx"] for s in ep["sites"]]].mean(0)
        e = np.array([np.linalg.norm(C[rng.integers(len(C), size=k)].mean(0) - truth) for _ in range(200)])
        for r in within:
            within[r].append(float((e <= r).mean()))
        meds.append(float(np.median(e)))
    L.append("\\midrule")
    L.append(f"no information (random registry points) & -- & 0 & {100 * np.mean(within[1000]):.1f} & {100 * np.mean(within[2000]):.1f} & {100 * np.mean(within[5000]):.1f} & {np.median(meds):.0f} \\\\")
    mac("regChanceOneKm", f"{100 * np.mean(within[1000]):.1f}")
    mac("regChanceTwoKm", f"{100 * np.mean(within[2000]):.1f}")
    mac("regChanceFiveKm", f"{100 * np.mean(within[5000]):.1f}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_region.tex"), "w").write("\n".join(L) + "\n")
    mac("regN", len(M))


def part_B():
    M = load("privacy5_B.json")
    if not M:
        return
    variants = [("base", "as in the paper (64 hypotheses, 6 iterations, longest pair)"), ("topk256", "256 hypotheses"), ("topk1024", "1024 hypotheses"),
                ("icp12", "12 ICP iterations"), ("base3", "three longest pairs as base"), ("strong", "three base pairs, 256 hypotheses, 12 iterations")]
    L = ["\\begin{tabular}{lcccc}", "\\toprule", "Attacker & dense (\\%) & medium (\\%) & sparse (\\%) & fixed 300\\,m (\\%) \\\\", "\\midrule"]
    gain = 0.0
    for v, name in variants:
        cells = []
        for cls in CLS:
            h = [x for x in M if x["area"].startswith(cls)]
            cells.append(100 * np.mean([x["pol"]["table"][v] for x in h]))
        f3 = 100 * np.mean([x["pol"]["fixed300"][v] for x in M])
        L.append(f"{name} & " + " & ".join(f"{c:.1f}" for c in cells) + f" & {f3:.1f} \\\\")
        code = {"base": "Base", "topk256": "KTwo", "topk1024": "KThous", "icp12": "Icp", "base3": "Three", "strong": "Strong"}[v]
        for cls, c in zip(CLS, cells):
            mac(f"sat{code}{cls.capitalize()}", f"{c:.1f}")
        mac(f"sat{code}Fixed", f"{f3:.1f}")
        if v != "base":
            b = [100 * np.mean([x["pol"]["table"]["base"] for x in M if x["area"].startswith(cls)]) for cls in CLS]
            gain = max(gain, max(c - bb for c, bb in zip(cells, b)), f3 - 100 * np.mean([x["pol"]["fixed300"]["base"] for x in M]))
    mac("satMaxGain", f"{gain:.1f}")
    mac("satN", len(M))
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_saturation.tex"), "w").write("\n".join(L) + "\n")


def part_C(fname="privacy5_C.json", S="", F=""):
    M = load(fname)
    if not M:
        return
    TH = [("blind", "blind registration", "Blind"), ("x4", "four fused releases", "Four"), ("pf100", "portfolio of 100", "PfH"), ("pf25", "portfolio of 25", "PfQ")]
    L = ["\\begin{tabular}{lcccccc}", "\\toprule", "Assumed attacker & \\multicolumn{3}{c}{Certified (\\%)} & Mean certified scale (m) & Certified and $\\le$ 400\\,m (\\%) & Attacks per mission \\\\",
         " & dense & medium & sparse & & & \\\\", "\\midrule"]
    for th, name, c in TH + ([("blind_any", "blind registration, some site identified (mission level)", "BlindAny")] if "blind_any" in M[0]["cert"] else []):
        cert = [100 * np.mean([x["cert"][th]["certified"] for x in M if x["area"].startswith(cls)]) for cls in CLS]
        sc = [x["cert"][th]["scale"] for x in M if x["cert"][th]["certified"]]
        ok400 = 100 * np.mean([x["cert"][th]["certified"] and x["cert"][th]["scale"] <= 400 for x in M])
        dr = np.mean([x["cert"][th]["draws"] for x in M])
        L.append(f"{name} & " + " & ".join(f"{v:.0f}" for v in cert) + f" & {np.mean(sc) if sc else float('nan'):.0f} & {ok400:.0f} & {dr:.0f} \\\\")
        for cls, v in zip(CLS, cert):
            mac(f"dec{S}{c}Cert{cls.capitalize()}", f"{v:.0f}")
        mac(f"dec{S}{c}Scale", f"{np.mean(sc) if sc else 0:.0f}")
        mac(f"dec{S}{c}CertAll", f"{100 * np.mean([x['cert'][th]['certified'] for x in M]):.0f}")
        mac(f"dec{S}{c}Ok", f"{ok400:.0f}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, f"table_decision{F}.tex"), "w").write("\n".join(L) + "\n")
    # decision for declared threat sets and caps
    sets = [("blind", ["blind"]), ("blind + four releases", ["blind", "x4"]), ("+ portfolio of 100", ["blind", "x4", "pf100"]), ("+ portfolio of 25", ["blind", "x4", "pf100", "pf25"])]
    L = ["\\begin{tabular}{lccccc}", "\\toprule", "Declared threats & \\multicolumn{3}{c}{Released to the planner at a cap of (\\%)} & Mean released scale at 400\\,m (m) & Planned on the edge at 400\\,m (\\%) \\\\",
         " & 200\\,m & 400\\,m & 800\\,m & & \\\\", "\\midrule"]
    for k, (name, ths) in enumerate(sets):
        rel = {}
        for cap in (200, 400, 800):
            rel[cap] = [all(x["cert"][t]["certified"] for t in ths) and max(x["cert"][t]["scale"] for t in ths) <= cap for x in M]
        sc = [max(x["cert"][t]["scale"] for t in ths) for x, r in zip(M, rel[400]) if r]
        scs = f"{np.mean(sc):.0f}" if sc else "--"
        L.append(f"{name} & " + " & ".join(f"{100 * np.mean(rel[c]):.0f}" for c in (200, 400, 800)) + f" & {scs} & {100 * (1 - np.mean(rel[400])):.0f} \\\\")
        for c in (200, 400, 800):
            mac(f"dec{S}Set{KW[k]}Cap{CW[c]}", f"{100 * np.mean(rel[c]):.0f}")
        mac(f"dec{S}Set{KW[k]}Edge", f"{100 * (1 - np.mean(rel[400])):.0f}")
        for cls in CLS:
            h = [i for i, x in enumerate(M) if x["area"].startswith(cls)]
            mac(f"dec{S}Set{KW[k]}{cls.capitalize()}CapFour", f"{100 * np.mean([rel[400][i] for i in h]):.0f}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, f"table_decision_sets{F}.tex"), "w").write("\n".join(L) + "\n")
    # misspecification matrix: design assumption (rows) against the actual attacker (columns), missions certified under the design assumption
    L = ["\\begin{tabular}{lccccc}", "\\toprule", "Scale designed for & mean scale (m) & against blind (\\%) & against four releases (\\%) & against portfolio of 100 (\\%) & against portfolio of 25 (\\%) \\\\", "\\midrule"]
    for th, name, c in TH:
        rows = [x["risk"][th] for x in M if th in x["risk"]]
        if not rows:
            continue
        cells = [100 * np.mean([r[a] for r in rows]) for a, _, _ in TH]
        L.append(f"{name} (certified missions: {len(rows)}) & {np.mean([r['scale'] for r in rows]):.0f} & " + " & ".join(f"{v:.1f}" for v in cells) + " \\\\")
        for (a, _, ac), v in zip(TH, cells):
            mac(f"mis{S}{c}{ac}", f"{v:.1f}")
        mac(f"mis{S}{c}Scale", f"{np.mean([r['scale'] for r in rows]):.0f}")
        mac(f"mis{S}{c}N", len(rows))
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, f"table_misspec{F}.tex"), "w").write("\n".join(L) + "\n")
    mac(f"dec{S}N", len(M))


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
    part_C("privacy5_C_bonf.json", "Bonf", "_bonf")
    open(os.path.join(AUTO, "numbers_p5.tex"), "w").write("\n".join(macros) + "\n")
    print(len(macros), "macros")
