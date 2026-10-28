"""Macros and tables for the trusted-component tests (fuzz, mutants, diversion bound) and the flight-dynamics check."""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
RES = os.path.join(ROOT, "results")
macros = []


def mac(n, v):
    macros.append(f"\\newcommand{{\\{n}}}{{{v}}}")


def fuzz():
    p = os.path.join(RES, "fuzz.json")
    if os.path.exists(p):
        F = json.load(open(p))
        s = F["stats"]
        mac("fuzzTrials", f"{F['n']:,}".replace(",", "\\,"))
        mac("fuzzViolations", s["violations"])
        mac("fuzzObsIn", f"{s['obs_in']:,}".replace(",", "\\,"))
        mac("fuzzObsAdmitted", f"{s['obs_admitted']:,}".replace(",", "\\,"))
        mac("fuzzSpatialAdmitted", f"{s['spatial_admitted']:,}".replace(",", "\\,"))
        mac("fuzzPlans", f"{s['plans']:,}".replace(",", "\\,"))
        mac("fuzzAccepted", f"{s['accepted']:,}".replace(",", "\\,"))
    p = os.path.join(RES, "fuzz_mutants.json")
    if os.path.exists(p):
        M = json.load(open(p))
        L = ["\\begin{tabular}{lcc}", "\\toprule", "Safeguard disabled (mutant) & Accepted plans & Invariant violations \\\\", "\\midrule"]
        for m in M:
            name = "None (intact verifier)" if m["label"] == "intact" else m["label"][0].upper() + m["label"][1:]
            L.append(f"{name} & {m['accepted']} & {m['violations']} \\\\")
            if m["label"] == "intact":
                L.append("\\midrule")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(AUTO, "table_safeguards.tex"), "w").write("\n".join(L) + "\n")
        mut = [m for m in M if m["label"] != "intact"]
        mac("mutTotal", len(mut))
        mac("mutDetected", sum(m["violations"] > 0 for m in mut))
        mac("mutTrials", M[0]["accepted"] and 4000)


def bound():
    p = os.path.join(RES, "eval", "bound.json")
    if not os.path.exists(p):
        return
    B = json.load(open(p))
    mac("boundValue", f"{B['bound']:.0f}")
    for lab, tag in (("ordinary missions", "Ord"), ("stress missions", "Str")):
        for metric, mt in (("route", "Route"), ("euclid", "Eucl")):
            r = B[lab][metric]
            mac(f"bound{tag}{mt}Max", f"{r['max']:.0f}")
            mac(f"bound{tag}{mt}N", f"{r['n']:,}".replace(",", "\\,"))
            mac(f"bound{tag}{mt}Over", f"{100 * r['share_over_bound']:.2f}")


def dynamics():
    p = os.path.join(RES, "eval", "dynamics.json")
    if not os.path.exists(p):
        return
    D = json.load(open(p))
    W, G = D["winds"], D["gps"]
    miss = [m for m in D["missions"] if "nominal" in m and f"{W[0]}_{G[0]}" in m["nominal"]]
    L = ["\\begin{tabular}{lcccccc}", "\\toprule", "Mean wind (m/s) & \\multicolumn{3}{c}{Budget exceeded (\\%)} & \\multicolumn{3}{c}{No-fly zone entered (\\%)} \\\\",
         "GPS noise $\\sigma$ (m) & " + " & ".join(f"{g:.0f}" for g in G) + " & " + " & ".join(f"{g:.0f}" for g in G) + " \\\\", "\\midrule"]
    for w in W:
        ex = [100 * np.mean([m["nominal"][f"{w}_{g}"]["exceed"] for m in miss]) for g in G]
        it = [100 * np.mean([m["nominal"][f"{w}_{g}"]["intrude"] for m in miss]) for g in G]
        ratio = np.mean([m["nominal"][f"{w}_{G[0]}"]["ratio"] for m in miss])
        L.append(f"{w} & " + " & ".join(f"{x:.1f}" for x in ex) + " & " + " & ".join(f"{x:.1f}" for x in it) + " \\\\")
        W_ = {0: "Zero", 2: "Two", 4: "Four", 6: "Six", 8: "Eight"}[w]
        for g, e_, i_ in zip(G, ex, it):
            G_ = {0: "Zero", 3: "Three", 10: "Ten"}[int(g)]
            mac(f"dynExceedW{W_}G{G_}", f"{e_:.1f}")
            mac(f"dynIntrudeW{W_}G{G_}", f"{i_:.1f}")
        mac(f"dynEnergyRatioW{W_}", f"{ratio:.2f}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_dynamics.tex"), "w").write("\n".join(L) + "\n")
    L = ["\\begin{tabular}{lcccc}", "\\toprule", "Forecast wind (m/s) & Plans found (\\%) & Site dropped (\\%) & Mission completed (\\%) & Unsafe at $\\sigma$=3\\,m (\\%) \\\\", "\\midrule"]
    for w in W[1:]:
        recs = [m["aware"].get(str(w), {}) for m in miss]
        found = [r.get("plan", False) for r in recs]
        n = len(recs)
        ok = [r.get("success", False) for r in recs if r.get("plan")]
        dropped = [r.get("dropped", False) for r in recs if r.get("plan")]
        unsafe = [r.get("unsafe_3.0", 0.0) for r in recs if r.get("plan")]
        L.append(f"{w} & {100 * np.mean(found):.1f} & {100 * np.mean(dropped) if dropped else 0:.1f} & {100 * sum(ok) / n:.1f} & {100 * np.mean(unsafe) if unsafe else 0:.1f} \\\\")
        W_ = {2: "Two", 4: "Four", 6: "Six", 8: "Eight"}[w]
        mac(f"awareFoundW{W_}", f"{100 * np.mean(found):.1f}")
        mac(f"awareCompletedW{W_}", f"{100 * sum(ok) / n:.1f}")
        mac(f"awareUnsafeW{W_}", f"{100 * np.mean(unsafe) if unsafe else 0:.1f}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_aware.tex"), "w").write("\n".join(L) + "\n")
    mac("dynMissions", len(miss))


def calib_time():
    p = os.path.join(RES, "eval", "calib_time.json")
    if not os.path.exists(p):
        return
    T = json.load(open(p))
    for rule, nm in (("mean10", "Mean"), ("ucb10", "Ucb")):
        for cls in ("dense", "medium", "sparse"):
            mac(f"calTime{nm}{cls.capitalize()}Med", f"{T[rule][cls]['median']:.0f}")
            mac(f"calTime{nm}{cls.capitalize()}Tail", f"{T[rule][cls]['p95']:.0f}")
    pc = os.path.join(RES, "eval", "calib_time_cert.json")
    if os.path.exists(pc):
        C = json.load(open(pc))
        for tt, nm in (("60", "CertSixty"), ("100", "CertHundred")):
            for cls in ("dense", "medium", "sparse"):
                mac(f"calTime{nm}{cls.capitalize()}Med", f"{C[tt][cls]['median']:.0f}")
                mac(f"calTime{nm}{cls.capitalize()}Tail", f"{C[tt][cls]['p95']:.0f}")


if __name__ == "__main__":
    calib_time()
    fuzz()
    bound()
    dynamics()
    open(os.path.join(AUTO, "numbers_misc.tex"), "w").write("\n".join(macros) + "\n")
    print(len(macros), "macros")
