"""Table and macros for the runtime-supervision experiment (results/runtime.json).
usage: python scripts/analyze_runtime.py   -> ../paper/auto/table_runtime.tex, numbers_runtime.tex"""
import collections
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
NAME = {"none": "no event", "wind_step": "wind rises", "nfz_grow": "no-fly zone grows", "floor_update": "altitude floor published",
        "gps_degraded": "positioning degraded", "combined": "wind, zone and floor"}
CODE = {"none": "None", "wind_step": "Wind", "nfz_grow": "Zone", "floor_update": "Floor", "gps_degraded": "Gps", "combined": "Comb"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def main():
    res = json.load(open(os.path.join(ROOT, "results", "runtime.json")))
    agg = collections.defaultdict(list)
    for r in res:
        agg[(r["event"], r["sup"])].append(r)
    L = ["\\begin{tabular}{llcccccc}", "\\toprule",
         "Event & Supervisor & Unsafe (\\%) & of which zone & of which floor & of which battery & Violation inside the announcement delay (\\%) & Completed / site dropped (\\%) \\\\",
         "\\midrule"]
    macros = []
    for ev in NAME:
        for sup, sn in (("open_loop", "open loop"), ("reverify", "re-verification")):
            rows = agg[(ev, sup)]
            n = len(rows)
            f = lambda k: 100 * sum(bool(r[k]) for r in rows) / n
            u = sum(r["unsafe"] for r in rows)
            lo, hi = wilson(u, n)
            L.append(f"{NAME[ev] if sup == 'open_loop' else ''} & {sn} & {f('unsafe'):.1f}\\,\\scriptsize[{max(lo, 0):.1f}, {hi:.1f}]\\normalsize & {f('nfz_post'):.1f} & "
                     f"{f('floor_post'):.1f} & {f('depleted'):.1f} & {f('latency_unsafe'):.1f} & {f('completed'):.1f} / {f('degraded'):.1f} \\\\")
            s = "Open" if sup == "open_loop" else "Rev"
            for k, kn in (("unsafe", "Unsafe"), ("completed", "Compl"), ("degraded", "Degr"), ("nfz_post", "Zone"), ("depleted", "Depl"),
                          ("floor_post", "Floor"), ("latency_unsafe", "Late")):
                macros.append(f"\\newcommand{{\\rt{kn}{CODE[ev]}{s}}}{{{f(k):.1f}}}")
            if sup == "open_loop":
                macros.append(f"\\newcommand{{\\rtN{CODE[ev]}}}{{{n}}}")
        if ev != "combined":
            L.append("\\midrule")
    zone_bad = [r for r in agg[("nfz_grow", "reverify")] if r["nfz_post"]]
    share = 100 * sum(bool(r.get("inside_at_ann")) for r in zone_bad) / max(len(zone_bad), 1)
    macros.append(f"\\newcommand{{\\rtInsideShare}}{{{share:.0f}}}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_runtime.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, "numbers_runtime.tex"), "w").write("\n".join(macros) + "\n")
    for ev in NAME:
        print(ev, [(sup, round(100 * np.mean([r["unsafe"] for r in agg[(ev, sup)]]), 1), round(100 * np.mean([r["latency_unsafe"] for r in agg[(ev, sup)]]), 1),
                    round(100 * np.mean([r["completed"] for r in agg[(ev, sup)]]), 1)) for sup in ("open_loop", "reverify")])


if __name__ == "__main__":
    main()
