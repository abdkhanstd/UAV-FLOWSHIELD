"""Parametric analysis of an operator confirmation gate for spatial requests (not a human study).

The authenticated-ticket attack AD1t is executed because origin authentication is not authorization of intent. A gate that asks the
operator to confirm every admitted spatial request before it enters the plan closes the residual only as far as the operator discriminates.
Using the measured outcomes of UAV-FlowShield on T2 (no gate), the effect of a gate is computed for simulated operators: the operator approves
a legitimate request with probability a_l and an attacker request with probability a_t, and answers after a log-normal delay (median m
seconds, sigma 0.8); a request not answered before the deadline D is dropped. For the profiles below the parameters are assumptions, chosen to span
the plausible range; the paper reports them as a sensitivity analysis.
usage: python scripts/exp_confirm.py   -> ../paper/auto/table_confirm.tex, numbers_confirm.tex"""
import os
import sys

import numpy as np
from scipy.stats import lognorm

sys.path.insert(0, os.path.dirname(__file__))
from analyze_eval import load  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
R = os.path.join(ROOT, "results", "eval")
PROFILES = [("approves everything (habit)", 1.00, 1.00, 15), ("attentive", 0.97, 0.30, 25), ("expert", 0.98, 0.05, 40), ("slow expert", 0.98, 0.05, 90)]
DEADLINES = [30, 60, 120]
DW = {30: "Thirty", 60: "Sixty", 120: "Hundred"}
SIGMA = 0.8


def p_in_time(median, deadline):
    return float(lognorm.cdf(deadline, s=SIGMA, scale=median))


def main():
    df = load([os.path.join(R, "t2_gemma_fs.jsonl")])
    fs = df[df.method == "flowshield"]
    import json
    B = {e["id"]: e for e in json.load(open(os.path.join(ROOT, "data", "bench_t2.json")))["episodes"]}
    ben_all = fs[fs.category.str.startswith("benign")]
    has_req = ben_all.id.map(lambda i: bool(B[i]["expected"].get("dynamic_task")))
    dyn = ben_all[has_req]
    base_ok = float(dyn.success.mean())
    ad1t = fs[(fs.category == "adaptive") & (fs.goal == "AD1t")]
    base_att = float(ad1t.attack_success.mean())
    n_dyn = len(dyn)
    L = ["\\begin{tabular}{lcccccc}", "\\toprule",
         "Operator (approves legitimate / attacker request; median delay) & Deadline (s) & Dynamic requests completed (\\%) & AD1t executed (\\%) & Requests answered in time (\\%) & Added delay of an approved request (s) \\\\",
         "\\midrule"]
    macros = [f"\\newcommand{{\\cfBaseDyn}}{{{100 * base_ok:.1f}}}", f"\\newcommand{{\\cfBaseAdOneT}}{{{100 * base_att:.1f}}}"]
    code = {0: "Habit", 1: "Attentive", 2: "Expert", 3: "Slow"}
    for k, (name, al, at, med) in enumerate(PROFILES):
        for d in DEADLINES:
            pt = p_in_time(med, d)
            comp = base_ok * al * pt
            att = base_att * at * pt
            # mean delay of an approved request that arrived in time: E[T | T < d]
            ts = lognorm.rvs(s=SIGMA, scale=med, size=200000, random_state=0)
            delay = float(ts[ts < d].mean())
            L.append(f"{name if d == DEADLINES[0] else ''} & {d} & {100 * comp:.1f} & {100 * att:.1f} & {100 * pt:.0f} & {delay:.0f} \\\\")
            macros += [f"\\newcommand{{\\cf{code[k]}D{DW[d]}Comp}}{{{100 * comp:.1f}}}", f"\\newcommand{{\\cf{code[k]}D{DW[d]}Att}}{{{100 * att:.1f}}}"]
        L.append("\\midrule" if k < len(PROFILES) - 1 else "")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_confirm.tex"), "w").write("\n".join(l for l in L if l) + "\n")
    # workload: confirmations per mission = requests per mission (one per mission at most): share of benign missions with a spatial request
    share = float(has_req.mean())
    macros.append(f"\\newcommand{{\\cfWorkload}}{{{100 * share:.0f}}}")
    open(os.path.join(AUTO, "numbers_confirm.tex"), "w").write("\n".join(macros) + "\n")
    print("baseline dynamic completion", base_ok, "AD1t", base_att, "n", n_dyn)


if __name__ == "__main__":
    main()
