"""Table and macros for the end-to-end run of the final configuration (results/eval/t2_gemma_final.jsonl), next to the frozen T2
configuration and the post-freeze run at the certified scales.
usage: python scripts/analyze_final.py -> ../paper/auto/table_final.tex, numbers_final.tex"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from stats_util import wilson  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
RES = os.path.join(ROOT, "results", "eval")
NC = "\\newcommand"
ROWS = [("t2_gemma_fs.jsonl", "flowshield", "Frozen T2 (Wilson scales, tag, abort)", "Frozen", "ucb10"),
        ("t2_gemma_fs.jsonl", "fs_fallback", "Frozen T2 + fallback", "FrozenFb", "ucb10"),
        ("t2_gemma_cert.jsonl", "flowshield", "Certified scales, tag (post-freeze)", "Cert", "cert"),
        ("t2_gemma_final.jsonl", "fs_final", "Final, attentive operator", "Fin", "final"),
        ("t2_gemma_final.jsonl", "fs_final_expert", "Final, expert operator", "FinExp", "final"),
        ("t2_gemma_final.jsonl", "fs_final_habit", "Final, operator approves all", "FinHab", "final")]


def risk_tables():
    old = {h["id"]: h["pol"]["ucb10"]["site"] for h in json.load(open(os.path.join(ROOT, "results", "privacy3_t2_heldout.json")))["heldout"]}
    cert = {h["id"]: h["pol"]["cert100"]["site"] for h in json.load(open(os.path.join(ROOT, "results", "privacy3_cert100_t2.json")))["heldout"]}
    return {"ucb10": old, "cert": cert, "final": cert}


def main():
    risk = risk_tables()
    L = ["\\begin{tabular}{lccccccccc}", "\\toprule",
         "Configuration & Inj. & Adapt. exec. & AD1t & AD6b & AD8 lost & Benign (\\%) & Dyn. & Cloud & Site id. \\\\",
         "\\midrule"]
    m = []
    for f, meth, name, code, rk in ROWS:
        rows = [json.loads(l) for l in open(os.path.join(RES, f))]
        r = [x for x in rows if x["method"] == meth]
        inj = [x for x in r if x["category"] == "attack"]
        # integrity attacks: AD1-AD8 without AD4 (private view only) and AD6b (availability), as in the main T2 table
        ad = [x for x in r if x["category"] == "adaptive" and x["goal"] not in ("AD4", "AD6b")]
        ad6b = [x for x in r if x["category"] == "adaptive" and x["goal"] == "AD6b"]
        k_6b = sum(x["attack_success"] for x in ad6b)      # AD6b success: the legitimate tight-budget mission does not complete
        ad1t = [x for x in ad if x["goal"] == "AD1t"]
        ben = [x for x in r if x["category"].startswith("benign")]
        dyn = [x for x in r if x["category"] == "benign_dynamic"]
        k_inj = sum(x["attack_success"] for x in inj)
        _, _, hi = wilson(k_inj, len(inj))
        k_ad = sum(x["attack_success"] for x in ad)
        k_8 = sum(x["attack_success"] for x in ad if x["goal"] == "AD8")   # AD8: the unsigned attacker request is never admitted; a success is a lost legitimate request
        k_ex = k_ad - k_8
        k_1t = sum(x["attack_success"] for x in ad1t)
        p_b, lo_b, hi_b = wilson(sum(x["success"] for x in ben), len(ben))
        k_dyn = sum(x["success"] for x in dyn)
        cloud = [x for x in r if x.get("fin_released", True)]
        share = 100 * len(cloud) / len(r)
        ident = 100 * np.mean([risk[rk][x["id"]] if x.get("fin_released", True) else 0.0 for x in r])
        ident_rel = 100 * np.mean([risk[rk][x["id"]] for x in cloud])
        if meth == "fs_final":
            L.append("\\midrule")
        L.append(f"{name} & {k_inj}/{len(inj)} & {k_ex}/{len(ad)} & {k_1t}/{len(ad1t)} & {k_6b}/{len(ad6b)} & {k_8}/10 & {100 * p_b:.1f}\\,\\scriptsize[{100 * lo_b:.1f}, {100 * hi_b:.1f}]\\normalsize & "
                 f"{k_dyn}/{len(dyn)} & {share:.0f}\\,\\% & {ident:.1f}\\,\\% \\\\")
        m += [f"{NC}{{\\fin{code}AsrHi}}{{{100 * hi:.1f}}}", f"{NC}{{\\fin{code}Adapt}}{{{k_ad}}}", f"{NC}{{\\fin{code}AdaptExec}}{{{k_ex}}}", f"{NC}{{\\fin{code}AdOneT}}{{{k_1t}}}", f"{NC}{{\\fin{code}AdSixB}}{{{k_6b}}}", f"{NC}{{\\fin{code}AdaptN}}{{{len(ad)}}}",
              f"{NC}{{\\fin{code}Mcr}}{{{100 * p_b:.1f}}}", f"{NC}{{\\fin{code}Dyn}}{{{k_dyn}}}", f"{NC}{{\\fin{code}Cloud}}{{{share:.1f}}}",
              f"{NC}{{\\fin{code}Ident}}{{{ident:.1f}}}", f"{NC}{{\\fin{code}IdentRel}}{{{ident_rel:.1f}}}",
              f"{NC}{{\\fin{code}AdEight}}{{{sum(x['attack_success'] for x in ad if x['goal'] == 'AD8')}}}"]
        if meth.startswith("fs_final"):
            edge = [x for x in ben if not x["fin_released"]]
            m += [f"{NC}{{\\fin{code}EdgeMcr}}{{{100 * np.mean([x['success'] for x in edge]):.1f}}}", f"{NC}{{\\fin{code}EdgeN}}{{{len(edge)}}}",
                  f"{NC}{{\\fin{code}CloudMcr}}{{{100 * np.mean([x['success'] for x in ben if x['fin_released']]):.1f}}}"]
            req = [x for x in r if x.get("fin_spatial") and x["category"] == "benign_dynamic"]
            for why, wc in (("approved", "Approved"), ("deadline", "Late"), ("no_approval", "Refused")):
                m.append(f"{NC}{{\\fin{code}Req{wc}}}{{{sum(x.get('fin_ticket') == why for x in req)}}}")
            delays = [x["fin_op_delay"] for x in req if x.get("fin_ticket") == "approved"]
            m.append(f"{NC}{{\\fin{code}Delay}}{{{np.median(delays):.0f}}}")
            m.append(f"{NC}{{\\fin{code}Leak}}{{{sum(x['leak_abs_prompt'] for x in r)}}}")
            m.append(f"{NC}{{\\fin{code}Fallback}}{{{sum(x['decision'] == 'FALLBACK' for x in r)}}}")
    L += ["\\bottomrule", "\\end{tabular}"]
    det = [json.loads(l) for l in open(os.path.join(RES, "t2_gemma_base.jsonl"))]
    m.append(f"{NC}{{\\finSolverMs}}{{{1000 * np.median([x['wall'] for x in det if x['method'] == 'det_plus']):.0f}}}")
    open(os.path.join(AUTO, "table_final.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, "numbers_final.tex"), "w").write("\n".join(m) + "\n")
    print("\n".join(L))
    print("\n".join(x for x in m if "Fin" in x and ("Req" in x or "Edge" in x or "Delay" in x or "Cloud" in x)))


if __name__ == "__main__":
    main()
