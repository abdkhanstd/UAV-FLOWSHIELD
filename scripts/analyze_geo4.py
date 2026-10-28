"""Table and macros for the third geography (results/geo4.json, scripts/exp_geo4.py).
usage: python scripts/analyze_geo4.py -> ../paper/auto/table_geo4.tex, numbers_geo4.tex"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
NC = "\\newcommand"
CLS = [("dense", "Dense", "Dense"), ("medium", "Medium", "Medium"), ("sparse", "Sparse", "Sparse"), ("all", "All", "All")]
POL = [("ladder", "Certified ladder", "Ladder"), ("tabfirst", "Table first", "Tab"), ("table", "Public table scale, no test", "Table")]


def main():
    d = json.load(open(os.path.join(ROOT, "results", "geo4.json")))
    cal = {c["id"]: c for c in d["calibration"]}
    held = {h["id"]: h for h in d["heldout"]}
    L = ["\\begin{tabular}{llccccccc}", "\\toprule",
         "Class (missions) & Policy & Mean scale (m) & Sites identified (\\%) & Some site identified (\\%) & Missions above 20\\,\\% & No claim (\\%) & Attacks & At table scale (\\%) \\\\", "\\midrule"]
    m = [f"{NC}{{\\gfN}}{{{len(cal)}}}", f"{NC}{{\\gfT}}{{{d['T']}}}"]
    areas = {c["area"] for c in cal.values()}
    m.append(f"{NC}{{\\gfAreas}}{{{len(areas)}}}")
    for cls, cname, ccode in CLS:
        ids = [i for i, c in cal.items() if cls == "all" or c["cls"] == cls]
        for pol, pname, pcode in POL:
            sc = np.mean([held[i]["pol"][pol]["scale"] for i in ids])
            site = 100 * np.mean([held[i]["pol"][pol]["site"] for i in ids])
            over = 100 * np.mean([held[i]["pol"][pol]["site"] > 0.20 for i in ids])
            anyid = 100 * np.mean([held[i]["pol"][pol]["any"] for i in ids])
            rule = {"ladder": "cp_fs", "tabfirst": "cp_tab"}.get(pol)
            unc = f"{100 * np.mean([not cal[i][rule]['passed'] for i in ids]):.1f}" if rule else "--"
            att = f"{np.mean([cal[i][rule]['draws'] for i in ids]):.0f}" if rule else "0"
            tab = f"{100 * np.mean([cal[i]['cp_tab']['table'] for i in ids]):.0f}" if pol == "tabfirst" else ("100" if pol == "table" else "--")
            L.append(f"{cname + ' (' + str(len(ids)) + ')' if pol == 'ladder' else ''} & {pname} & {sc:.0f} & {site:.1f} & {anyid:.1f} & {over:.1f}\\,\\% & {unc} & {att} & {tab} \\\\")
            m += [f"{NC}{{\\gf{pcode}Scale{ccode}}}{{{sc:.0f}}}", f"{NC}{{\\gf{pcode}Site{ccode}}}{{{site:.1f}}}", f"{NC}{{\\gf{pcode}Over{ccode}}}{{{over:.1f}}}", f"{NC}{{\\gf{pcode}Any{ccode}}}{{{anyid:.1f}}}"]
            if rule:
                m += [f"{NC}{{\\gf{pcode}Unc{ccode}}}{{{unc}}}", f"{NC}{{\\gf{pcode}Draws{ccode}}}{{{att}}}"]
            if pol == "tabfirst":
                m.append(f"{NC}{{\\gf{pcode}Share{ccode}}}{{{tab}}}")
        if cls != "all":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_geo4.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, "numbers_geo4.tex"), "w").write("\n".join(m) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
