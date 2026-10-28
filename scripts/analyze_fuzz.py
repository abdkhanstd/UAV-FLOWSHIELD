"""Fuzz-test and mutation-test summary -> LaTeX macros and table."""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
F = json.load(open(os.path.join(ROOT, "results", "fuzz.json")))
M = json.load(open(os.path.join(ROOT, "results", "fuzz_mutants.json")))
s = F["stats"]


def num(x):
    return f"{x:,}".replace(",", r"\,")


mac = []
for k, name in (("obs_in", "fuzzObs"), ("obs_admitted", "fuzzAdmitted"), ("plans", "fuzzPlans"), ("accepted", "fuzzAccepted"),
                ("violations", "fuzzViolations")):
    mac.append("\\newcommand{\\" + name + "}{" + num(s[k]) + "}")
mac.append("\\newcommand{\\fuzzTrials}{" + num(F["n"]) + "}")
names = {"intact": "None (intact verifier)", "no spatial envelope": "Spatial envelope", "no loiter cap": "Loiter-duration cap",
         "no altitude bound": "Altitude bounds", "no number redaction": "Number redaction in messages",
         "coordinate targets allowed": "Handle-only targets (literal coordinates allowed)",
         "any channel may add spatial task": "Channel authority for spatial tasks"}
L = [r"\begin{tabular}{lcc}", r"\toprule", r"Safeguard disabled (mutant) & Accepted plans & Invariant violations \\", r"\midrule"]
for m in M:
    L.append(names.get(m["label"], m["label"]) + " & " + str(m["accepted"]) + " & " + str(m["violations"]) + r" \\")
    if m["label"] == "intact":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
det = sum(1 for m in M if m["label"] != "intact" and m["violations"] > 0)
tot = sum(1 for m in M if m["label"] != "intact")
mac.append("\\newcommand{\\mutDetected}{" + str(det) + "}")
mac.append("\\newcommand{\\mutTotal}{" + str(tot) + "}")
mac.append("\\newcommand{\\mutTrials}{4\\,000}")
open(os.path.join(AUTO, "table_mutants.tex"), "w").write("\n".join(L) + "\n")
open(os.path.join(AUTO, "numbers_fuzz.tex"), "w").write("\n".join(mac) + "\n")
print("\n".join(mac))
