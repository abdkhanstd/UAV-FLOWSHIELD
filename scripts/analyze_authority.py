"""Tables and macros for the authority-semantics test (results/authority.json).
usage: python scripts/analyze_authority.py   -> ../paper/auto/table_authority_attacks.tex, table_authority_mutants.tex, numbers_authority.tex"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
SCEN = [("replay_1s", "Replay of an accepted ticket after 1 s", "ReplayShort"), ("replay_1h", "Replay of an accepted ticket after 1 h", "ReplayLong"),
        ("revoked", "Ticket revoked before presentation", "Revoked"), ("other_mission", "Ticket presented for another mission", "Other"),
        ("approval_other_action", "Approval of another action presented", "Approval"), ("no_approval", "Spatial request without approval", "NoApproval")]
MUT = [("no_expiry", "no validity window", "NoExpiry"), ("no_ttl_cap", "no cap on the declared lifetime", "NoCap"), ("no_single_use", "no single-use rule", "NoSingle"),
       ("no_revocation", "no revocation check", "NoRevoke"), ("approval_unbound", "approval not bound to the typed action", "Unbound"),
       ("no_mission", "ticket not bound to the mission", "NoMission"), ("early_consume", "nonce consumed by a rejected presentation", "Early")]
TTL = {"30": "Thirty", "60": "Sixty", "120": "OneTwenty", "300": "ThreeHundred", "600": "SixHundred", "900": "NineHundred"}
NC = "\\newcommand"


def thousands(n):
    return f"{n:,}".replace(",", "\\,")


def main():
    d = json.load(open(os.path.join(ROOT, "results", "authority.json")))
    m = [f"{NC}{{\\auSeq}}{{{thousands(d['base']['sequences'])}}}", f"{NC}{{\\auSteps}}{{{thousands(d['base']['steps'])}}}",
         f"{NC}{{\\auBaseUnsafe}}{{{d['base']['unsafe']}}}", f"{NC}{{\\auBaseRefused}}{{{d['base']['refused']}}}",
         f"{NC}{{\\auSkew}}{{{d['params']['skew']:.0f}}}", f"{NC}{{\\auTtlMax}}{{{d['params']['ttl_max']:.0f}}}"]
    L = ["\\begin{tabular}{lcc}", "\\toprule", "Attempt (500 each) & Origin tag only & Authority semantics \\\\", "\\midrule"]
    for k, name, code in SCEN:
        r = d["scenarios"][k]
        a, b = 100 * r["tag_only"] / r["n"], 100 * r["semantics"] / r["n"]
        L.append(f"{name} & {a:.1f}\\,\\% accepted & {b:.1f}\\,\\% accepted \\\\")
        m += [f"{NC}{{\\auTag{code}}}{{{a:.0f}}}", f"{NC}{{\\auSem{code}}}{{{b:.0f}}}"]
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_authority_attacks.tex"), "w").write("\n".join(L) + "\n")
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Safeguard removed & Unsafe accepts & Wrong refusals & Found by the test \\\\", "\\midrule"]
    found = 0
    for k, name, code in MUT:
        r = d["mutants"][k]
        f = r["unsafe"] + r["refused"] > 0
        found += f
        L.append(f"{name} & {r['unsafe']} & {r['refused']} & {'yes' if f else 'no'} \\\\")
        m += [f"{NC}{{\\auMutUnsafe{code}}}{{{r['unsafe']}}}", f"{NC}{{\\auMutRefused{code}}}{{{r['refused']}}}"]
    L += ["\\bottomrule", "\\end{tabular}"]
    m += [f"{NC}{{\\auMutFound}}{{{found}}}", f"{NC}{{\\auMutTotal}}{{{len(MUT)}}}"]
    open(os.path.join(AUTO, "table_authority_mutants.tex"), "w").write("\n".join(L) + "\n")
    for t, code in TTL.items():
        m.append(f"{NC}{{\\auAvail{code}}}{{{100 * d['availability'][t]:.1f}}}")
    open(os.path.join(AUTO, "numbers_authority.tex"), "w").write("\n".join(m) + "\n")
    print("\n".join(m[:8]))


if __name__ == "__main__":
    main()
