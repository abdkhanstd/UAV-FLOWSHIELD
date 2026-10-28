"""Test of the authority semantics (flowshield/authority.py), independent of any LLM.

(1) Property test against an independent oracle: random event sequences (issue, deliver, replay, revoke, approve, tamper) are
    pushed through `authority.check`; the decision must equal the oracle's at every step (safety: nothing the oracle refuses is
    accepted; liveness: nothing the oracle accepts is refused).
(2) Test of the test: each safeguard is removed in turn (mutants) and the property test must find a difference.
(3) Attack scenarios and availability: replay, late replay, revoked ticket, ticket moved to another mission, approval of
    another action, against (a) the earlier design (origin tag only) and (b) the authority semantics; and the share of
    legitimate tickets accepted as a function of the declared lifetime, with log-normal delivery delay.

usage: python scripts/exp_authority.py --out results/authority.json"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield import auth as OLD
from flowshield import authority as A

ROOT = os.path.join(os.path.dirname(__file__), "..")
MUTANTS = ["no_expiry", "no_ttl_cap", "no_single_use", "no_revocation", "approval_unbound", "no_mission", "early_consume"]
FIELDS = ["text", "channel", "mission", "client", "id", "iat", "exp"]


class Oracle:
    """Reference semantics, written separately from the implementation."""

    def __init__(self):
        self.done, self.rev_id, self.cut = set(), set(), {}

    def decide(self, t, mission, now, spatial, action, approval, genuine_tag, approved_for):
        if not genuine_tag:
            return False
        if t["mission"] != mission or t["channel"] != "portal":
            return False
        if not (t["iat"] - A.SKEW <= now < t["exp"]) or t["exp"] - t["iat"] > A.TTL_MAX:
            return False
        if t["id"] in self.rev_id or t["iat"] <= self.cut.get(t["client"], -1e18):
            return False
        if t["id"] in self.done:
            return False
        if spatial and not (approval is not None and approved_for == json.dumps(action, sort_keys=True)):
            return False
        self.done.add(t["id"])
        return True


def run_sequences(n_seq, seed, mutants=()):
    rng = np.random.default_rng(seed)
    diffs = {"unsafe": 0, "refused": 0}
    steps = 0
    for q in range(n_seq):
        st, orc = A.AuthorityState(), Oracle()
        tickets, mission = [], f"M{q}"
        t_now = 1000.0
        for step in range(int(rng.integers(6, 20))):
            t_now += float(rng.exponential(60.0))
            ev = rng.choice(["issue", "deliver", "deliver", "replay", "revoke_t", "revoke_c", "tamper", "other_mission"])
            if ev == "issue" or not tickets:
                ttl = float(rng.choice([30, 300, 900, 1500]))
                c = f"C{int(rng.integers(3))}"
                t = A.issue(c, mission, "portal", f"task {step}", t_now - float(rng.uniform(0, 100)), ttl, nonce=f"{q}-{step}")
                act = {"kind": "inspection_request", "site": f"S{int(rng.integers(1, 6))}", "xy": [int(rng.integers(0, 9999)), int(rng.integers(0, 9999))]}
                tickets.append((t, act, bool(rng.random() < 0.5)))
                continue
            if ev == "revoke_t":
                tid = tickets[int(rng.integers(len(tickets)))][0]["id"]
                st.revoke_ticket(tid)
                orc.rev_id.add(tid)
                continue
            if ev == "revoke_c":
                c = f"C{int(rng.integers(3))}"
                st.revoke_client(c, t_now - float(rng.uniform(0, 400)))
                orc.cut[c] = max(orc.cut.get(c, -1e18), st.client_cutoff[c])
                continue
            t, act, spatial = tickets[int(rng.integers(len(tickets)))]
            presented, genuine, mis = dict(t), True, mission
            if ev == "tamper":
                f = str(rng.choice(FIELDS))
                v = presented[f]
                presented[f] = (v + 1.0) if isinstance(v, float) else (str(v) + "x")
                genuine = False
            if ev == "other_mission":
                mis = mission + "x"
            appr, appr_for = None, None
            if spatial:
                kind = rng.random()
                if kind < 0.55:
                    appr, appr_for = A.approve(t["id"], act), json.dumps(act, sort_keys=True)
                elif kind < 0.8:
                    other = dict(act, site="S9")
                    appr, appr_for = A.approve(t["id"], other), json.dumps(other, sort_keys=True)
            want = orc.decide(presented, mis, t_now, spatial, act, appr, genuine, appr_for)
            got, _ = A.check(st, presented, mis, t_now, spatial, act, appr, mutants=mutants)
            steps += 1
            if got and not want:
                diffs["unsafe"] += 1
            elif want and not got:
                diffs["refused"] += 1
    return {"sequences": n_seq, "steps": steps, **diffs}


def scenarios(n, seed):
    rng = np.random.default_rng(seed)
    cases = ["replay_1s", "replay_1h", "revoked", "other_mission", "approval_other_action", "no_approval"]
    res = {c: {"tag_only": 0, "semantics": 0} for c in cases}
    for i in range(n):
        t0 = 5000.0
        act = {"kind": "inspection_request", "site": "S3", "xy": [1200, 3400]}
        for c in cases:
            tk = A.issue("C1", "M1", "portal", "inspect the road", t0, 300.0, nonce=f"s{i}{c}")
            st = A.AuthorityState()
            ep_old = {"id": "M2" if c == "other_mission" else "M1", "operator": {"client": "C1"}}
            doc_old = {"channel": "portal", "text": "inspect the road", "sig": OLD.sign("C1", "M1", "portal", "inspect the road")}
            tag_only = OLD.verify(ep_old, doc_old)       # earlier design: origin tag only (binds mission, channel and text; no clock, nonce, revocation or approval)
            appr = A.approve(tk["id"], act)
            if c in ("replay_1s", "replay_1h"):
                A.check(st, tk, "M1", t0 + 1, True, act, appr)
                ok, _ = A.check(st, tk, "M1", t0 + (1 if c == "replay_1s" else 3600) + float(rng.uniform(0, 5)), True, act, appr)
            elif c == "revoked":
                st.revoke_ticket(tk["id"])
                ok, _ = A.check(st, tk, "M1", t0 + 5, True, act, appr)
            elif c == "other_mission":
                ok, _ = A.check(st, tk, "M2", t0 + 5, True, act, appr)
            elif c == "approval_other_action":
                ok, _ = A.check(st, tk, "M1", t0 + 5, True, dict(act, xy=[9000, 100]), appr)
            else:
                ok, _ = A.check(st, tk, "M1", t0 + 5, True, act, None)
            res[c]["tag_only"] += int(tag_only)
            res[c]["semantics"] += int(ok)
    for c in cases:
        res[c]["n"] = n
    return res


def availability(n, seed):
    """Legitimate tickets with log-normal delivery delay (median 20 s, sigma 1.0) and an edge clock offset (+-3 s)."""
    rng = np.random.default_rng(seed)
    out = {}
    for ttl in (30, 60, 120, 300, 600, 900):
        ok = 0
        for i in range(n):
            tk = A.issue("C1", "M1", "portal", "x", 1000.0, float(ttl), nonce=f"a{ttl}-{i}")
            delay = float(rng.lognormal(np.log(20.0), 1.0))
            now = 1000.0 + delay + float(rng.uniform(-3, 3))
            ok += A.check(A.AuthorityState(), tk, "M1", now)[0]
        out[str(ttl)] = ok / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=4000)
    ap.add_argument("--out", default="results/authority.json")
    a = ap.parse_args()
    out = {"base": run_sequences(a.n_seq, 11)}
    out["mutants"] = {m: run_sequences(a.n_seq, 11, (m,)) for m in MUTANTS}
    out["scenarios"] = scenarios(500, 5)
    out["availability"] = availability(5000, 9)
    out["params"] = {"skew": A.SKEW, "ttl_max": A.TTL_MAX}
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
