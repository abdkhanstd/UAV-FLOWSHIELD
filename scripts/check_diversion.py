"""Energy that an admitted spatial request added to the executed plans of the recorded T2 runs (cache replay: every model
response is taken from the cache, no model is called).

For every executed plan that contains a loiter at an admitted request it reports
  loiter_cost  executed energy minus the energy of the same route with the loiter removed (what the request added), and
  vs_minimum   executed energy minus the minimum-energy mission over the same inspected sites without the request
               (shield.min_energy_plan); this also contains the suboptimality of the planner's route,
and whether the loiter was flown right after the inspection of its qualifying site (the canonical insertion that
Proposition 4 of the paper bounds by 2 rho + 8 T <= 2560 metre-equivalents).

usage: python scripts/check_diversion.py  -> results/eval/diversion_check.json and a summary on stdout"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = [("t2_gemma_fs", {"flowshield", "fs_fallback"}), ("t2_gemma_final", None)]
G = {}


def init():
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import flowshield.llm as LL
    import flowshield.pipeline as PL
    import flowshield.shield as SH
    import flowshield.pipeline_final as PF
    from flowshield.data import load_bench
    from flowshield.executor import execute

    def chat(self, system, user, tag=""):
        k = self.key(system, user)
        if k in self.cache:
            return self.cache[k]
        raise KeyError(tag)
    LL.LLM.chat = chat
    verify = SH.verify_plan

    def checked(ep, plan, obs, allow_points=False, typed=True):
        out = verify(ep, plan, obs, allow_points, typed)
        if out[0]:
            G["obs"] = obs
        return out
    SH.verify_plan = checked

    def ex(ep, acts):
        log = execute(ep, acts)
        G["acts"], G["log"] = acts, log
        return log
    PL.execute = ex
    G.update(PF=PF, SH=SH, LL=LL, execute=execute, llms={},
             eps={e["id"]: e for e in load_bench(os.path.join(ROOT, "data", "bench_t2.json"))["episodes"]})


def job(r):
    G["obs"] = G["acts"] = G["log"] = None
    ep = G["eps"][r["id"]]
    key = (r.get("model", "gemma4:12b"), r.get("temperature", 0.0), r.get("seed", 0))
    if key not in G["llms"]:
        name = f"cache_{key[0].replace(':', '-')}_t{key[1]}_s{key[2]}.jsonl"
        G["llms"][key] = G["LL"].LLM(key[0], os.path.join(ROOT, "results", name), temperature=key[1], seed=key[2])
    try:
        out = G["PF"].run_episode(ep, r["method"], G["llms"][key], scale=r.get("scale"), opts=r.get("opts") or {})
    except KeyError:
        return None
    if not out["executed"] or not G["log"] or not G["log"]["loiters"] or G["obs"] is None:
        return None
    SH, acts, obs = G["SH"], G["acts"], G["obs"]
    names = [s["name"] for s in ep["sites"]]
    insp = {a["name"] for a in acts if a["op"] == "inspect"}
    sub = [i + 1 for i, n in enumerate(names) if n in insp]
    base, _ = SH.resolve_shield(ep, SH.min_energy_plan(ep, obs, sub, request=False), obs)
    seq = [a for a in acts if a["op"] in ("inspect", "loiter")]
    site_of = {o["handle"]: o["site_name"] for o in obs if o["kind"] == "inspection_request"}
    after = all(i > 0 and seq[i - 1]["op"] == "inspect" and seq[i - 1]["name"] == site_of.get(a.get("obs"))
                for i, a in enumerate(seq) if a["op"] == "loiter")
    e = G["log"]["energy"]
    return {"id": r["id"], "method": r["method"], "category": r["category"], "after_site": after,
            "loiter_cost": e - G["execute"](ep, [a for a in acts if a["op"] != "loiter"])["energy"],
            "vs_minimum": e - G["execute"](ep, base)["energy"]}


def main():
    import multiprocessing as mp
    summary = {}
    for name, methods in RUNS:
        rows = [json.loads(line) for line in open(os.path.join(ROOT, "results", "eval", name + ".jsonl"), encoding="utf-8")]
        rows = [r for r in rows if methods is None or r["method"] in methods]
        with mp.get_context("spawn").Pool(min(12, os.cpu_count() or 2), initializer=init) as pool:
            outs = [o for o in pool.map(job, rows, chunksize=8) if o]
        summary[name] = {"plans_with_loiter": len(outs), "after_site": sum(o["after_site"] for o in outs),
                         "attack_episodes": sum(not o["category"].startswith("benign") for o in outs),
                         "max_loiter_cost": round(max(o["loiter_cost"] for o in outs), 1),
                         "loiter_cost_above_2560": sum(o["loiter_cost"] > 2560 for o in outs),
                         "max_vs_minimum": round(max(o["vs_minimum"] for o in outs), 1)}
        print(name, summary[name], flush=True)
    json.dump(summary, open(os.path.join(ROOT, "results", "eval", "diversion_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
