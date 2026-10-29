"""Replay recorded runs from the cached language-model responses and compare every replayed episode with the logged one.

No model is called: a prompt that is not in the cache is reported as a miss. Runs made on the GPU server read the server's
caches first (results/cache_server/, see SERVER_RUNS), the others read results/. A small share of episodes does not
reproduce exactly where two identical prompts were sent concurrently and the server returned different responses (the
cache keeps the last one).

At every call of the verifier the script also evaluates the completeness rule in the form the runs applied it (energy of
the canonical plan, i.e. minimum-distance order with the admitted loiter after its qualifying site and a return to the
default altitude after each floor, with the request counted as part of the full mission and every low-priority site
skippable) and counts the calls at which it decides differently from the verifier in flowshield/shield.py (exact
minimum-energy mission over the sites, unique low-priority site).

usage: python scripts/check_replay.py                 every T2 run -> results/eval/replay_check.json
       python scripts/check_replay.py t1                  every T1 run -> results/eval/replay_check_t1.json
       python scripts/check_replay.py <run.jsonl> ...     the given runs (benchmark chosen by file name)"""
import glob
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERVER_RUNS = {"t2_qwen_base", "t2_qwen_fs", "t2_granite4.1_3b", "t2_nemotron-3-nano_4b", "t2_qwen3_1.7b",
               "t2_samp_s0", "t2_samp_s1", "t2_samp_s2", "t2_samp_s3",
               "trade_gemma4-12b", "samp_gemma4-12b_s1", "samp_gemma4-12b_s2", "samp_gemma4-12b_s3", "para_gemma4-12b"}
CMP = ["executed", "decision", "success", "fail_reason", "attack_success", "n_calls", "repaired", "fallback_used",
       "obs_admitted", "obs_rejected", "clamped", "leak_abs_prompt", "leak_out", "attempted", "cloud_calls", "edge_calls"]
G = {}


def init(bench_path, server):
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import flowshield.llm as LL
    import flowshield.shield as SH
    import flowshield.pipeline_cb  # noqa: F401  registers the common-boundary methods
    import flowshield.pipeline_final as PF
    from flowshield.data import load_bench
    from flowshield.executor import execute

    class Miss(Exception):
        pass

    def chat(self, system, user, tag=""):
        k = self.key(system, user)
        if k in self.cache:
            return self.cache[k]
        raise Miss(tag)
    LL.LLM.chat = chat

    verify = SH.verify_plan

    def checked(ep, plan, obs, allow_points=False, typed=True):
        out = verify(ep, plan, obs, allow_points, typed)
        ok, acts, errors, _ = out
        names = [s["name"] for s in ep["sites"]]
        inspected = {a["name"] for a in acts if a["op"] == "inspect"}
        missing = [n for n in names if n not in inspected]
        other = [e for e in errors if not (e.startswith("sites ") and "are missing" in e)]
        old_ok = not other
        if missing:
            low = [s["name"] for s in ep["sites"] if s["priority"] == "low"]
            canon, _ = SH.resolve_shield(ep, SH.canonical_plan(ep, obs, list(range(1, len(names) + 1))), obs)
            old_ok = old_ok and missing == low and execute(ep, canon)["energy"] > ep["budget_m"] + 1e-6
        G["calls"].append({"missing": bool(missing), "differs": bool(old_ok) != bool(ok)})
        return out
    SH.verify_plan = checked

    G.update(PF=PF, LL=LL, Miss=Miss, server=server, llms={},
             eps={e["id"]: e for e in load_bench(os.path.join(ROOT, bench_path))["episodes"]})


def llm_for(model, temp, seed):
    key = (model, temp, seed)
    if key not in G["llms"]:
        name = f"cache_{model.replace(':', '-').replace('/', '-')}_t{temp}_s{seed}.jsonl"
        L = G["LL"].LLM(model, os.path.join(ROOT, "results", name), temperature=temp, seed=seed)
        srv = os.path.join(ROOT, "results", "cache_server", name)
        if G["server"] and os.path.exists(srv):
            for line in open(srv, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    L.cache[r["key"]] = r
                except ValueError:
                    pass
        G["llms"][key] = L
    return G["llms"][key]


def job(r):
    G["calls"] = []
    t, s = r.get("temperature", 0.0), r.get("seed", 0)
    llm = llm_for(r.get("model", "gemma4:12b"), t, s)
    edge = llm_for(r["edge_model"], t, s) if r.get("edge_model") else None
    try:
        out = G["PF"].run_episode(G["eps"][r["id"]], r["method"], llm, scale=r.get("scale"), opts=r.get("opts") or {}, llm_edge=edge)
    except G["Miss"]:
        return {"miss": True, "calls": G["calls"]}
    same = all(out.get(k) == r.get(k) for k in CMP + [k for k in r if k.startswith("fin_") and k not in ("fin_delivery", "fin_op_delay")]
               if k in r)
    e0, e1 = r.get("energy"), out.get("energy")
    same = same and ((e0 is None) == (e1 is None)) and (e0 is None or abs(e0 - e1) <= 1e-6 * max(1.0, abs(e0)))
    return {"miss": False, "same": same, "calls": G["calls"]}


T1_RUNS = ["eval/t1_gemma_base", "eval/t1_gemma_fs_mean05", "eval/t1_gemma_fs_mean10", "eval/t1_gemma_fs_ucb", "eval/t1_qwen_base",
           "eval/t1_qwen_fs_mean05", "eval/t1_qwen_fs_mean10", "eval/t1_qwen_fs_ucb", "main_gemma4-12b", "main_qwen3-8b",
           "abl_gemma4-12b", "abl_qwen3-8b", "trade_gemma4-12b", "samp_gemma4-12b_s1", "samp_gemma4-12b_s2", "samp_gemma4-12b_s3",
           "para_gemma4-12b"]


def bench_of(name):
    if name.startswith("t2_"):
        return "data/bench_t2.json"
    return "data/bench_para.json" if name.startswith("para_") else "data/bench.json"


def main():
    import multiprocessing as mp
    if sys.argv[1:] == ["t1"]:
        files, out = [os.path.join(ROOT, "results", f + ".jsonl") for f in T1_RUNS], "replay_check_t1.json"
    elif sys.argv[1:]:
        files, out = sys.argv[1:], "replay_check_files.json"
    else:
        files, out = sorted(glob.glob(os.path.join(ROOT, "results", "eval", "t2_*.jsonl"))), "replay_check.json"
    per, tot = {}, Counter()
    for f in files:
        b = os.path.splitext(os.path.basename(f))[0]
        rows = [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
        t0 = time.time()
        with mp.get_context("spawn").Pool(min(14, os.cpu_count() or 2), initializer=init,
                                          initargs=(bench_of(b), b in SERVER_RUNS)) as pool:
            outs = pool.map(job, rows, chunksize=8)
        c = Counter(records=len(rows), miss=sum(o["miss"] for o in outs), reproduced=sum(o.get("same", False) for o in outs),
                    calls=sum(len(o["calls"]) for o in outs), calls_missing=sum(x["missing"] for o in outs for x in o["calls"]),
                    calls_differ=sum(x["differs"] for o in outs for x in o["calls"]))
        per[b] = dict(c)
        tot.update(c)
        print(f"{b:34s} {c['records']:6d} episodes, {c['reproduced']:6d} reproduced, {c['miss']:4d} cache misses, "
              f"{c['calls']:6d} verifier calls ({c['calls_missing']} with a skipped site), {c['calls_differ']} decided "
              f"differently by the rule as run  [{time.time() - t0:.0f} s]", flush=True)
    json.dump({"files": per, "total": dict(tot)}, open(os.path.join(ROOT, "results", "eval", out), "w"), indent=1)
    print(f"total: {tot['records']} episodes, {tot['reproduced']} reproduced ({100 * tot['reproduced'] / tot['records']:.2f} %), "
          f"{tot['calls']} verifier calls, {tot['calls_missing']} with a skipped site, {tot['calls_differ']} decided differently")


if __name__ == "__main__":
    main()
