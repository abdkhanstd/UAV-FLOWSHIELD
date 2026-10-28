"""End-to-end run of the final recommended configuration (flowshield/pipeline_final.py) on T2.

  python scripts/run_final.py decide                      -> data/decision_t2_final.json (from results/privacy3_cert100_t2.json)
  python scripts/run_final.py freeze                      -> results/eval/FROZEN_FINAL.json (hashes of the configuration, before the run)
  python scripts/run_final.py run --model gemma4:12b --bench data/bench_t2.json --methods fs_final,fs_final_expert,fs_final_habit \
         --out results/eval/t2_gemma_final.jsonl          (the arguments of run_llm.py)"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
FILES = ["flowshield/pipeline_final.py", "flowshield/authority.py", "flowshield/certify.py", "flowshield/shield.py", "flowshield/pipeline.py",
         "flowshield/deterministic.py", "flowshield/prompts.py", "flowshield/auth.py", "scripts/run_final.py", "scripts/run_llm.py",
         "data/bench_t2.json", "data/decision_t2_final.json", "results/privacy3_cert100_t2.json"]


def decide():
    from flowshield.pipeline_final import CAP
    cal = json.load(open(os.path.join(ROOT, "results", "privacy3_cert100_t2.json")))["calibration"]
    out = {c["id"]: {"scale": c["scale"], "certified": bool(c["certified"]), "release": bool(c["certified"] and c["scale"] <= CAP)} for c in cal}
    json.dump({"rule": f"certified (blind registration, T=100, alpha=0.05) and scale <= {CAP:.0f} m", "missions": out},
              open(os.path.join(ROOT, "data", "decision_t2_final.json"), "w"), indent=0)
    n = len(out)
    print(n, "missions; released", sum(v["release"] for v in out.values()), "; uncertified", sum(not v["certified"] for v in out.values()))


def freeze():
    h = {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest() for f in FILES}
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "files": h}, open(os.path.join(ROOT, "results", "eval", "FROZEN_FINAL.json"), "w"), indent=1)
    print("frozen", len(h), "files")


if __name__ == "__main__":
    cmd = sys.argv.pop(1)
    if cmd == "decide":
        decide()
    elif cmd == "freeze":
        freeze()
    else:
        import flowshield.pipeline_final as PF
        import run_llm
        run_llm.run_episode = PF.run_episode
        run_llm.main()
