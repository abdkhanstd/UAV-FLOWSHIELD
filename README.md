# UAV-FlowShield

Code, benchmarks and raw results for the paper *UAV-FlowShield: A Trusted Execution Boundary for LLM-Assisted UAV
Mission Planning with Attack-Calibrated Private Views and Authority-Bounded Observations*.

All data are synthetic: mission areas are generated from Matern cluster processes, so no real location, person or
infrastructure is involved. The language-model backbones run locally through Ollama.

## Layout

```
flowshield/            library
  geo.py               secret frames, planar Laplace noise, no-fly-zone-aware routing, exact tour search
  reid.py              registration attacker (long base pair + ICP; weak, known-anchor, fused-release variants),
                       mean-rule and upper-limit noise calibration
  certify.py           certified calibration (exact binomial test over a fixed sequence of scales)
  data.py, attacks.py  benchmark generator T1 (areas, missions, documents, injection test cases)
  attacks2.py          second wording family and the extended adaptive attacks of T2
  auth.py              origin tags of client-portal tickets (HMAC)
  authority.py         ticket semantics: validity window, single use, revocation, approval of the typed action
  prompts.py           prompt builders of every method (two versions of the T3 extractor prompt)
  methods.py           baselines: Prompt-RAG-Validator, EAMSR, MCP-PEP; shared checker; delegation rule
  shield.py            FlowShield: private view, admission (channel authority, origin tag, router-metric
                       envelope), resolver, verifier, canonical and minimum-energy plans
  deterministic.py     reference planners without a language model (basic and extended parser, exact tour, same verifier)
  pipeline.py          one episode through one method (language-model calls, execution, judging)
  pipeline_cb.py       baselines behind the common trusted boundary
  pipeline_t3.py, specs.py   benchmark T3 of open-ended route specifications
  pipeline_final.py    the final configuration end to end
  runtime.py           runtime supervision of a verified plan under mid-flight updates
  dynamics.py          stochastic wind and positioning-noise flight check of a verified plan
  executor.py          kinematic executor and outcome judges
  llm.py               Ollama client with on-disk cache and load-aware routing
scripts/               experiments, analysis and figures (see below)
scripts/remote/        helpers for running the experiments on a second GPU server
tests/                 regression tests of the verifier's completeness rule
data/                  benchmarks (bench.json = T1, bench_t2.json = T2, bench_t3.json, bench_t3h.json), noise scales
results/               raw per-episode results (jsonl), summaries (json) and the language-model caches
results/cache_server/  language-model caches of the runs made on the GPU server (see "Replay check")
results/eval/          runs on T1, T2 and T3; FROZEN.json holds the hashes recorded before T2 was evaluated
```

## Verifier rules

The verifier (`flowshield/shield.py`, `verify_plan`) accepts a plan that omits a site only if the site is the unique
lowest-priority site and even the minimum-energy mission over all sites (required cameras, admitted altitude floors)
exceeds the budget. The minimum is exact: climbs are charged and descents are free, so climbing only where a floor
requires it gives the least climb in every visiting order, and the tour is an exact Held-Karp search in the router's
metric (`min_energy_plan`). An admitted spatial request is a permission, not an obligation: at most one loiter, only at
its handle, clamped to its duration; a plan that omits it verifies and the omission counts as a failed dynamic task in
the mission-success judge. The request is not part of the full mission, so it can never make a site skippable. The
deterministic planner and the verified fallback (`deterministic.plan_for`) follow the same order of preference.

## Benchmarks

* **T1** (`data/bench.json`): development benchmark. Its test split informed one adjustment of the extractor prompt,
  so it is reported as a *post-diagnostic* test.
* **T2** (`data/bench_t2.json`): generated after the pipeline was fixed. `scripts/freeze.py` recorded the SHA-256 of
  every pipeline source and of both benchmarks in `results/eval/FROZEN.json`, and `scripts/remote/verify_frozen.py`
  compares a checkout with this record. T2 has new geography, a second wording family, 305 documents written by a
  model of another family (`granite4.1:3b`), multi-document episodes and extended adaptive attacks; it was evaluated
  in one pass. After the record was made, one unused calibration rule was removed from `scripts/exp_privacy3.py`; for
  the public release documentation strings, comments, display labels and file paths were edited; and the
  implementation of the verifier's completeness rule was corrected, which is a change of executable logic. The recorded
  runs used the earlier implementation, which computed the full mission with the canonical plan (minimum-distance order
  with a return to the default altitude after each floor) and counted the admitted request as part of it; the verifier
  files of commit `f1be207` (`flowshield/shield.py`, `flowshield/deterministic.py`) are identical to the recorded ones
  up to line endings. Commit `5cd01f2` introduced the exact rule described under "Verifier rules".
  `scripts/check_replay.py` shows that the correction alters no verifier decision of any recorded run (see "Replay
  check"). The hash verifier therefore lists these files as differing from the record.
* **T3** (`data/bench_t3.json`): open-ended route specifications; `data/bench_t3h.json` holds held-out rewordings,
  and `data/bench_t3h_screening.json` lists the rewordings removed because their meaning changed.

## Reproduce

Requirements: Python 3.11+, `pip install -r requirements.txt`, Ollama with `gemma4:12b` and `qwen3:8b` (plus
`granite4.1:3b`, `qwen3:1.7b` and `nemotron-3-nano:4b` for the edge-model and small-planner runs). The privacy
experiments run on the CPU; set `OMP_NUM_THREADS=1` and `FS_KD_WORKERS=1` when running a process pool (the KD-tree
queries otherwise use every core in every worker).

### Benchmarks, privacy and trusted components

```bash
python scripts/build_bench.py                         # data/bench.json (T1, seed 11)
FS_OLLAMA=http://127.0.0.1:11500 python scripts/build_bench2.py --author-model granite4.1:3b   # data/bench_t2.json

# privacy: calibration, held-out evaluation, attackers, regret (CPU)
FS_KD_WORKERS=1 FS_POOL=40 python scripts/exp_privacy3.py                       # results/privacy3.json
python scripts/exp_regret.py                                                     # results/regret.json
FS_KD_WORKERS=1 FS_POOL=40 python scripts/exp_privacy4.py                       # results/privacy4.json (side-information attackers)
python scripts/make_scales.py results/privacy3.json ucb10 data/scales_ucb.json  # per-episode noise scales

# trusted components (no language model)
python scripts/exp_fuzz.py 30000; python scripts/exp_fuzz_mutants.py; python scripts/exp_bound.py
python scripts/exp_dynamics.py                                                   # wind and positioning check
```

### Frozen evaluation on T2

```bash
python scripts/freeze.py
python scripts/run_llm.py --model gemma4:12b --split all --bench data/bench_t2.json --scales data/scales_t2.json \
   --methods flowshield,undefended,... --out results/eval/t2_gemma.jsonl     # full list of runs: scripts/remote/run_t2_mata.sh
python scripts/exp_optattack.py --bench data/bench_t2.json --target-model gemma4:12b --attacker-model gemma4:12b --out results/eval/optattack.jsonl
python scripts/exp_optattack.py --bench data/bench_t2.json --target-model gemma4:12b --attacker-model qwen3:8b --scales data/scales_t2.json --out results/eval/optattack_qwen_attacker.jsonl
```

### Calibration statistics and the disclosure decision

```bash
# pools of attack outcomes, coverage of every rule, certified calibration and held-out risk
FS_BENCH=data/bench.json FS_KD_WORKERS=1 FS_POOL=48 python scripts/exp_calib_pool.py results/calib_pool_t1.jsonl 150 40 test
python scripts/exp_calib_coverage.py results/calib_pool_t1.jsonl results/calib_cov_t1.json 300; python scripts/analyze_calib_coverage.py results/calib_cov_t1.json ta
FS_ONLY=cp_fs60,cp_fs100,cp_fs200,cp_tab60,cp_tab100,cp_tab200 python scripts/exp_calib_coverage.py results/calib_pool_t1.jsonl results/calib_cov_t1_tab.json 300 && python scripts/analyze_calib_tab.py
FS_BENCH=data/bench_t2.json FS_KD_WORKERS=1 FS_POOL=40 python scripts/exp_privacy3d.py cert100 100 results/privacy3_cert100_t2.json all   # about 80 min on 44 cores
# region-level disclosure (A), attacker saturation (B), disclosure decision (C, with alpha/4 per declared threat)
FS_PER_AREA=20 FS_KD_WORKERS=1 FS_POOL=10 python scripts/exp_privacy5.py A results/privacy5_A.json   # also B; C: scripts/remote/run_partC.sh
python scripts/analyze_p5.py; python scripts/fig_frontier.py results/privacy5_C_bonf.json b
python scripts/build_geo4.py && FS_BENCH=data/bench_geo4.json python scripts/exp_geo4.py 100 results/geo4.json && python scripts/analyze_geo4.py   # third geography
python scripts/exp_calib_time_cert.py; python scripts/exp_router_triangle.py results/router_triangle.json
```

### Common boundary, open-ended specifications, runtime supervision and ticket authority

```bash
python scripts/run_cb.py --model gemma4:12b --split all --bench data/bench_t2.json --scales data/scales_t2.json --methods undefended_cb,prompt_only_cb,spotlight_guard_cb,prv_cb,eamsr_cb,mcp_pep_cb --out results/eval/t2_gemma_cb.jsonl
python scripts/build_bench3.py --author-model granite4.1:3b --out data/bench_t3.json
python scripts/run_t3.py --model gemma4:12b --split all --bench data/bench_t3.json --scales data/scales_t3.json --methods flowshield,fs_no_view,undefended,det_plus,t3_regex,t3_dsl_cov,t3_dsl_all,t3_llm_dslv,t3_dsl_cov2,t3_dsl_all2,t3_llm_dslv2 --out results/eval/t3_gemma.jsonl
python scripts/build_bench3h.py && python scripts/run_t3.py --model gemma4:12b --bench data/bench_t3h.json --scales data/scales_t3.json \
       --methods flowshield,fs_no_view,undefended,det_plus,t3_regex,t3_dsl_cov2,t3_dsl_all2,t3_llm_dslv2 --tag t3h --out results/eval/t3h_gemma.jsonl
python scripts/exp_runtime.py results/runtime.json
python scripts/exp_authority.py --out results/authority.json     # ticket semantics: oracle, 7 weakened variants, replay attempts
```

### Final configuration end to end

```bash
python scripts/run_final.py decide && python scripts/run_final.py freeze      # decision file and hashes, before the run
python scripts/run_final.py run --model gemma4:12b --bench data/bench_t2.json --split all --methods fs_final,fs_final_expert,fs_final_habit --out results/eval/t2_gemma_final.jsonl
```

### Tables, figures and macros

`bash scripts/analyze_all.sh` regenerates every table, figure and number of the paper from the raw results (written to
`../paper/auto` and `../paper/figs`); no model is needed, because every language-model call is cached by prompt hash.

### Replay check

```bash
python scripts/check_replay.py        # every T2 run -> results/eval/replay_check.json
python scripts/check_replay.py t1     # every T1 run -> results/eval/replay_check_t1.json
python tests/test_verifier.py         # regression tests of the completeness rule (or: python -m pytest tests)
```

`check_replay.py` re-runs every recorded episode run (one per episode, method and configuration) with the released code,
taking each model response from the cache (a missing response is reported, no model is called), and compares the
outcome with the logged one. Runs made on the GPU server read `results/cache_server/` first. On T2 (378 episodes),
39,775 of the 39,870 recorded episode runs (99.76 %) reproduce exactly and on T1 29,049 of 29,131 (99.72 %); the others involve a prompt for which the cache holds two different responses (requests
sent concurrently), so a replay can use only one of them. At every verifier call the script also evaluates the
completeness rule in the form the runs used; it decides as the released rule at all 30,424 T2 calls (6,996 on a plan
that skips a site) and all 17,844 T1 calls. T3 admits no observation, so the two forms coincide there.
The main analysis of T1 and T2 is `scripts/analyze_eval.py`; `scripts/eval_common.py` holds the loaders and method
labels that the other analysis scripts share. `scripts/fig_compact.py` draws the small figures of the main paper.

## Settings

Ollama endpoints are read from `FS_OLLAMA` (comma separated) with per-endpoint slot counts in `FS_OLLAMA_CAP`.
Decoding: temperature 0, seed 0, context 4096, JSON mode, 700 new tokens.

The Ollama build used here also exposes a Vulkan backend that ignores `CUDA_VISIBLE_DEVICES`. Start each server with
`CUDA_VISIBLE_DEVICES=<GPU UUID>`, `OLLAMA_VULKAN=0` (and `GGML_VK_VISIBLE_DEVICES`), check the `inference compute`
line of the server log (it must list only the intended PCI device) and confirm with `nvidia-smi` after the first
request (`scripts/remote/start_ollama.sh <gpu-uuid> [port]`).

T1 runs and the privacy experiments ran on a workstation (2 x RTX 2080 Ti) and the T2 runs on a server (2 x RTX 3090).
Decoding is not bit-identical across the two machines (about 97 % identical decisions on a 142-case check), so each
batch that is compared inside one table was run on one machine.

The attack simulations of dense registries allocate several hundred MB per process; on a 64 GB machine keep the
process pool at about 30 workers or fewer. On Windows the pool workers re-import the scripts, so parameters are read
at import time (`sys.argv`, environment), as in `exp_privacy3d.py`.

## Fairness of the comparison

The baselines are re-implementations from the published descriptions, run under identical missions, documents,
backbone, decoding, router and executor. They receive absolute coordinates because that is how they are specified.
Where a baseline was designed for a neighbouring task (EAMSR admits operator submissions, not runtime documents), the
adaptation is stated in the paper. FlowShield gives its planner a verifier feedback loop and a delegation rule; to
separate these capabilities from the security mechanisms, every baseline is also run with the same repair feedback
(`*_rep`) and, for EAMSR and MCP-PEP, with the same delegation rule (`*_repdel`).
