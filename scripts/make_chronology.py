"""Times of the T2 test protocol, written as macros for the paper.

The hashes of the evaluated files and the time of recording are read from results/eval/FROZEN.json. The time of the
last edit of a pipeline source and the time at which T2 was generated are file-system times on the authors'
workstation, read before the public release (git does not preserve file times), and are therefore stated here.
usage: python scripts/make_chronology.py   -> ../paper/auto/numbers_chron.tex"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
LAST_PIPELINE_EDIT = "22:15"   # 2026-10-26, last modification of a recorded pipeline source
T2_GENERATED = "22:16"         # 2026-10-26, data/bench_t2.json written


def main():
    F = json.load(open(os.path.join(ROOT, "results", "eval", "FROZEN.json")))
    out = (f"\\newcommand{{\\chronHashN}}{{{len(F['files'])}}}\n"
           f"\\newcommand{{\\chronFrozenAt}}{{{F['time'][11:16]}}}\n"
           f"\\newcommand{{\\chronLastEdit}}{{{LAST_PIPELINE_EDIT}}}\n"
           f"\\newcommand{{\\chronTwoAt}}{{{T2_GENERATED}}}\n")
    open(os.path.join(AUTO, "numbers_chron.tex"), "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
