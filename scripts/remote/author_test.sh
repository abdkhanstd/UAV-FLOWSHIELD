#!/bin/bash
# Small authoring test on the remote machine: model name in $1
cd "$HOME/cae_exp/code" || exit 1
export FS_OLLAMA=http://127.0.0.1:11500
export FS_OLLAMA_CAP=8
python3 scripts/build_bench2.py --small --author-model "$1" --out data/t2_small_$$.json 2>&1 | tail -3
python3 - "$$" <<'EOF'
import json, sys
B = json.load(open(f"data/t2_small_{sys.argv[1]}.json"))
n = 0
for e in B["episodes"]:
    for d in e["docs"]:
        if d.get("author") not in ("tmpl2", None) and (e["category"] in ("attack",) or n < 6) and n < 24:
            n += 1
            print(e["id"], e["category"], (e["attack"] or {}).get("goal"), d["channel"], "|", d["text"][:330])
EOF
rm -f data/t2_small_$$.json
