#!/bin/bash
# Status of the CAE experiment environment on the remote machine.
BASE="$HOME/cae_exp"
echo "== pull log (last progress line)"
tail -c 400 "$BASE/logs/pull.log" 2>/dev/null | tr '\r' '\n' | grep -v '^$' | tail -2
echo "== running pulls / runs"
ps -eo pid,etime,args | grep -E 'ollama pull|run_llm|exp_' | grep -v grep | cut -c1-150
echo "== models in my store"
find "$BASE/ollama_models/manifests" -type f 2>/dev/null | sed 's#.*/library/##'
du -sh "$BASE/ollama_models" 2>/dev/null
echo "== GPUs"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
echo "== servers"
for p in 11500 11501; do echo -n "$p: "; curl -s 127.0.0.1:$p/api/ps | head -c 200; echo; done
