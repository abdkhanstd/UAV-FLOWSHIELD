#!/bin/bash
# tokens per second of a private server for one short request (the server is busy with other requests, so this is a floor)
for port in 11500 11501; do
  curl -s 127.0.0.1:$port/api/generate -d '{"model":"gemma4:12b","prompt":"List ten words about drones.","stream":false,"options":{"num_predict":80,"temperature":0}}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print("port '$port'", "tok/s", round(d["eval_count"]/(d["eval_duration"]/1e9),1), "prompt tok/s", round(d["prompt_eval_count"]/max(d["prompt_eval_duration"]/1e9,1e-6),1), "load s", round(d["load_duration"]/1e9,2))'
done
nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader
