#!/bin/bash
# Give the CPU-bound privacy workers the lowest priority so that the model servers and the LLM runs keep their CPU time.
for p in $(pgrep -u "$USER" -f "scripts/exp_privacy"); do renice -n 19 -p "$p" > /dev/null; done
echo "reniced $(pgrep -u "$USER" -f 'scripts/exp_privacy' | wc -l) processes"
