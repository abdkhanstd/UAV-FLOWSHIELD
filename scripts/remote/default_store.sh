#!/bin/bash
# Read-only: models known to the machine's default Ollama service (port 11434).
curl -s 127.0.0.1:11434/api/tags | python3 -c 'import sys,json; [print(m["name"], round(m["size"]/1e9,1), "GB", m["modified_at"][:19]) for m in json.load(sys.stdin)["models"]]'
