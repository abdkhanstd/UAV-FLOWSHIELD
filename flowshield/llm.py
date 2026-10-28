"""Minimal Ollama chat client with on-disk caching and multi-endpoint load balancing."""
import hashlib
import json
import os
import threading
import time
import itertools

import requests

ENDPOINTS = [e for e in os.environ.get("FS_OLLAMA", "http://127.0.0.1:11435").split(",") if e]
CAPS = [int(x) for x in os.environ.get("FS_OLLAMA_CAP", "8").split(",") if x]


class LLM:
    def __init__(self, model, cache_path, endpoints=None, temperature=0.0, seed=0, num_ctx=4096,
                 num_predict=700, json_mode=True):
        self.model = model
        self.endpoints = endpoints or ENDPOINTS
        self.caps = (CAPS + [4] * len(self.endpoints))[:len(self.endpoints)]
        self.inflight = [0] * len(self.endpoints)
        self.opts = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": num_predict}
        self.json_mode = json_mode
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.cache[r["key"]] = r
                    except Exception:
                        pass

    def key(self, system, user):
        h = hashlib.sha256(json.dumps([self.model, self.opts, self.json_mode, system, user]).encode()).hexdigest()
        return h

    def chat(self, system, user, tag=""):
        k = self.key(system, user)
        if k in self.cache:
            return self.cache[k]
        payload = {"model": self.model, "stream": False, "think": False, "options": self.opts,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.json_mode:
            payload["format"] = "json"
        last = None
        for attempt in range(6):
            with self.lock:
                i = min(range(len(self.endpoints)), key=lambda j: self.inflight[j] / self.caps[j])
                self.inflight[i] += 1
            ep = self.endpoints[i]
            t0 = time.time()
            try:
                r = requests.post(ep + "/api/chat", json=payload, timeout=600)
                d = r.json()
                if "message" not in d:
                    raise RuntimeError(str(d)[:300])
                rec = {"key": k, "tag": tag, "model": self.model, "content": d["message"]["content"],
                       "prompt_tokens": d.get("prompt_eval_count", 0), "output_tokens": d.get("eval_count", 0),
                       "wall_s": time.time() - t0, "endpoint": ep,
                       "prompt_chars": len(system) + len(user)}
                with self.lock:
                    self.cache[k] = rec
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                return rec
            except Exception as ex:
                last = ex
                time.sleep(3 * (attempt + 1))
            finally:
                with self.lock:
                    self.inflight[i] -= 1
        raise RuntimeError(f"LLM call failed: {last}")


def parse_json(text):
    """Lenient JSON extraction from a model response."""
    try:
        return json.loads(text)
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except Exception:
            return None
    return None
