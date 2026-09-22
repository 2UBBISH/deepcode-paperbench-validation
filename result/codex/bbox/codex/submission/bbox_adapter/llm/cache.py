"""A tiny on-disk cache for LLM calls.

Sampling the black-box LLM is by far the most expensive part of BBOX-ADAPTER,
both in dollars and in wall-clock time.  Every request is therefore cached on
disk (keyed by model, prompt and decoding parameters) so that interrupted runs
resume for free and repeated evaluations are deterministic.

Note for the cost analysis of Table 4: cached responses still carry the token
counts of the original request and are therefore still counted by
``UsageTracker``.  A resumed run then reports the same cost as the run that
produced the cache, which is what we want when estimating dollars per 1,000
questions.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Dict, List, Optional


class ResponseCache:
    def __init__(self, cache_dir: Optional[str]) -> None:
        self.cache_dir = cache_dir
        self.path: Optional[str] = None
        self._mem: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.path = os.path.join(cache_dir, "llm_cache.jsonl")
            self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._mem[record["key"]] = record["value"]

    @staticmethod
    def key(model: str, prompt: str, params: Dict[str, Any]) -> str:
        payload = json.dumps(
            {"model": model, "prompt": prompt, "params": params}, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------- access
    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if not self.cache_dir:
            return None
        return self._mem.get(key)

    def put(self, key: str, value: List[Dict[str, Any]]) -> None:
        if not self.cache_dir:
            return
        with self._lock:
            self._mem[key] = value
            if self.path:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "value": value}) + "\n")
