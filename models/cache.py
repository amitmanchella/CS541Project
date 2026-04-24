"""Disk-backed LLM response cache.

Wraps any BaseLLM and caches (prompt, model) -> response on disk using SQLite.
With temperature=0, identical prompts produce identical outputs, so caching
eliminates redundant API calls across experiments.

Usage:
    from models.cache import CachedLLM
    llm = CachedLLM(FireworksLLM())
"""

import hashlib
import json
import os
import sqlite3
import time
from models.base import BaseLLM

DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "llm_cache.db"
)


class CachedLLM(BaseLLM):
    """Transparent caching wrapper around any BaseLLM."""

    _conn = None  # shared connection (per-process)

    def __init__(self, inner: BaseLLM, cache_path: str = None):
        self.inner = inner
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self.model_name = getattr(inner, "model", "unknown")
        self.hits = 0
        self.misses = 0
        self._ensure_table()

    def _get_conn(self):
        if CachedLLM._conn is None or self.cache_path != getattr(CachedLLM, '_conn_path', None):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            CachedLLM._conn = sqlite3.connect(self.cache_path)
            CachedLLM._conn_path = self.cache_path
        return CachedLLM._conn

    def _ensure_table(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT,
                prompt_hash TEXT,
                response_json TEXT,
                created_at REAL
            )
        """)
        conn.commit()

    @staticmethod
    def _make_key(model: str, prompt: str) -> str:
        h = hashlib.sha256(f"{model}||{prompt}".encode()).hexdigest()
        return h

    def complete(self, prompt: str, output_schema: dict, **kwargs) -> dict:
        key = self._make_key(self.model_name, prompt)
        conn = self._get_conn()

        row = conn.execute(
            "SELECT response_json FROM llm_cache WHERE cache_key = ?", (key,)
        ).fetchone()

        if row is not None:
            self.hits += 1
            cached = json.loads(row[0])
            # Update inner stats as if the call happened (for token accounting)
            meta = cached.get("_meta", {})
            if hasattr(self.inner, 'total_input_tokens'):
                self.inner.total_input_tokens += meta.get("input_tokens", 0)
                self.inner.total_output_tokens += meta.get("output_tokens", 0)
                self.inner.total_cost += meta.get("cost", 0)
                self.inner.call_count += 1
            # Set latency to 0 for cached results (no API call made)
            cached["_meta"]["latency"] = 0.0
            return cached

        self.misses += 1
        result = self.inner.complete(prompt, output_schema, **kwargs)

        # Store the result
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, model, prompt_hash, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, self.model_name, key[:16], json.dumps(result, default=str), time.time()),
        )
        conn.commit()
        return result

    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)

    def get_stats(self) -> dict:
        stats = self.inner.get_stats() if hasattr(self.inner, 'get_stats') else {}
        stats["cache_hits"] = self.hits
        stats["cache_misses"] = self.misses
        return stats

    def reset_stats(self):
        if hasattr(self.inner, 'reset_stats'):
            self.inner.reset_stats()
        self.hits = 0
        self.misses = 0
