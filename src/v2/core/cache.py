"""
cache.py — Exact-Match LRU Query Cache (v2)

Provides an in-process LRU cache for RAG answers.  When the same
question is asked twice (exact match), the cached answer is returned
instantly without hitting the LLM or retrieval stack.

Design choices:
  - Uses Python's built-in functools.lru_cache under the hood (fast, GIL-safe).
  - Cache key = (question, workspace_id, search_mode) so workspace-scoped
    results don't bleed across tenants.
  - TTL is NOT implemented at this layer — the cache is cleared on server
    restart (acceptable for local deployments).  Redis would add TTL.
  - Thread-safe: lru_cache uses a per-key lock internally.

Usage:
    cache = QueryCache(max_size=100)
    hit = cache.get(question, workspace_id, search_mode)
    if hit:
        return hit
    result = pipeline.ask(...)
    cache.set(question, workspace_id, search_mode, result)
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIZE = 100   # Entries; each ~2-5 KB → ~500 KB total RAM


class QueryCache:
    """
    Thread-safe LRU cache for (question, workspace_id, search_mode) → result.

    Evicts the least-recently-used entry when capacity is exceeded.
    """

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE) -> None:
        self._cache: OrderedDict[tuple, dict] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        logger.info(f"QueryCache initialised (max_size={max_size})")

    # ------------------------------------------------------------------

    def _make_key(
        self,
        question: str,
        workspace_id: str,
        search_mode: str,
    ) -> tuple:
        """Normalise the cache key."""
        return (question.strip().lower(), workspace_id, search_mode)

    def get(
        self,
        question: str,
        workspace_id: str = "default",
        search_mode: str = "combined",
    ) -> Optional[dict]:
        """
        Return cached result or None (cache miss).

        Also promotes the entry to the front of the LRU order.
        """
        key = self._make_key(question, workspace_id, search_mode)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug(f"Cache HIT: '{question[:50]}' (hits={self._hits})")
                return self._cache[key]
            self._misses += 1
            return None

    def set(
        self,
        question: str,
        workspace_id: str,
        search_mode: str,
        result: dict,
    ) -> None:
        """Store a result, evicting the LRU entry if at capacity."""
        key = self._make_key(question, workspace_id, search_mode)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = result
            if len(self._cache) > self._max_size:
                evicted = self._cache.popitem(last=False)
                logger.debug(f"Cache evicted LRU key: {evicted[0]}")

    def clear(self) -> None:
        """Flush all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        """Return cache stats for the /metrics endpoint."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "cache_size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
            }
