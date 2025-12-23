import os
import threading
from typing import Any, Callable, Dict, Hashable, Iterable, Tuple

from flask import g


def get_perf_stats() -> Dict[str, Any]:
    stats = getattr(g, "_perf_stats", None)
    if stats is None:
        stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "fmp_calls": 0,
            "fmp_batches": 0,
            "yf_calls": 0,
            "yf_batches": 0,
            "parallel_tasks": 0,
        }
        g._perf_stats = stats
    return stats


def get_request_cache() -> Dict[Hashable, Any]:
    cache = getattr(g, "_req_cache", None)
    if cache is None:
        cache = {}
        g._req_cache = cache
    return cache


def cache_get_or_set(key: Hashable, loader: Callable[[], Any]) -> Any:
    cache = get_request_cache()
    stats = get_perf_stats()
    if key in cache:
        stats["cache_hits"] += 1
        return cache[key]
    stats["cache_misses"] += 1
    val = loader()
    cache[key] = val
    return val


def perf_env_threads_enabled() -> bool:
    return str(os.getenv("PERF_ENABLE_THREADS", "0")).lower() in ("1", "true", "yes")


def perf_max_workers() -> int:
    try:
        return max(1, min(8, int(os.getenv("PERF_MAX_WORKERS", "4"))))
    except Exception:
        return 4


thread_lock = threading.Lock()
