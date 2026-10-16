"""A small thread-safe LRU memo for expensive pure computations.

Results are computed on the first request and kept in memory, keyed by the
function's arguments; nothing is precomputed at build time. ``memo`` returns
``(result, was_cached)`` so endpoints can report cache hits in ``meta``.
"""
from __future__ import annotations

import functools
import threading
from collections import OrderedDict
from typing import Any, Callable


def memo(maxsize: int = 64) -> Callable:
    def deco(fn: Callable) -> Callable:
        store: OrderedDict = OrderedDict()
        lock = threading.Lock()
        inflight: dict = {}          # key -> lock held while that key is being computed

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> tuple[Any, bool]:
            key = (args, tuple(sorted(kwargs.items())))
            with lock:
                if key in store:
                    store.move_to_end(key)
                    return store[key], True
                key_lock = inflight.setdefault(key, threading.Lock())
            with key_lock:           # concurrent first requests wait instead of computing twice
                with lock:
                    if key in store:
                        store.move_to_end(key)
                        return store[key], True
                try:
                    value = fn(*args, **kwargs)
                except BaseException:
                    with lock:
                        inflight.pop(key, None)
                    raise
                with lock:           # store before releasing the in-flight marker
                    store[key] = value
                    store.move_to_end(key)
                    while len(store) > maxsize:
                        store.popitem(last=False)
                    inflight.pop(key, None)
            return value, False

        wrapper.cache_clear = store.clear  # type: ignore[attr-defined]
        wrapper.uncached = fn  # type: ignore[attr-defined]
        return wrapper
    return deco
