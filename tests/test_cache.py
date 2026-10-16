import threading
import time

from modlab.server.cache import memo


def test_concurrent_first_calls_compute_once():
    calls = []

    @memo(maxsize=4)
    def slow(x):
        calls.append(x)
        time.sleep(0.05)
        return x * 2

    results = []
    threads = [threading.Thread(target=lambda: results.append(slow(3))) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert calls == [3]
    assert sorted(r[1] for r in results) == [False] + [True] * 7


def test_lru_eviction_and_errors_not_cached():
    n = {"calls": 0}

    @memo(maxsize=2)
    def f(x):
        n["calls"] += 1
        if x < 0:
            raise ValueError("negative")
        return x

    f(1), f(2), f(3)
    assert f(1) == (1, False)                 # evicted, recomputed
    for _ in range(2):
        try:
            f(-1)
        except ValueError:
            pass
    assert n["calls"] == 6                     # failures are retried, never cached
