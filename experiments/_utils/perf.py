"""
Tiny helpers for repeatable wall-time benchmarks.

Usage
-----
from experiments._utils.perf import benchmark

results = benchmark(
    {
        "pandas read_csv": lambda: pd.read_csv("file.csv"),
        "dask read_csv"  : lambda: dd.read_csv("file.csv").compute(),
    },
    repeat=5,
)
"""

from __future__ import annotations
import time
import statistics
from typing import Callable, Dict, List, Tuple

def _time_once(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def timeit_best(fn: Callable[[], None], repeat: int = 3) -> Tuple[float, List[float]]:
    """Return (best, all_runs) wall times for *fn*, repeated *repeat* times."""
    runs = [_time_once(fn) for _ in range(repeat)]
    return min(runs), runs


def benchmark(cases: Dict[str, Callable[[], None]], repeat: int = 3):
    """
    cases = {"label": callable, ...}
    Prints best wall-time for each label and returns a results list.
    """
    results = []
    for label, fn in cases.items():
        best, runs = timeit_best(fn, repeat)
        print(f"{label:<25}: {best:8.3f} s  (median {statistics.median(runs):.3f})")
        results.append((label, best, runs))
    return results