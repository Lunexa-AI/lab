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

def benchmark(
    cases: Dict[str, Callable[[], None]],
    repeat: int = 3,
    baseline: str | None = None,
):
    """
    cases   : {"label": callable, ...}
    repeat  : best-of-N timing per case
    baseline: label to compare against (defaults to *first* item)
    """
    results: list[tuple[str, float, list[float]]] = []

    for label, fn in cases.items():
        best, runs = timeit_best(fn, repeat)
        results.append((label, best, runs))


    if baseline is None:
        baseline = results[0][0]        # first label
    base_time = dict((lbl, t) for lbl, t, _ in results)[baseline]

    summary = {}
    for label, best, runs in results:
        factor = base_time / best
        print(
            f"{label:<25}: {best:8.3f} s "
            f"(×{factor:5.1f} faster vs. {baseline})"
        )
        summary[label] = best
    return summary
