"""
Compare 10 000 X 10 000 matrix multiplication speed
across three implementations:
  1. Pure-Python nested loops
  2. NumPy broadcast multiply
  3. PyTorch broadcast multiply
  
"""

import datetime, pathlib,timeit
import numpy as np
import torch

from experiments._utils.perf import benchmark

N = 10_000         

# 1: Pure-Python nested loops:
def py_loop():
    a = [[1.0] * N for _ in range(N)]
    b = [[2.0] * N for _ in range(N)]
    c = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            c[i][j] = a[i][j] * b[i][j]
    return c

# 2: NumPy 
def np_mult():
    a = np.ones((N, N), dtype=np.float32)
    b = np.full((N, N), 2.0, dtype=np.float32)
    c = a * b
    return c

# 3: PyTorch
def torch_mult():
    a = torch.ones((N, N), dtype=torch.float32)
    b = torch.full((N, N), 2.0, dtype=torch.float32)
    c = a * b
    return c

CASES = {
    "Python loops": py_loop,
    "NumPy *":      np_mult,
    "PyTorch *":    torch_mult,
}

def run_timeit(label, fn, repeat=3, number=1):
    """Return best wall-time from timeit for parity check."""
    timer = timeit.Timer(fn)
    return min(timer.timeit(number=number) for _ in range(repeat))

if __name__ == "__main__":
    print(f"Matrix size: {N:,} × {N:,}\n")
    
    bests = benchmark(CASES, repeat=3, baseline="Python loops")
    
    timeit_bests = {lbl: run_timeit(lbl, fn) for lbl, fn in CASES.items()}

    out = pathlib.Path(__file__).with_name("results.md")
    header = "| date | variant | perf_counter_s | timeit_s |\n|---|---|---|---|\n"
    if not out.exists():
        out.write_text(header)

    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = [
        f"| {today} | {lbl} | {bests[lbl]:.3f} | {timeit_bests[lbl]:.3f} |"
        for lbl in CASES
    ]
    with out.open("a") as f:
        f.write("\n".join(rows) + "\n")

    print(f"\nLogged results → {out}")
