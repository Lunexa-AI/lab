"""
Compare 10 000 X 10 000 matrix multiplication speed
across three implementations:
  1. Pure-Python nested loops
  2. NumPy broadcast multiply
  3. PyTorch broadcast multiply
  
"""

import time
import numpy as np
import torch

N = 10_000         
REPEAT = 3          

def wall(func, label):
    """Return best-of-REPEAT wall-time for func()."""
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    print(f"{label:<20}: {min(times):8.3f} s")
    return min(times)

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

if __name__ == "__main__":
    print(f"Matrix size: {N:,} x {N:,}\n")
    wall(py_loop,   "Python loops")
    wall(np_mult,   "NumPy *")
    wall(torch_mult,"PyTorch *")
