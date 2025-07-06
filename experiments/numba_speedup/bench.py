import numpy as np
from experiments._utils.perf import benchmark
from experiments.numba_speedup.slow import py_mul
from experiments.numba_speedup.fast import nb_mul

N = 4_000            # 4 000² ≈ 16 M elements -> fits laptop RAM
A_list = [[1.0]*N for _ in range(N)]
B_list = [[2.0]*N for _ in range(N)]
A_np   = np.array(A_list, dtype=np.float32)
B_np   = np.array(B_list, dtype=np.float32)

CASES = {
    "python loops":     lambda: py_mul(A_list, B_list),
    "numba njit":       lambda: nb_mul(A_np, B_np),
}

if __name__ == "__main__":
    benchmark(CASES, repeat=3, baseline="python loops")
