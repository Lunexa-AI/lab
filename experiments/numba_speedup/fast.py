"""
Numba-accelerated element-wise multiply
"""

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def nb_mul(A, B):
    N = A.shape[0]
    C = np.empty_like(A)
    for i in prange(N):
        for j in prange(N):
            C[i, j] = A[i, j] * B[i, j]
    return C
