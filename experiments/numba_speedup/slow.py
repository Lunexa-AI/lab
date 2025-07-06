"""
Baseline element-wise multiply using plain Python loops

"""

import random

def py_mul(A, B):
    """
    Multiply two same-size lists of lists (NxN) element-wise.

    Parameters
    ----------
    A, B : list[list[float]]
    Returns
    -------
    C     : list[list[float]]
    """
    N = len(A)
    C = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            C[i][j] = A[i][j] * B[i][j]
    return C


# tiny smoke-test
if __name__ == "__main__":
    N = 2
    a = [[random.random() for _ in range(N)] for _ in range(N)]
    b = [[random.random() for _ in range(N)] for _ in range(N)]
    print(py_mul(a, b))
