import numpy as np
import torch

def allocate_arrays(power_max: int = 12, backend: str = "numpy", touch=True):
    """
    Allocate arrays of size 2**k elements (float32) for k = 0..power_max.
    Keeps them in a list so they stay resident in RAM.
    """
    arrs = []
    for k in range(power_max + 1):
        n = 2 ** k
        if backend == "numpy":
            a = np.empty(n, dtype=np.float32)
            if touch:
                a.fill(0)
        elif backend == "torch":
            a = torch.empty(n, dtype=torch.float32, device="cpu")
            if touch:
                a.zero_()
        arrs.append(a)
    return arrs
