import numpy as np, torch, time

def allocate_arrays(power_max=17, backend="numpy", touch=True, pause=0.0):
    """
    Allocate (2**k x 2**k) float32 tensors for k = 0..power_max.
    If *touch* is True we write to every element so pages are committed.
    *pause* inserts a small sleep after each allocation so external samplers
    can catch every step. Returns the list so caller keeps them alive.
    """
    arrs = []
    for k in range(power_max + 1):
        n = 2 ** k
        if backend == "numpy":
            a = (np.ones if touch else np.empty)((n, n), dtype=np.float32)
        else:
            tensor_ctor = torch.ones if touch else torch.empty
            a = tensor_ctor((n, n), dtype=torch.float32, device="cpu")
        arrs.append(a)
        if pause:
            time.sleep(pause)
    return arrs

