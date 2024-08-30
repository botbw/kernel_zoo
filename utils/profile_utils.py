import torch
from time import time
PROFILE_STEPS = 10
# WARM_UP_STEPS = 1000

def profile(func, *args, **kwargs):
    torch.cuda.synchronize()
    start_t = time()
    for _ in range(PROFILE_STEPS):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    duration = time() - start_t
    return f"\033[31m{duration=}\033[0ms"
