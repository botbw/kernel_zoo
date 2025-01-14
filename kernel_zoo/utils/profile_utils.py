import torch
from time import time
PROFILE_STEPS = 1
WARM_UP_STEPS = 5

def profile(func, *args, run=PROFILE_STEPS, warm_up=WARM_UP_STEPS, **kwargs):
    for _ in range(warm_up):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    start_t = time()
    for _ in range(run):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    duration = time() - start_t
    return f"\033[31m{duration=}\033[0ms"
