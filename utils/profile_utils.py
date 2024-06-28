import torch

PROFILE_STEPS = 10
# WARM_UP_STEPS = 1000

def profile(func, *args, **kwargs):
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # schedule=torch.profiler.schedule(wait=0, warmup=WARM_UP_STEPS, active=PROFILE_STEPS - WARM_UP_STEPS),
    ) as p:
        for _ in range(PROFILE_STEPS):
            func(*args, **kwargs)
            p.step()
    return p.key_averages()
