import time

import torch

from .metrics import attention_bwd_tflops, attention_fwd_tflops, gbps


def time_ms(fn, *, warmup=500, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_ms(fn_for_group, group_count, *, warmup=500, iters=100, cooldown_s=0.2):
    for i in range(warmup):
        fn_for_group(i % group_count)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        fn_for_group(i % group_count)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    if cooldown_s > 0:
        time.sleep(cooldown_s)
    return ms


def recommended_group_count(bytes_per_group, *, l2_multiplier=3, min_groups=1):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2 = getattr(props, "l2_cache_size", 0) or getattr(props, "L2_cache_size", 0) or 0
    if l2 <= 0 or bytes_per_group <= 0:
        return min_groups
    target = l2_multiplier * l2
    if bytes_per_group >= target:
        return min_groups
    return max(min_groups, int((target + bytes_per_group - 1) // bytes_per_group))


def uniform_tensor(shape, *, generator, device="cuda"):
    return torch.empty(shape, device=device, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)


def print_profile_line(label, ms, *, bytes_moved=None, flops_shape=None, kind=None):
    suffix = ""
    if bytes_moved is not None:
        suffix = f"  {gbps(bytes_moved, ms):8.1f} GB/s"
    if flops_shape is not None:
        B, H, N, D = flops_shape
        tflops = attention_fwd_tflops(B, H, N, D, ms) if kind == "fwd" else attention_bwd_tflops(B, H, N, D, ms)
        suffix = f"  {tflops:7.2f} TFLOP/s"
    print(f"  {label:<28} {ms:7.4f} ms{suffix}")
