import time
import torch
from typing import Callable
from dataclasses import dataclass

default_warmup = 500
default_iters = 100
cooldown_time = 2.0

@dataclass(frozen=True)
class BenchResult:
    name: str
    us: float
    tflops: float | None = None
    bandwidth_tb_s: float | None = None
    groups: int | None = None

def l2_cache_size_bytes(device: int = 0) -> int:
    return torch.cuda.get_device_properties(device).L2_cache_size

def input_group_count(input_bytes: int, l2_bytes: int | None = None) -> int:
    if l2_bytes is None: l2_bytes = l2_cache_size_bytes()
    return 1 if input_bytes >= 3 * l2_bytes else int(3 * l2_bytes / input_bytes) + 1

def uniform_bf16(shape: tuple[int, ...], seed: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.rand(shape, device="cuda", dtype=torch.float32, generator=gen)
    x = x * (high - low) + low
    return x.to(torch.bfloat16)

def normal_bf16(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)

def profile_groups(
    name: str,
    groups: list[object],
    fn: Callable[[object], None],
    *,
    warmup: int = default_warmup,
    iters: int = default_iters,
    cooldown_s: float = cooldown_time,
    flops: float | None = None,
    bytes_moved: int | None = None,
) -> BenchResult:
    if not groups: raise ValueError("profile_groups requires at least one input group")
    torch.cuda.synchronize()
    for i in range(warmup): fn(groups[i % len(groups)])
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(iters): fn(groups[i % len(groups)])
    end.record()
    torch.cuda.synchronize()

    us = start.elapsed_time(end) * 1000.0 / iters
    tflops = None if flops is None else flops / (us * 1e-6) / 1e12
    bandwidth = None if bytes_moved is None else bytes_moved / (us * 1e-6) / 1e12

    if cooldown_s:
        torch.cuda.synchronize()
        time.sleep(cooldown_s)
    return BenchResult(name=name, us=us, tflops=tflops, bandwidth_tb_s=bandwidth, groups=len(groups))

def print_bench(result: BenchResult) -> None:
    print(f"\n{result.name}:")
    if result.groups is not None: print(f"  groups: {result.groups}")
    if result.tflops is not None: print(f"  TFLOPS: {result.tflops:.0f}")
    if result.bandwidth_tb_s is not None: print(f"  BW:     {result.bandwidth_tb_s:.3f} TB/s")
    print(f"  time:   {result.us:.2f} us")

def max_mean_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()

def check_close(name: str, a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float = 5e-2) -> bool:
    max_diff, mean_diff = max_mean_diff(a, b)
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    print(f"  {name}: max={max_diff:.6g} mean={mean_diff:.6g} {'PASS' if ok else 'FAIL'}")
    return ok

def total_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)
