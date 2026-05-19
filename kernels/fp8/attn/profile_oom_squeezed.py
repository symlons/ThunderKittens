"""Forward-only FP8 attention profile that fits in <1 GB free HBM.

Used when the GPU is shared and the full long-context sweep would OOM.
Single group per shape, minimal warmup/iters, no fp32 SDPA baseline.
"""

import argparse
import gc

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from fp8_suite.kernel_api import fp8_forward, require_extension
from fp8_suite.metrics import attention_fwd_tflops
from fp8_suite.profiling import benchmark_ms, uniform_tensor
from fp8_suite.recipe import prepare_forward_inputs


def round_to_384(n):
    return (n // 384) * 384


SHAPES_BHND = [
    (1, 1, round_to_384(8192),   128),
    (1, 1, round_to_384(16384),  128),
    (1, 1, round_to_384(32768),  128),
    (1, 1, round_to_384(65536),  128),
    (1, 1, round_to_384(98304),  128),
    (1, 1, round_to_384(131072), 128),
    (1, 1, round_to_384(196608), 128),
    (1, 1, round_to_384(262144), 128),
    (1, 1, round_to_384(307200), 128),  # ~300K
]


def time_fp8_fwd(shape, seed, warmup, iters):
    B, H, N, D = shape
    gen = torch.Generator(device="cuda").manual_seed(seed)
    Q = uniform_tensor(shape, generator=gen)
    K = uniform_tensor(shape, generator=gen)
    V = uniform_tensor(shape, generator=gen)
    fwd = prepare_forward_inputs(Q, K, V, use_cuda_quant=True)
    def _run(_):
        return fp8_forward(fwd)
    torch.cuda.synchronize()
    ms = benchmark_ms(_run, 1, warmup=warmup, iters=iters, cooldown_s=0.1)
    del Q, K, V, fwd
    gc.collect(); torch.cuda.empty_cache()
    return ms


def time_bf16_sdpa_fwd(shape, seed, warmup, iters):
    B, H, N, D = shape
    Q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    def _run(_):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    torch.cuda.synchronize()
    ms = benchmark_ms(_run, 1, warmup=warmup, iters=iters, cooldown_s=0.1)
    del Q, K, V
    gc.collect(); torch.cuda.empty_cache()
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()
    require_extension("fp8_mha_forward")

    free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    print(f"# free HBM at start: {free_mib} MiB")
    print(f"# B H      N      D | FP8 fwd ms / TFLOP/s | bf16 SDPA ms / TFLOP/s | FP8/bf16 speedup")

    for shape in SHAPES_BHND:
        B, H, N, D = shape
        try:
            t_fp8 = time_fp8_fwd(shape, args.seed, args.warmup, args.iters)
            tflops_fp8 = attention_fwd_tflops(B, H, N, D, t_fp8)
            fp8_str = f"{t_fp8:7.2f} ms / {tflops_fp8:6.1f} TF"
        except torch.cuda.OutOfMemoryError as e:
            fp8_str = "OOM"
            t_fp8 = None
        try:
            t_bf = time_bf16_sdpa_fwd(shape, args.seed, args.warmup, args.iters)
            tflops_bf = attention_fwd_tflops(B, H, N, D, t_bf)
            bf_str = f"{t_bf:7.2f} ms / {tflops_bf:6.1f} TF"
            speed = f"{t_bf/t_fp8:.2f}x" if t_fp8 else "n/a"
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            bf_str = "OOM/err"
            speed = "n/a"
        print(f"  {B} {H} {N:>6} {D:>3} | {fp8_str} | {bf_str} | {speed}")


if __name__ == "__main__":
    main()
