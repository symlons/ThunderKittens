"""Profile FP8 attention prep, forward, and backward kernels."""

import argparse

import torch
import torch.nn.functional as F

from fp8_suite.kernel_api import require_extension
from fp8_suite.metrics import attention_bwd_tflops, attention_fwd_tflops
from fp8_suite.profiling import benchmark_ms, print_profile_line, recommended_group_count, uniform_tensor
from fp8_suite.test_backward_kernel import profile_kernels


def make_sdpa_groups(shape, dtype, *, seed, group_count):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    groups = []
    for _ in range(group_count):
        Q = uniform_tensor(shape, generator=generator).to(dtype).requires_grad_(True)
        K = uniform_tensor(shape, generator=generator).to(dtype).requires_grad_(True)
        V = uniform_tensor(shape, generator=generator).to(dtype).requires_grad_(True)
        dO = uniform_tensor(shape, generator=generator).to(dtype)
        groups.append((Q, K, V, dO))
    torch.cuda.synchronize()
    return groups


def profile_sdpa(shape, dtype, *, seed, group_count, warmup, iters, cooldown_s):
    groups = make_sdpa_groups(shape, dtype, seed=seed, group_count=group_count)

    def fwd(i):
        Q, K, V, _ = groups[i]
        return F.scaled_dot_product_attention(Q, K, V, is_causal=False)

    def bwd(i):
        Q, K, V, dO = groups[i]
        O = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        return torch.autograd.grad(O, (Q, K, V), dO, retain_graph=False, create_graph=False)

    fwd_ms = benchmark_ms(fwd, group_count, warmup=warmup, iters=iters, cooldown_s=cooldown_s)
    bwd_ms = benchmark_ms(bwd, group_count, warmup=warmup, iters=iters, cooldown_s=cooldown_s)
    name = "torch sdpa " + str(dtype).replace("torch.", "")
    print_profile_line(f"{name} fwd", fwd_ms, flops_shape=shape, kind="fwd")
    print_profile_line(f"{name} bwd", bwd_ms, flops_shape=shape, kind="bwd")
    return {"fwd_ms": fwd_ms, "bwd_ms": bwd_ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--N", type=int, default=1536)
    parser.add_argument("--D", type=int, default=128, choices=(64, 128))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--bench-warmup", type=int, default=500)
    parser.add_argument("--bench-groups", type=int, default=None)
    parser.add_argument("--bench-cooldown", type=float, default=0.2)
    parser.add_argument("--no-sdpa", action="store_true")
    args = parser.parse_args()

    require_extension("fp8_mha_forward", "fp8_mha_backward")
    shape = (args.B, args.H, args.N, args.D)
    print(f"[profile B={args.B} H={args.H} N={args.N} D={args.D} seed={args.seed}]")
    bytes_per_group = 4 * args.B * args.H * args.N * args.D * 4
    group_count = args.bench_groups or recommended_group_count(bytes_per_group)
    fp8 = profile_kernels(
        shape=shape,
        seed=args.seed,
        bench_iters=args.bench_iters,
        warmup=args.bench_warmup,
        group_count=group_count,
        cooldown_s=args.bench_cooldown,
    )
    if args.no_sdpa:
        return

    print("\n[torch SDPA baselines]")
    fp32 = profile_sdpa(
        shape,
        torch.float32,
        seed=args.seed,
        group_count=group_count,
        warmup=args.bench_warmup,
        iters=args.bench_iters,
        cooldown_s=args.bench_cooldown,
    )
    bf16 = profile_sdpa(
        shape,
        torch.bfloat16,
        seed=args.seed,
        group_count=group_count,
        warmup=args.bench_warmup,
        iters=args.bench_iters,
        cooldown_s=args.bench_cooldown,
    )
    print("\n[speedup: FP8 raw attention kernel vs SDPA]")
    for name, base in (("torch sdpa fp32", fp32), ("torch sdpa bf16", bf16)):
        print(f"  vs {name:<16} fwd {base['fwd_ms'] / fp8['fwd_ms']:.2f}x  bwd {base['bwd_ms'] / fp8['bwd_ms']:.2f}x")
    fwd_with_quant = fp8["quant_q_ms"] + fp8["quant_k_ms"] + fp8["fwd_ms"]
    print("\n[speedup: FP8 forward including Q/K token quantization]")
    for name, base in (("torch sdpa fp32", fp32), ("torch sdpa bf16", bf16)):
        print(f"  vs {name:<16} fwd {base['fwd_ms'] / fwd_with_quant:.2f}x")


if __name__ == "__main__":
    main()
