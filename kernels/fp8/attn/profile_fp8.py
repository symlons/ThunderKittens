"""Profile FP8 attention prep, forward, and backward kernels."""

import argparse

import torch
import torch.nn.functional as F

from fp8_suite.kernel_api import require_extension
from fp8_suite.profiling import benchmark_ms, print_profile_line, recommended_group_count, uniform_tensor
from fp8_suite.test_backward_kernel import profile_kernels


def make_sdpa_groups(shape, dtype, *, seed, group_count, profile_bwd):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    groups = []
    for _ in range(group_count):
        Q = uniform_tensor(shape, generator=generator).to(dtype).requires_grad_(profile_bwd)
        K = uniform_tensor(shape, generator=generator).to(dtype).requires_grad_(profile_bwd)
        V = uniform_tensor(shape, generator=generator).to(dtype).requires_grad_(profile_bwd)
        dO = uniform_tensor(shape, generator=generator).to(dtype) if profile_bwd else None
        O = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        groups.append((Q, K, V, dO, O))
    torch.cuda.synchronize()
    return groups


def profile_sdpa(shape, dtype, *, seed, group_count, warmup, iters, cooldown_s, profile_fwd, profile_bwd):
    groups = make_sdpa_groups(shape, dtype, seed=seed, group_count=group_count, profile_bwd=profile_bwd)

    out = {}
    name = "torch sdpa " + str(dtype).replace("torch.", "")

    if profile_fwd:
        def fwd(i):
            Q, K, V, _, _ = groups[i]
            return F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        fwd_ms = benchmark_ms(fwd, group_count, warmup=warmup, iters=iters, cooldown_s=cooldown_s)
        print_profile_line(f"{name} fwd", fwd_ms, flops_shape=shape, kind="fwd")
        out["fwd_ms"] = fwd_ms

    if profile_bwd:
        def bwd(i):
            Q, K, V, dO, O = groups[i]
            return torch.autograd.grad(O, (Q, K, V), dO, retain_graph=True, create_graph=False)
        bwd_ms = benchmark_ms(bwd, group_count, warmup=warmup, iters=iters, cooldown_s=cooldown_s)
        print_profile_line(f"{name} bwd-only", bwd_ms, flops_shape=shape, kind="bwd")
        out["bwd_ms"] = bwd_ms

    return out


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
    parser.add_argument("--stages", nargs="+", choices=("fwd", "bwd"), default=("fwd", "bwd"))
    parser.add_argument("--sdpa-dtypes", nargs="+", choices=("bf16", "fp32"), default=("bf16",),
                        help="Torch SDPA baseline dtypes to run; defaults to bf16 only.")
    parser.add_argument("--no-sdpa", action="store_true")
    parser.add_argument("--bwd-sdp-descale-mode", choices=("estimate", "constant"), default="estimate",
                        help="Use constant for profile-only long-N backward; estimate materializes O(N^2) Python reference state.")
    args = parser.parse_args()

    require_extension("fp8_mha_forward", "fp8_mha_backward")
    shape = (args.B, args.H, args.N, args.D)
    profile_fwd = "fwd" in args.stages
    profile_bwd = "bwd" in args.stages

    print(f"[profile B={args.B} H={args.H} N={args.N} D={args.D} seed={args.seed}]")
    if profile_bwd and args.bwd_sdp_descale_mode == "constant":
        print("[profile note] FP8 backward uses constant sdp_row; timing-only, not correctness-quality.")

    bytes_per_group = 4 * args.B * args.H * args.N * args.D * 4
    group_count = args.bench_groups or recommended_group_count(bytes_per_group)
    fp8 = profile_kernels(
        shape=shape,
        seed=args.seed,
        bench_iters=args.bench_iters,
        warmup=args.bench_warmup,
        group_count=group_count,
        cooldown_s=args.bench_cooldown,
        profile_fwd=profile_fwd,
        profile_bwd=profile_bwd,
        sdp_descale_mode=args.bwd_sdp_descale_mode,
    )
    if args.no_sdpa:
        return

    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32}
    baselines = {}
    print("\n[torch SDPA baselines]")
    for dtype_name in args.sdpa_dtypes:
        baselines[dtype_name] = profile_sdpa(
            shape,
            dtype_map[dtype_name],
            seed=args.seed,
            group_count=group_count,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            cooldown_s=args.bench_cooldown,
            profile_fwd=profile_fwd,
            profile_bwd=profile_bwd,
        )

    print("\n[speedup: FP8 raw attention kernel vs SDPA]")
    for dtype_name, base in baselines.items():
        parts = []
        if profile_fwd and "fwd_ms" in base and "fwd_ms" in fp8:
            parts.append(f"fwd {base['fwd_ms'] / fp8['fwd_ms']:.2f}x")
        if profile_bwd and "bwd_ms" in base and "bwd_ms" in fp8:
            parts.append(f"bwd {base['bwd_ms'] / fp8['bwd_ms']:.2f}x")
        print(f"  vs torch sdpa {dtype_name:<5} " + "  ".join(parts))

    if profile_fwd and "fwd_ms" in fp8:
        fwd_with_quant = fp8["quant_q_ms"] + fp8["quant_k_ms"] + fp8["fwd_ms"]
        print("\n[speedup: FP8 forward including Q/K token quantization]")
        for dtype_name, base in baselines.items():
            if "fwd_ms" in base:
                print(f"  vs torch sdpa {dtype_name:<5} fwd {base['fwd_ms'] / fwd_with_quant:.2f}x")


if __name__ == "__main__":
    main()
