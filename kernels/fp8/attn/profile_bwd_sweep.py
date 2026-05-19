"""In-depth sweep of the FP8 attention backward kernel vs bf16 SDPA bwd-only.

The default backward recipe in fp8_suite.recipe builds an O(N^2) reference
attention matrix inside `estimate_dS_descale` to set the dS descale. That
extra Python compute OOMs above ~32k. The kernel itself has no such
limitation: it just consumes a per-row descale tensor. To profile the
kernel across long-context shapes without OOM we monkey-patch
`estimate_dS_descale` to a fast HBM-only heuristic that returns a constant
fp32 scalar. This affects numerical accuracy, not throughput.

For every shape we report:
  - FP8 bwd ms / TFLOP/s
  - bf16 SDPA bwd-only ms / TFLOP/s (FlashAttention backend)
  - speedup FP8 vs bf16 SDPA

We sweep B, H, N, D to find any shape where FP8 bwd beats bf16 SDPA bwd.
"""

import argparse
import gc

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# ---- Monkey-patch estimate_dS_descale to avoid O(N^2) reference compute ----
# Must happen before fp8_suite.recipe is imported by anything else.
import fp8_suite.recipe as _recipe


def _fast_estimate_dS_descale(Qq, Kq, Vq, dOq, sq, sk, sv, sdo_descale):
    # Heuristic: peak |dS| ~ peak|P| * (peak|dP| + small) * 1/sqrt(D).
    # We use a constant midpoint that keeps the kernel numerically stable
    # for benchmarking. Throughput does not depend on this value.
    inv_sqrt_d = 1.0 / (Qq.shape[-1] ** 0.5)
    val = torch.tensor(0.05 * inv_sqrt_d, device=Qq.device, dtype=torch.float32)
    return val


_recipe.estimate_dS_descale = _fast_estimate_dS_descale


from fp8_suite.kernel_api import fp8_backward, fp8_forward, require_extension
from fp8_suite.profiling import benchmark_ms, print_profile_line, recommended_group_count, uniform_tensor
from fp8_suite.recipe import prepare_backward_inputs, prepare_forward_inputs


# Shapes are chosen to span:
#   - the moderate-context regime where the recipe could already run (≤ 32k)
#   - the long-context regime that was OOM-blocked
#   - varying head count (H=4,8,16) and head-dim (64, 128)
# All N values are multiples of 384 (kernel tiling constraint).
SHAPES = [
    # D=128
    (1,  8, 1536,  128),
    (2, 16, 3072,  128),
    (4, 16, 3072,  128),
    (8, 16, 3072,  128),
    (4,  8, 8064,  128),
    (4,  4, 16128, 128),
    (4,  2, 32256, 128),
    (2,  4, 57600, 128),
    (4,  4, 57600, 128),
    (4,  2, 71808, 128),
    (4,  1, 144000, 128),
    # D=64
    (4, 16, 3072,  64),
    (4,  8, 8064,  64),
    (4,  4, 16128, 64),
    (2,  4, 57600, 64),
]


def _hbm_peak_gbps():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    bus = getattr(props, "memory_bus_width", 0) or 0
    clk = getattr(props, "memory_clock_rate", 0) or 0
    if bus > 0 and clk > 0:
        return (bus * clk * 2.0 / 8.0) / 1.0e6
    name = props.name.lower()
    if "h200" in name:
        return 4480.0
    if "h100" in name and "pcie" in name:
        return 2000.0
    if "h100" in name:
        return 3350.0
    return 2000.0


def build_fp8_bwd_groups(shape, *, seed, group_count):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    groups = []
    for _ in range(group_count):
        Q = uniform_tensor(shape, generator=gen)
        K = uniform_tensor(shape, generator=gen)
        V = uniform_tensor(shape, generator=gen)
        dO = uniform_tensor(shape, generator=gen)
        fwd = prepare_forward_inputs(Q, K, V, use_cuda_quant=True)
        O0 = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        L0 = torch.empty((*shape[:-1], 1), dtype=torch.float32, device="cuda")
        bwd_seed = prepare_backward_inputs(fwd, O0, L0, dO, sr_dO=True)
        fwd.Vbf = (bwd_seed.Vq_ch.to(torch.float32) * bwd_seed.sv_ch.unsqueeze(-2)).to(torch.bfloat16).contiguous()
        O, L = fp8_forward(fwd)
        bwd = prepare_backward_inputs(fwd, O, L, dO, sr_dO=True)
        del Q, K, V, dO, O0, L0, O, L
        groups.append(bwd)
    torch.cuda.synchronize()
    return groups


def build_sdpa_groups(shape, *, seed, group_count):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    groups = []
    for _ in range(group_count):
        Q = uniform_tensor(shape, generator=gen).to(torch.bfloat16).requires_grad_(True)
        K = uniform_tensor(shape, generator=gen).to(torch.bfloat16).requires_grad_(True)
        V = uniform_tensor(shape, generator=gen).to(torch.bfloat16).requires_grad_(True)
        dO = uniform_tensor(shape, generator=gen).to(torch.bfloat16)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            O = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        groups.append((Q, K, V, dO, O))
    torch.cuda.synchronize()
    return groups


def profile_shape(shape, *, seed, warmup, iters, cooldown, summary):
    B, H, N, D = shape
    print(f"\n[bwd profile B={B} H={H} N={N} D={D} seed={seed}]")
    fp8_bytes = B * H * N * D * 6 + B * H * N * 4
    sdpa_bytes = B * H * N * D * 4 * 2
    fp8_g = recommended_group_count(fp8_bytes)
    sdpa_g = recommended_group_count(sdpa_bytes)

    try:
        fp8_groups = build_fp8_bwd_groups(shape, seed=seed, group_count=fp8_g)
        fp8_ms = benchmark_ms(
            lambda i: fp8_backward(fp8_groups[i], fp8_dS_mode=2),
            fp8_g, warmup=warmup, iters=iters, cooldown_s=cooldown,
        )
        print_profile_line("fp8 attention bwd", fp8_ms, flops_shape=shape, kind="bwd")
        del fp8_groups
        gc.collect()
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        print(f"  FP8 bwd OOM: {str(e).splitlines()[0]}")
        torch.cuda.empty_cache()
        return

    try:
        sdpa_groups = build_sdpa_groups(shape, seed=seed, group_count=sdpa_g)
        def _bwd(i):
            Q, K, V, dO, O = sdpa_groups[i]
            return torch.autograd.grad(O, (Q, K, V), dO, retain_graph=True, create_graph=False)
        sdpa_ms = benchmark_ms(_bwd, sdpa_g, warmup=warmup, iters=iters, cooldown_s=cooldown)
        print_profile_line("torch sdpa bfloat16 bwd-only", sdpa_ms, flops_shape=shape, kind="bwd")
        del sdpa_groups
        gc.collect()
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        print(f"  SDPA bwd OOM: {str(e).splitlines()[0]}")
        torch.cuda.empty_cache()
        return

    speedup = sdpa_ms / fp8_ms
    from fp8_suite.metrics import attention_bwd_tflops
    fp8_tf = attention_bwd_tflops(B, H, N, D, fp8_ms)
    sdpa_tf = attention_bwd_tflops(B, H, N, D, sdpa_ms)
    print(f"  speedup FP8 vs bf16 SDPA bwd: {speedup:.2f}x")
    summary.append((shape, fp8_ms, fp8_tf, sdpa_ms, sdpa_tf, speedup))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-warmup", type=int, default=20)
    ap.add_argument("--bench-iters", type=int, default=15)
    ap.add_argument("--bench-cooldown", type=float, default=0.2)
    args = ap.parse_args()
    require_extension("fp8_mha_forward", "fp8_mha_backward")
    print(f"HBM peak = {_hbm_peak_gbps():.0f} GB/s  ({torch.cuda.get_device_properties(0).name})")
    print(f"protocol: uniform[-1,1], warmup={args.bench_warmup}, iters={args.bench_iters}, "
          f"sdp_descale heuristic (kernel throughput unaffected)")

    summary = []
    for shape in SHAPES:
        profile_shape(shape, seed=args.seed,
                      warmup=args.bench_warmup, iters=args.bench_iters,
                      cooldown=args.bench_cooldown, summary=summary)

    print("\n=== FP8 backward summary ===")
    print(f"{'shape':<26} {'fp8 ms':>9} {'fp8 TF/s':>9}  {'sdpa ms':>9} {'sdpa TF/s':>10}  {'speedup':>8}")
    for shape, fp8_ms, fp8_tf, sdpa_ms, sdpa_tf, sp in summary:
        s = f"B={shape[0]} H={shape[1]} N={shape[2]} D={shape[3]}"
        print(f"{s:<26} {fp8_ms:9.3f} {fp8_tf:9.2f}  {sdpa_ms:9.3f} {sdpa_tf:10.2f}  {sp:7.2f}x")

    wins = [r for r in summary if r[-1] >= 1.0]
    print(f"\nShapes where FP8 bwd ≥ bf16 SDPA bwd: {len(wins)} / {len(summary)}")
    if wins:
        for shape, fp8_ms, fp8_tf, sdpa_ms, sdpa_tf, sp in wins:
            print(f"  B={shape[0]} H={shape[1]} N={shape[2]} D={shape[3]}  {sp:.2f}x  ({fp8_tf:.1f} vs {sdpa_tf:.1f} TFLOP/s)")
    best = max(summary, key=lambda r: r[2]) if summary else None
    if best:
        s = best[0]
        print(f"Best FP8 bwd absolute: {best[2]:.2f} TFLOP/s @ B={s[0]} H={s[1]} N={s[2]} D={s[3]}")
    best_sp = max(summary, key=lambda r: r[-1]) if summary else None
    if best_sp:
        s = best_sp[0]
        print(f"Best FP8 bwd speedup:  {best_sp[-1]:.2f}x @ B={s[0]} H={s[1]} N={s[2]} D={s[3]}")


if __name__ == "__main__":
    main()
