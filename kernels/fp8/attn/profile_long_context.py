"""Profile FP8 attention vs bf16 SDPA at long sequence lengths.

Uses the same benchmarking protocol as profile_fp8.py / docs/BENCHMARKING.md:
  - bitwise-identical uniform[-1, 1] float32 inputs (uniform_tensor)
  - automatic input groups sized to exceed 3x L2 cache when needed
  - 500 warmup launches, 100 measured launches (overridable)
  - two CUDA events around the full measured launch loop
  - no synchronization between measured launches
  - short cooldown between benchmarked kernels (cooldown_s = 0.2)

Sequence lengths approximate the targets:
  57600, 72000, 144000, 162000, 324000
all rounded to multiples of 384 (LCM of forward 3*qo_height=192 and
kv_height=128 tiles, so the existing kernel's tiling assertions hold).

We use the existing fp8_suite.profiling helpers (benchmark_ms,
recommended_group_count, uniform_tensor, print_profile_line) so the numbers
are directly comparable to profile_fp8.py's output for the shorter shapes
already in profile_fp8_runs.sh.
"""

import argparse
import gc

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from fp8_suite.kernel_api import (
    cuda_quantize_per_channel_out,
    cuda_quantize_per_token_out,
    fp8_backward,
    fp8_forward,
    require_extension,
)
from fp8_suite.metrics import gbps
from fp8_suite.profiling import (
    benchmark_ms,
    print_profile_line,
    recommended_group_count,
    uniform_tensor,
)
from fp8_suite.recipe import prepare_backward_inputs, prepare_forward_inputs


# (B, H, N, D) - H is chosen so the working set fits on an 80 GB H100 PCIe.
# B sweep included so we can see how the speedup behaves as wave count grows.
LONG_SHAPES = [
    # ≈57600
    (1, 4,  57600, 128),
    (2, 4,  57600, 128),
    (4, 4,  57600, 128),
    (8, 2,  57600, 128),

    # ≈72000
    (1, 2,  71808, 128),
    (2, 2,  71808, 128),
    (4, 2,  71808, 128),
    (8, 1,  71808, 128),

    # ≈144000
    (1, 2, 144000, 128),
    (2, 2, 144000, 128),
    (4, 1, 144000, 128),
    (8, 1, 144000, 128),

    # ≈162000
    (1, 1, 162048, 128),
    (2, 1, 162048, 128),
    (4, 1, 162048, 128),
    (8, 1, 162048, 128),

    # ≈324000
    (1, 1, 323712, 128),
    (2, 1, 323712, 128),
    (4, 1, 323712, 128),
]


def _all_bwd_sweep_shapes():
    """Comprehensive backward sweep. All N values are multiples of 384."""
    shapes = []
    short_n = (384, 768, 1536, 3072)
    mod_n = (8064, 16128, 32256)
    long_n = (57600, 71808, 144000)

    for D in (128, 64):
        for B in (1, 2, 4, 8):
            for H in (1, 2, 4, 8, 16):
                for N in short_n:
                    shapes.append((B, H, N, D))
        for B in (1, 2, 4):
            for H in (1, 2, 4, 8):
                for N in mod_n:
                    shapes.append((B, H, N, D))
        for B in (1, 2, 4):
            for H in (1, 2, 4):
                shapes.append((B, H, 57600, D))
        for N in long_n:
            for H in (1, 2):
                shapes.append((1, H, N, D))
    return shapes


def _quick_bwd_sweep_shapes():
    return [
        (1, 8, 1536, 128),
        (1, 8, 1536, 64),
        (2, 16, 1536, 128),
        (4, 8, 3072, 128),
        (1, 4, 16128, 128),
        (2, 2, 16128, 64),
    ]


SDPA_BWD_PEAK_SHAPES = [
    (8, 16, 256, 128), (8, 16, 512, 128), (8, 16, 512, 256),
    (16, 16, 256, 128), (16, 16, 512, 128), (16, 8, 512, 256),
    (4, 16, 1024, 128), (4, 16, 1024, 256),
    (8, 16, 1024, 128), (8, 8, 1024, 256),
    (2, 16, 2048, 128), (2, 16, 2048, 256),
    (4, 16, 2048, 128), (4, 8, 2048, 256),
    (8, 16, 2048, 128), (8, 8, 2048, 256),
    (1, 16, 4096, 128), (2, 8, 4096, 128), (2, 16, 4096, 128),
    (4, 8, 4096, 128), (4, 16, 4096, 128), (4, 8, 4096, 256),
    (2, 16, 8192, 128), (4, 8, 8192, 128), (4, 16, 8192, 128),
    (2, 8, 16384, 128), (2, 16, 16384, 128), (4, 8, 16384, 128),
]


def parse_shapes(shapes_str):
    """Parse 'B,H,N,D;B,H,N,D' into shape tuples."""
    return [tuple(int(x) for x in s.split(',')) for s in shapes_str.split(';') if s]


# HBM peak in GB/s, queried at runtime from the actual device so SOL is
# always correct (H100 PCIe ≈ 2000, H100 SXM ≈ 3350, H200 ≈ 4480).
def _hbm_peak_gbps():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    # memoryBusWidth (bits) × memoryClockRate (kHz, effective) × 2 / 8 / 1e6 → GB/s
    bus = getattr(props, "memory_bus_width", 0) or 0
    clk = getattr(props, "memory_clock_rate", 0) or 0
    if bus > 0 and clk > 0:
        return (bus * clk * 2.0 / 8.0) / 1.0e6
    # Fallback: hard-coded per known device.
    name = props.name.lower()
    if "h200" in name:
        return 4480.0
    if "h100" in name and "pcie" in name:
        return 2000.0
    if "h100" in name:
        return 3350.0
    return 2000.0


HBM_PEAK_GBPS = _hbm_peak_gbps()


# ---------------------------------------------------------------------------
# FP8 input groups (forward-only — the backward recipe materialises an O(N^2)
# reference attention matrix in estimate_dS_descale, which OOMs above ~32k).
# ---------------------------------------------------------------------------

def make_fp8_groups_fwd_only(shape, *, seed, group_count):
    """Build group_count groups of FP8-prepared forward inputs. Same input
    distribution as profile_fp8.profile_sdpa (uniform[-1,1] fp32 → cast)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    groups = []
    for _ in range(group_count):
        Q = uniform_tensor(shape, generator=gen)
        K = uniform_tensor(shape, generator=gen)
        V = uniform_tensor(shape, generator=gen)
        fwd_inp = prepare_forward_inputs(Q, K, V, use_cuda_quant=True)
        del Q, K, V
        groups.append(fwd_inp)
    torch.cuda.synchronize()
    return groups


def make_fp8_groups_full(shape, *, seed, group_count, sdp_descale_mode="estimate"):
    """Build group_count groups of FP8-prepared fwd+bwd inputs. Only safe at
    moderate N (the recipe runs an O(N^2) reference SDPA on Q/K/V)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    groups = []
    for _ in range(group_count):
        Q = uniform_tensor(shape, generator=gen)
        K = uniform_tensor(shape, generator=gen)
        V = uniform_tensor(shape, generator=gen)
        dO = uniform_tensor(shape, generator=gen)

        fwd_inp = prepare_forward_inputs(Q, K, V, use_cuda_quant=True)
        O0, L0 = fp8_forward(fwd_inp)
        bwd_inp = prepare_backward_inputs(
            fwd_inp, O0, L0, dO, sr_dO=True,
            sdp_descale_mode=sdp_descale_mode,
        )
        fwd_inp.Vbf = (bwd_inp.Vq_ch.to(torch.float32) * bwd_inp.sv_ch.unsqueeze(-2)).to(torch.bfloat16).contiguous()
        O, L = fp8_forward(fwd_inp)
        bwd_inp = prepare_backward_inputs(
            fwd_inp, O, L, dO, sr_dO=True,
            sdp_descale_mode=sdp_descale_mode,
        )

        del Q, K, V, dO, O0, L0, O, L
        groups.append((fwd_inp, bwd_inp))
    torch.cuda.synchronize()
    return groups


# ---------------------------------------------------------------------------
# bf16 SDPA input groups (FlashAttention backend, same distribution as fp8).
# ---------------------------------------------------------------------------

def make_sdpa_groups(shape, *, seed, group_count):
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


# ---------------------------------------------------------------------------
# Top-level profile.
# ---------------------------------------------------------------------------

def profile_quant(shape, *, seed, warmup, iters, cooldown, group_count):
    """Profile the three FP8 input-quantization kernels (Q token, K token,
    V channel) and print bandwidth + % of HBM SOL."""
    B, H, N, D = shape
    gen = torch.Generator(device="cuda").manual_seed(seed)
    groups = []
    for _ in range(group_count):
        Q = uniform_tensor(shape, generator=gen)
        Ks = uniform_tensor(shape, generator=gen)   # K_s := K - mean_K
        Vs = uniform_tensor(shape, generator=gen)   # V_s := V - mean_V
        groups.append({
            "Q": Q, "Ks": Ks, "Vs": Vs,
            "q_out": torch.empty_like(Q, dtype=torch.float8_e4m3fn),
            "q_scale": torch.empty(shape[:3], device="cuda", dtype=torch.float32),
            "k_out": torch.empty_like(Ks, dtype=torch.float8_e4m3fn),
            "k_scale": torch.empty(shape[:3], device="cuda", dtype=torch.float32),
            "v_out": torch.empty_like(Vs, dtype=torch.float8_e4m3fn),
            "v_scale": torch.empty((shape[0], shape[1], shape[3]), device="cuda", dtype=torch.float32),
        })
    torch.cuda.synchronize()

    # Bytes moved: read fp32 (4 B/elem) + write fp8 (1 B/elem) + write scale.
    token_bytes = B * H * N * D * 5 + B * H * N * 4
    channel_bytes = B * H * N * D * 5 + B * H * D * 4

    def _sol(ms, b):
        return gbps(b, ms) / HBM_PEAK_GBPS * 100.0

    qms = benchmark_ms(
        lambda i: cuda_quantize_per_token_out(groups[i]["Q"], groups[i]["q_out"], groups[i]["q_scale"]),
        group_count, warmup=warmup, iters=iters, cooldown_s=cooldown)
    print(f"  quant Q token                {qms:8.4f} ms  {gbps(token_bytes, qms):7.1f} GB/s  {_sol(qms, token_bytes):5.1f}% SOL")

    kms = benchmark_ms(
        lambda i: cuda_quantize_per_token_out(groups[i]["Ks"], groups[i]["k_out"], groups[i]["k_scale"]),
        group_count, warmup=warmup, iters=iters, cooldown_s=cooldown)
    print(f"  quant K token                {kms:8.4f} ms  {gbps(token_bytes, kms):7.1f} GB/s  {_sol(kms, token_bytes):5.1f}% SOL")

    vms = benchmark_ms(
        lambda i: cuda_quantize_per_channel_out(groups[i]["Vs"], groups[i]["v_out"], groups[i]["v_scale"]),
        group_count, warmup=warmup, iters=iters, cooldown_s=cooldown)
    print(f"  quant V channel              {vms:8.4f} ms  {gbps(channel_bytes, vms):7.1f} GB/s  {_sol(vms, channel_bytes):5.1f}% SOL")

    del groups
    gc.collect()
    torch.cuda.empty_cache()
    return qms, kms, vms


def profile_one(shape, *, seed, warmup, iters, cooldown, fwd_only,
                bwd_modes=(2,), sdp_descale_mode="estimate",
                include_sdpa_bwd=True):
    B, H, N, D = shape
    print(f"\n[profile B={B} H={H} N={N} D={D} seed={seed}]")

    # Group count: target 3x L2 of FP8 working set per group (fp8 Q+K + bf16 V
    # + bf16 O ≈ 6 bytes/elem total, plus L).
    bytes_per_group = B * H * N * D * 6 + B * H * N * 4
    fp8_groups_n = recommended_group_count(bytes_per_group)
    sdpa_bytes_per_group = B * H * N * D * 4 * 2   # bf16 Q,K,V,dO,O ≈ 4×2 B/elem
    sdpa_groups_n = recommended_group_count(sdpa_bytes_per_group)
    print(f"  benchmark protocol: uniform[-1,1], fp8 groups={fp8_groups_n}, "
          f"sdpa groups={sdpa_groups_n}, warmup={warmup}, iters={iters}, "
          f"cooldown={cooldown:.2f}s, mode={'fwd-only' if fwd_only else 'fwd+bwd'}, "
          f"sdp_descale={sdp_descale_mode}, "
          f"hbm SOL={HBM_PEAK_GBPS:.0f} GB/s")

    # ---- Quantization overhead ----------------------------------------------
    try:
        profile_quant(shape, seed=seed, warmup=warmup, iters=iters,
                      cooldown=cooldown, group_count=fp8_groups_n)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  quant OOM: {str(e).splitlines()[0]}")
        torch.cuda.empty_cache()

    # ---- FP8 fwd (and optional bwd) -----------------------------------------
    try:
        if fwd_only:
            fp8_groups = make_fp8_groups_fwd_only(shape, seed=seed, group_count=fp8_groups_n)
            fwd_ms = benchmark_ms(
                lambda i: fp8_forward(fp8_groups[i]), fp8_groups_n,
                warmup=warmup, iters=iters, cooldown_s=cooldown,
            )
            print_profile_line("fp8 attention fwd", fwd_ms, flops_shape=shape, kind="fwd")
            bwd_ms = None
        else:
            fp8_groups = make_fp8_groups_full(
                shape, seed=seed, group_count=fp8_groups_n,
                sdp_descale_mode=sdp_descale_mode,
            )
            fwd_ms = benchmark_ms(
                lambda i: fp8_forward(fp8_groups[i][0]), fp8_groups_n,
                warmup=warmup, iters=iters, cooldown_s=cooldown,
            )
            print_profile_line("fp8 attention fwd", fwd_ms, flops_shape=shape, kind="fwd")
            bwd_ms = None
            for mode in bwd_modes:
                label = {0: "bf16-dS", 1: "RTNE-dS", 2: "SR-dS"}.get(mode, f"dS={mode}")
                bwd_ms = benchmark_ms(
                    lambda i, m=mode: fp8_backward(fp8_groups[i][1], fp8_dS_mode=m),
                    fp8_groups_n, warmup=warmup, iters=iters, cooldown_s=cooldown,
                )
                print_profile_line(f"fp8 attention bwd ({label})", bwd_ms, flops_shape=shape, kind="bwd")
        del fp8_groups
        gc.collect()
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        print(f"  FP8 OOM: {str(e).splitlines()[0]}")
        torch.cuda.empty_cache()
        return

    # ---- bf16 SDPA baselines ------------------------------------------------
    # In fwd-only mode we have no FP8 bwd to compare against, and at long N
    # the SDPA backward dominates the entire sweep (700 ms - 1.1 s per launch
    # × (warmup+iters) per shape). Skip it in that case to keep the sweep
    # tractable. Force-enable with --include-sdpa-bwd.
    try:
        sdpa_groups = make_sdpa_groups(shape, seed=seed, group_count=sdpa_groups_n)
        def sdpa_fwd(i):
            Q, K, V, _, _ = sdpa_groups[i]
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                return F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        def sdpa_bwd(i):
            Q, K, V, dO, O = sdpa_groups[i]
            return torch.autograd.grad(O, (Q, K, V), dO, retain_graph=True, create_graph=False)

        sdpa_fwd_ms = benchmark_ms(sdpa_fwd, sdpa_groups_n,
                                   warmup=warmup, iters=iters, cooldown_s=cooldown)
        print_profile_line("torch sdpa bfloat16 fwd", sdpa_fwd_ms, flops_shape=shape, kind="fwd")
        sdpa_bwd_ms = None
        if include_sdpa_bwd and bwd_ms is not None:
            sdpa_bwd_ms = benchmark_ms(sdpa_bwd, sdpa_groups_n,
                                       warmup=warmup, iters=iters, cooldown_s=cooldown)
            print_profile_line("torch sdpa bfloat16 bwd-only", sdpa_bwd_ms, flops_shape=shape, kind="bwd")
        del sdpa_groups
        gc.collect()
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        print(f"  SDPA bf16 OOM: {str(e).splitlines()[0]}")
        torch.cuda.empty_cache()
        return

    # ---- Speedup summary ----------------------------------------------------
    line = f"  speedup vs bf16 SDPA: fwd {sdpa_fwd_ms / fwd_ms:.2f}x"
    if bwd_ms is not None and sdpa_bwd_ms is not None:
        line += f"  bwd {sdpa_bwd_ms / bwd_ms:.2f}x"
    print(line)


def profile_sdpa_bwd_peak(shapes, *, seed, warmup, iters, cooldown):
    """Sweep bf16 SDPA backward-only throughput across baseline shapes."""
    from fp8_suite.metrics import attention_bwd_tflops

    print(f"Device: {torch.cuda.get_device_name(0)}  SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    results = []
    for shape in shapes:
        B, H, N, D = shape
        try:
            bytes_per_group = B * H * N * D * 4 * 2
            group_count = recommended_group_count(bytes_per_group)
            groups = make_sdpa_groups(shape, seed=seed, group_count=group_count)
            def _bwd(i):
                Q, K, V, dO, O = groups[i]
                return torch.autograd.grad(O, (Q, K, V), dO, retain_graph=True, create_graph=False)
            ms = benchmark_ms(_bwd, group_count, warmup=warmup, iters=iters, cooldown_s=cooldown)
            tf = attention_bwd_tflops(B, H, N, D, ms)
            results.append((shape, ms, tf))
            print(f"\n[sdpa bwd B={B} H={H} N={N} D={D} seed={seed} groups={group_count}]")
            print_profile_line("torch sdpa bfloat16 bwd-only", ms, flops_shape=shape, kind="bwd")
            del groups
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  B={B} H={H} N={N} D={D} | SDPA bwd OOM")
            torch.cuda.empty_cache()

    results.sort(key=lambda r: r[2], reverse=True)
    if results:
        shape, ms, tf = results[0]
        print(f"\nPeak SDPA bf16 bwd: {tf:.2f} TFLOP/s @ B={shape[0]} H={shape[1]} N={shape[2]} D={shape[3]} ({ms:.4f} ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("long", "bwd-sweep", "sdpa-bwd-peak"), default="long",
                    help="long: canonical long-context profile; bwd-sweep: backward dS-mode sweep; sdpa-bwd-peak: bf16 SDPA bwd baseline sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-warmup", type=int, default=500)
    ap.add_argument("--bench-iters", type=int, default=100)
    ap.add_argument("--bench-cooldown", type=float, default=0.2)
    ap.add_argument("--quick-profile", action="store_true",
                    help="Use low warmup/iteration counts for quick local timing; not for final reporting")
    ap.add_argument("--bwd-threshold", type=int, default=32000,
                    help="run fwd+bwd up to this N, fwd-only above")
    ap.add_argument("--bwd-modes", type=int, nargs="+", default=None,
                    help="fp8_dS modes to benchmark: 0=bf16/off, 1=RTNE, 2=SR")
    ap.add_argument("--bwd-sdp-descale-mode", choices=("estimate", "constant"), default=None,
                    help="constant avoids the O(N^2) Python dS range estimate; timing-only, not correctness-quality")
    ap.add_argument("--quick", action="store_true", help="quick subset for --mode bwd-sweep")
    ap.add_argument("--case", type=int, default=None, help="only run this shape index in the selected sweep")
    ap.add_argument("--shapes", type=str, default=None,
                    help="custom shapes as 'B,H,N,D;B,H,N,D'")
    ap.add_argument("--skip-sdpa-bwd", action="store_true",
                    help="skip bf16 SDPA backward baseline in bwd-capable modes")
    args = ap.parse_args()
    if args.quick_profile:
        args.bench_warmup = 20
        args.bench_iters = 15
        print("[profile note] --quick-profile uses warmup=20, iters=15; not for final reported numbers.")

    require_extension("fp8_mha_forward", "fp8_mha_backward")
    if args.shapes:
        shapes = parse_shapes(args.shapes)
    elif args.mode == "bwd-sweep":
        shapes = _quick_bwd_sweep_shapes() if args.quick else _all_bwd_sweep_shapes()
    elif args.mode == "sdpa-bwd-peak":
        shapes = SDPA_BWD_PEAK_SHAPES
    else:
        shapes = LONG_SHAPES

    if args.case is not None:
        if args.case < 0 or args.case >= len(shapes):
            raise SystemExit(f"--case must be in [0, {len(shapes) - 1}]")
        shapes = [shapes[args.case]]

    if args.mode == "sdpa-bwd-peak":
        profile_sdpa_bwd_peak(
            shapes,
            seed=args.seed,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            cooldown=args.bench_cooldown,
        )
        return

    default_bwd_modes = (0, 1, 2) if args.mode == "bwd-sweep" else (2,)
    bwd_modes = tuple(args.bwd_modes or default_bwd_modes)
    sdp_descale_mode = args.bwd_sdp_descale_mode or ("constant" if args.mode == "bwd-sweep" else "estimate")
    if args.mode == "bwd-sweep" and sdp_descale_mode == "constant":
        print("[profile note] bwd-sweep uses constant sdp descale by default; "
              "timing-only, not correctness-quality.")

    for shape in shapes:
        fwd_only = args.mode == "long" and shape[2] > args.bwd_threshold
        profile_one(
            shape,
            seed=args.seed,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            cooldown=args.bench_cooldown,
            fwd_only=fwd_only,
            bwd_modes=bwd_modes,
            sdp_descale_mode=sdp_descale_mode,
            include_sdpa_bwd=not args.skip_sdpa_bwd,
        )


if __name__ == "__main__":
    main()
