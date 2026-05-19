"""Profile the standalone FP8 quantization CUDA kernels.

Sweeps over batch sizes and (longer) sequence lengths and reports
achieved bandwidth (GB/s) and elements/s for both the per-token (row)
and per-channel quantization paths exposed via the `_C` extension:

  - fp8_quantize_per_token_out (rowwise amax + scale)
  - fp8_quantize_per_channel_out (channelwise amax + scale)

Usage:
  python3 profile_quant.py
  python3 profile_quant.py --B 4 8 --N 4096 16384 --D 128
"""

import argparse
import gc
import sys

import torch

sys.path.insert(0, ".")  # for the in-tree _C extension

from fp8_suite.kernel_api import (
    cuda_quantize_per_channel_out,
    cuda_quantize_per_token_out,
    require_extension,
)
from fp8_suite.profiling import benchmark_ms, recommended_group_count, uniform_tensor


def make_groups(shape, *, seed, group_count, channel_scale_shape):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    B, H, N, D = shape
    groups = []
    for _ in range(group_count):
        x = uniform_tensor(shape, generator=gen)  # fp32
        xq = torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn)
        s_row = torch.empty((B, H, N), device="cuda", dtype=torch.float32)
        s_ch = torch.empty(channel_scale_shape, device="cuda", dtype=torch.float32)
        groups.append((x, xq, s_row, s_ch))
    torch.cuda.synchronize()
    return groups


def fmt_n(x):
    """Human-readable element count."""
    if x >= 1e9:
        return f"{x / 1e9:6.2f} G"
    if x >= 1e6:
        return f"{x / 1e6:6.2f} M"
    if x >= 1e3:
        return f"{x / 1e3:6.2f} K"
    return f"{x:6.0f}  "


def profile_one(shape, *, seed, warmup, iters, cooldown_s, group_count):
    B, H, N, D = shape
    elems = B * H * N * D
    # Traffic per launch:
    #   read  : 4 * elems  (fp32 in)
    #   write : 1 * elems  (fp8 out) + 4 * scale_count (fp32 scales)
    bytes_per_token_scale = 4 * (B * H * N)
    bytes_per_channel_scale = 4 * (B * H * D)
    token_bytes = 4 * elems + elems + bytes_per_token_scale
    channel_bytes = 4 * elems + elems + bytes_per_channel_scale

    groups = make_groups(
        shape,
        seed=seed,
        group_count=group_count,
        channel_scale_shape=(B, H, D),
    )

    def run_token(i):
        x, xq, s_row, _ = groups[i]
        cuda_quantize_per_token_out(x, xq, s_row)

    def run_channel(i):
        x, xq, _, s_ch = groups[i]
        cuda_quantize_per_channel_out(x, xq, s_ch)

    tok_ms = benchmark_ms(run_token, group_count, warmup=warmup, iters=iters, cooldown_s=cooldown_s)
    ch_ms = benchmark_ms(run_channel, group_count, warmup=warmup, iters=iters, cooldown_s=cooldown_s)

    tok_gbps = token_bytes / (tok_ms * 1e-3) / (1024 ** 3)
    ch_gbps = channel_bytes / (ch_ms * 1e-3) / (1024 ** 3)
    tok_eps = elems / (tok_ms * 1e-3)
    ch_eps = elems / (ch_ms * 1e-3)

    print(
        f"  B={B:<3} H={H:<3} N={N:<6} D={D:<4} "
        f"elems={fmt_n(elems)} "
        f"| per-token: {tok_ms:7.4f} ms  {tok_gbps:7.1f} GB/s  {fmt_n(tok_eps)}elem/s "
        f"| per-channel: {ch_ms:7.4f} ms  {ch_gbps:7.1f} GB/s  {fmt_n(ch_eps)}elem/s"
    )

    # Free for next config.
    del groups
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "token_ms": tok_ms, "channel_ms": ch_ms,
        "token_gbps": tok_gbps, "channel_gbps": ch_gbps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--H", type=int, nargs="+", default=[16])
    p.add_argument("--N", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    p.add_argument("--D", type=int, nargs="+", default=[128])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bench-iters", type=int, default=100)
    p.add_argument("--bench-warmup", type=int, default=50)
    p.add_argument("--bench-cooldown", type=float, default=0.05)
    p.add_argument("--bench-groups", type=int, default=None,
                   help="If unset, picks enough groups to exceed L2.")
    args = p.parse_args()

    require_extension("fp8_quantize_per_token_out", "fp8_quantize_per_channel_out")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"Device: {props.name}  ({props.multi_processor_count} SMs, "
          f"L2={getattr(props, 'l2_cache_size', 0) // (1024*1024)} MB)")
    print(f"Iters per measurement: {args.bench_iters}  (warmup {args.bench_warmup})\n")

    print("[FP8 quantization throughput] (input fp32 -> fp8 + fp32 scale)")
    for D in args.D:
        if D not in (64, 128):
            print(f"  skipping D={D} (kernel supports 64 or 128 only)")
            continue
        for B in args.B:
            for H in args.H:
                for N in args.N:
                    shape = (B, H, N, D)
                    bytes_per_group = 5 * B * H * N * D  # fp32 in + fp8 out
                    group_count = args.bench_groups or recommended_group_count(bytes_per_group)
                    try:
                        profile_one(
                            shape,
                            seed=args.seed,
                            warmup=args.bench_warmup,
                            iters=args.bench_iters,
                            cooldown_s=args.bench_cooldown,
                            group_count=group_count,
                        )
                    except torch.cuda.OutOfMemoryError:
                        print(f"  B={B} H={H} N={N} D={D} : OOM, skipping")
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
