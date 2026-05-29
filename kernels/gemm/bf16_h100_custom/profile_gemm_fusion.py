from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F

import _C
from tk_bench import input_group_count, print_bench, profile_groups, uniform_bf16


def parse_shape_pair(value: str) -> tuple[int, int]:
    for sep in ("x", ":", ","):
        if sep not in value:
            continue
        batch_s, tokens_s = value.split(sep, 1)
        return int(batch_s), int(tokens_s)
    raise argparse.ArgumentTypeError(f"shape must be BxT, B:T, or B,T; got {value!r}")


def make_groups(
    batch: int,
    tokens: int,
    k: int,
    n: int,
    seed: int,
) -> list[tuple[torch.Tensor, ...]]:
    m = batch * tokens
    bytes_per_group = 2 * (m * k + n * k + n + m * n * 4 + batch * n)
    groups_n = min(input_group_count(bytes_per_group), 4)
    groups = []
    for idx in range(groups_n):
        group_seed = seed + idx * 20
        a = uniform_bf16((m, k), group_seed, -1.0, 1.0)
        w = uniform_bf16((n, k), group_seed + 1, -0.02, 0.02)
        b = uniform_bf16((n,), group_seed + 2, -0.02, 0.02)
        residual = uniform_bf16((m, n), group_seed + 3, -1.0, 1.0)
        gate = uniform_bf16((batch, n), group_seed + 4, -0.5, 0.5)
        native_out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        fused_out = torch.empty_like(native_out)
        projected = torch.empty_like(native_out)
        groups.append((a, w, b, residual, gate, native_out, fused_out, projected))
    return groups


def torch_linear(group: tuple[torch.Tensor, ...]) -> torch.Tensor:
    a, w, b, *_ = group
    return F.linear(a, w, b)


def torch_linear_gated(group: tuple[torch.Tensor, ...], tokens: int) -> torch.Tensor:
    a, w, b, residual, gate, *_ = group
    projected = F.linear(a, w, b)
    return residual + gate.repeat_interleave(tokens, dim=0) * projected


def tk_native_linear(group: tuple[torch.Tensor, ...]) -> torch.Tensor:
    a, w, b, _residual, _gate, native_out, *_ = group
    _C.gemm_linear_native(a, w, native_out, b)
    return native_out


def tk_fused_linear_gated(group: tuple[torch.Tensor, ...], tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    a, w, b, residual, gate, _native_out, fused_out, projected = group
    _C.gemm_linear_gated_residual(a, w, residual, gate, fused_out, projected, b, tokens)
    return fused_out, projected


def tk_fused_linear_gated_out(group: tuple[torch.Tensor, ...], tokens: int) -> torch.Tensor:
    a, w, b, residual, gate, _native_out, fused_out, _projected = group
    _C.gemm_linear_gated_residual_out(a, w, residual, gate, fused_out, b, tokens)
    return fused_out


def check_outputs(label: str, group: tuple[torch.Tensor, ...], tokens: int) -> None:
    torch_out = torch_linear(group)
    native_out = tk_native_linear(group)
    fused_out, projected = tk_fused_linear_gated(group, tokens)
    fused_out_only = tk_fused_linear_gated_out(group, tokens)
    _a, _w, _b, residual, gate, *_ = group
    expected_fused = residual + gate.repeat_interleave(tokens, dim=0) * torch_out
    torch.cuda.synchronize()

    for name, actual, expected in (
        ("native", native_out, torch_out),
        ("fused_projected", projected, torch_out),
        ("fused_out", fused_out, expected_fused),
        ("fused_out_only", fused_out_only, expected_fused),
    ):
        diff = (actual.float() - expected.float()).abs()
        print(f"  correctness {label} {name}: max={diff.max().item():.6g} mean={diff.mean().item():.6g}", flush=True)


def bench_one(
    label: str,
    batch: int,
    tokens: int,
    k: int,
    n: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    m = batch * tokens
    print(f"\n=== {label} B={batch} T={tokens} M={m} N={n} K={k} ===", flush=True)
    groups = make_groups(batch, tokens, k, n, 71000 + batch + tokens + k + n)
    check_outputs(label, groups[0], tokens)
    flops = 2.0 * m * n * k

    compiled_linear: Callable[[tuple[torch.Tensor, ...]], torch.Tensor] = torch.compile(torch_linear, dynamic=False)
    compiled_linear_gated: Callable[[tuple[torch.Tensor, ...]], torch.Tensor] = torch.compile(
        lambda group: torch_linear_gated(group, tokens),
        dynamic=False,
    )
    results = {}
    for name, fn in (
        ("torch_linear", torch_linear),
        ("torch_compile_linear", compiled_linear),
        ("torch_linear_gated", lambda group: torch_linear_gated(group, tokens)),
        ("torch_compile_linear_gated", compiled_linear_gated),
        ("tk_native_linear", tk_native_linear),
        ("tk_fused_linear_gated", lambda group: tk_fused_linear_gated(group, tokens)),
        ("tk_fused_linear_gated_out", lambda group: tk_fused_linear_gated_out(group, tokens)),
    ):
        result = profile_groups(
            f"{label} {name}",
            groups,
            fn,
            warmup=warmup,
            iters=iters,
            cooldown_s=0.0,
            flops=flops,
        )
        print_bench(result)
        results[name] = result.us

    native_ratio = results["torch_linear"] / results["tk_native_linear"]
    fused_ratio = results["torch_linear_gated"] / results["tk_fused_linear_gated"]
    out_ratio = results["torch_linear_gated"] / results["tk_fused_linear_gated_out"]
    compiled_fused_ratio = results["torch_compile_linear_gated"] / results["tk_fused_linear_gated"]
    compiled_out_ratio = results["torch_compile_linear_gated"] / results["tk_fused_linear_gated_out"]
    fused_vs_native = results["tk_native_linear"] / results["tk_fused_linear_gated"]
    out_vs_native = results["tk_native_linear"] / results["tk_fused_linear_gated_out"]
    print(
        f"RESULT {label}: torch={results['torch_linear']:.2f}us "
        f"torch_compile={results['torch_compile_linear']:.2f}us "
        f"torch_gated={results['torch_linear_gated']:.2f}us "
        f"torch_compile_gated={results['torch_compile_linear_gated']:.2f}us "
        f"tk_native={results['tk_native_linear']:.2f}us "
        f"tk_fused={results['tk_fused_linear_gated']:.2f}us "
        f"tk_out_only={results['tk_fused_linear_gated_out']:.2f}us "
        f"native_vs_torch={native_ratio:.2f}x "
        f"fused_vs_torch_gated={fused_ratio:.2f}x "
        f"out_vs_torch_gated={out_ratio:.2f}x "
        f"fused_vs_compile_gated={compiled_fused_ratio:.2f}x "
        f"out_vs_compile_gated={compiled_out_ratio:.2f}x "
        f"fused_vs_native={fused_vs_native:.2f}x "
        f"out_vs_native={out_vs_native:.2f}x",
        flush=True,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile TK GEMM against TK fused GEMM epilogue and torch linear.")
    parser.add_argument("--shapes", type=parse_shape_pair, nargs="+", default=[(64, 1024), (80, 1024), (16, 4096), (20, 4096)])
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cases", nargs="+", default=["fc2", "fc1", "qkv"], choices=["fc2", "fc1", "qkv"])
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.init()
    print(f"GEMM fusion profile gpu={torch.cuda.get_device_name()} torch={torch.__version__}", flush=True)
    case_dims = {
        "fc2": (4 * args.dim, args.dim),
        "fc1": (args.dim, 4 * args.dim),
        "qkv": (args.dim, 3 * args.dim),
    }
    for batch, tokens in args.shapes:
        for case in args.cases:
            k, n = case_dims[case]
            bench_one(case, batch, tokens, k, n, args.warmup, args.iters)


if __name__ == "__main__":
    main()
