from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F

import _C
import _linear_bwd_fused
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
    bytes_per_group = 2 * (m * k * 2 + n * k * 2 + n + m * n * 7 + batch * n * 2)
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
        grad_out = uniform_bf16((m, n), group_seed + 5, -1.0, 1.0)
        dx = torch.empty_like(a)
        dw = torch.empty_like(w)
        db = torch.empty((n,), device="cuda", dtype=torch.float32)
        dresidual = torch.empty_like(residual)
        dprojected = torch.empty_like(projected)
        dgate = torch.empty((batch, n), device="cuda", dtype=torch.float32)
        groups.append((a, w, b, residual, gate, native_out, fused_out, projected, grad_out, dx, dw, db, dresidual, dprojected, dgate))
    return groups


def torch_linear(group: tuple[torch.Tensor, ...]) -> torch.Tensor:
    a, w, b, *_ = group
    return F.linear(a, w, b)


def torch_linear_gelu_train(group: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    a, w, b, *_ = group
    preact = F.linear(a, w, b)
    return F.gelu(preact, approximate="tanh"), preact


def torch_linear_gated(group: tuple[torch.Tensor, ...], tokens: int) -> torch.Tensor:
    a, w, b, residual, gate, *_ = group
    projected = F.linear(a, w, b)
    return residual + gate.repeat_interleave(tokens, dim=0) * projected


def tk_native_linear(group: tuple[torch.Tensor, ...]) -> torch.Tensor:
    a, w, b, _residual, _gate, native_out, *_ = group
    _C.gemm_linear_native(a, w, native_out, b)
    return native_out


def tk_native_linear_variant(group: tuple[torch.Tensor, ...], name: str) -> torch.Tensor:
    a, w, b, _residual, _gate, native_out, *_ = group
    getattr(_C, name)(a, w, native_out, b)
    return native_out


def tk_native_gelu_variant(group: tuple[torch.Tensor, ...], name: str) -> torch.Tensor:
    a, w, b, _residual, _gate, native_out, _fused_out, preact, *_ = group
    getattr(_C, name)(a, w, native_out, b, preact)
    return native_out


def tk_fused_linear_gated(group: tuple[torch.Tensor, ...], tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    a, w, b, residual, gate, _native_out, fused_out, projected, *_ = group
    _C.gemm_linear_gated_residual(a, w, residual, gate, fused_out, projected, b, tokens)
    return fused_out, projected


def tk_fused_linear_gated_out(group: tuple[torch.Tensor, ...], tokens: int) -> torch.Tensor:
    a, w, b, residual, gate, _native_out, fused_out, _projected, *_ = group
    _C.gemm_linear_gated_residual_out(a, w, residual, gate, fused_out, b, tokens)
    return fused_out


def tk_native_then_gated(group: tuple[torch.Tensor, ...], tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    a, w, b, residual, gate, _native_out, fused_out, projected, *_ = group
    _C.gemm_linear_native(a, w, projected, b)
    _C.gated_residual(residual, projected, gate, fused_out, tokens)
    return fused_out, projected


def torch_linear_gated_fwd_bwd(group: tuple[torch.Tensor, ...], tokens: int) -> tuple[torch.Tensor, ...]:
    a, w, b, residual, gate, *_rest = group
    grad_out = group[8]
    projected = F.linear(a, w, b)
    gate_tokens = gate.repeat_interleave(tokens, dim=0)
    out = residual + gate_tokens * projected
    dprojected = grad_out * gate_tokens
    dx = dprojected.matmul(w)
    dw = dprojected.transpose(0, 1).matmul(a)
    db = dprojected.float().sum(dim=0)
    dgate = (grad_out.reshape(gate.shape[0], tokens, -1).float() * projected.reshape(gate.shape[0], tokens, -1).float()).sum(dim=1)
    return out, dx, dw, db, grad_out, dgate


def tk_projected_fwd_bwd(group: tuple[torch.Tensor, ...], tokens: int) -> tuple[torch.Tensor, ...]:
    a, w, b, residual, gate, _native_out, fused_out, projected, grad_out, dx, dw, db, dresidual, dprojected, dgate = group
    _C.gemm_linear_gated_residual(a, w, residual, gate, fused_out, projected, b, tokens)
    _C.gated_residual_backward_no_dx_db(grad_out, projected, gate, dprojected, dgate, db, tokens)
    _linear_bwd_fused.dw_gemm(dprojected, a, dw)
    _linear_bwd_fused.dx_gemm_native(dprojected, w.contiguous(), dx)
    return fused_out, dx, dw, db, grad_out, dgate


def tk_native_then_gated_fwd_bwd(group: tuple[torch.Tensor, ...], tokens: int) -> tuple[torch.Tensor, ...]:
    a, w, b, residual, gate, _native_out, fused_out, projected, grad_out, dx, dw, db, dresidual, dprojected, dgate = group
    _C.gemm_linear_native(a, w, projected, b)
    _C.gated_residual(residual, projected, gate, fused_out, tokens)
    _C.gated_residual_backward_no_dx_db(grad_out, projected, gate, dprojected, dgate, db, tokens)
    _linear_bwd_fused.dw_gemm(dprojected, a, dw)
    _linear_bwd_fused.dx_gemm_native(dprojected, w.contiguous(), dx)
    return fused_out, dx, dw, db, grad_out, dgate


def tk_out_recompute_fwd_bwd(group: tuple[torch.Tensor, ...], tokens: int) -> tuple[torch.Tensor, ...]:
    a, w, b, residual, gate, _native_out, fused_out, projected, grad_out, dx, dw, db, dresidual, dprojected, dgate = group
    _C.gemm_linear_gated_residual_out(a, w, residual, gate, fused_out, b, tokens)
    _C.gemm_linear_native(a, w, projected, b)
    _C.gated_residual_backward_no_dx_db(grad_out, projected, gate, dprojected, dgate, db, tokens)
    _linear_bwd_fused.dw_gemm(dprojected, a, dw)
    _linear_bwd_fused.dx_gemm_native(dprojected, w.contiguous(), dx)
    return fused_out, dx, dw, db, grad_out, dgate


def tk_forward_projected(group: tuple[torch.Tensor, ...], tokens: int) -> torch.Tensor:
    a, w, b, residual, gate, _native_out, fused_out, projected, *_ = group
    _C.gemm_linear_gated_residual(a, w, residual, gate, fused_out, projected, b, tokens)
    return fused_out


def tk_forward_out_only(group: tuple[torch.Tensor, ...], tokens: int) -> torch.Tensor:
    a, w, b, residual, gate, _native_out, fused_out, *_ = group
    _C.gemm_linear_gated_residual_out(a, w, residual, gate, fused_out, b, tokens)
    return fused_out


def tk_backward_gate(group: tuple[torch.Tensor, ...], tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    _a, _w, _b, _residual, gate, _native_out, _fused_out, projected, grad_out, _dx, _dw, db, _dresidual, dprojected, dgate = group
    _C.gated_residual_backward_no_dx_db(grad_out, projected, gate, dprojected, dgate, db, tokens)
    return dprojected, dgate


def tk_backward_dw(group: tuple[torch.Tensor, ...]) -> torch.Tensor:
    a = group[0]
    dprojected = group[13]
    dw = group[10]
    _linear_bwd_fused.dw_gemm(dprojected, a, dw)
    return dw


def tk_backward_dx(group: tuple[torch.Tensor, ...]) -> torch.Tensor:
    w = group[1]
    dprojected = group[13]
    dx = group[9]
    _linear_bwd_fused.dx_gemm_native(dprojected, w.contiguous(), dx)
    return dx


def tk_backward_db(group: tuple[torch.Tensor, ...]) -> torch.Tensor:
    dprojected = group[13]
    db = group[11]
    _linear_bwd_fused.bias_reduce(dprojected, db)
    return db


def prepare_native_gelu(groups: list[tuple[torch.Tensor, ...]]) -> None:
    for group in groups:
        tk_native_gelu_variant(group, "gemm_custom_native")
    torch.cuda.synchronize()


def tk_gelu_bwd_bias_variant(group: tuple[torch.Tensor, ...], name: str) -> tuple[torch.Tensor, torch.Tensor]:
    preact = group[7]
    grad_out = group[8]
    dz = group[13]
    db = group[11]
    getattr(_linear_bwd_fused, name)(grad_out, preact, dz, db)
    return dz, db


def tk_gelu_bwd_dw_variant(group: tuple[torch.Tensor, ...], name: str) -> torch.Tensor:
    x = group[0]
    dz = group[13]
    dw = group[10]
    getattr(_linear_bwd_fused, name)(x, dz, dw)
    return dw


def tk_gelu_bwd_dx_variant(group: tuple[torch.Tensor, ...], name: str) -> torch.Tensor:
    w = group[1]
    dz = group[13]
    dx = group[9]
    getattr(_linear_bwd_fused, name)(dz, w.contiguous(), dx)
    return dx


def prepare_projected_and_dprojected(groups: list[tuple[torch.Tensor, ...]], tokens: int) -> None:
    for group in groups:
        tk_forward_projected(group, tokens)
        tk_backward_gate(group, tokens)
    torch.cuda.synchronize()


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


def check_fwd_bwd(label: str, group: tuple[torch.Tensor, ...], tokens: int) -> None:
    expected = torch_linear_gated_fwd_bwd(group, tokens)
    projected_actual = tk_projected_fwd_bwd(group, tokens)
    recompute_actual = tk_out_recompute_fwd_bwd(group, tokens)
    torch.cuda.synchronize()

    names = ("out", "dx", "dw", "db", "dresidual", "dgate")
    for variant, actual in (("tk_projected", projected_actual), ("tk_out_recompute", recompute_actual)):
        for name, a, e in zip(names, actual, expected, strict=True):
            diff = (a.float() - e.float()).abs()
            print(
                f"  correctness {label} {variant} {name}: "
                f"max={diff.max().item():.6g} mean={diff.mean().item():.6g}",
                flush=True,
            )


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
    compiled_linear_gelu_train: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, torch.Tensor]] = torch.compile(
        torch_linear_gelu_train,
        dynamic=False,
    )
    compiled_linear_gated: Callable[[tuple[torch.Tensor, ...]], torch.Tensor] = torch.compile(
        lambda group: torch_linear_gated(group, tokens),
        dynamic=False,
    )
    results = {}
    for name, fn in (
        ("torch_linear", torch_linear),
        ("torch_compile_linear", compiled_linear),
        ("torch_linear_gelu_train", torch_linear_gelu_train),
        ("torch_compile_linear_gelu_train", compiled_linear_gelu_train),
        ("torch_linear_gated", lambda group: torch_linear_gated(group, tokens)),
        ("torch_compile_linear_gated", compiled_linear_gated),
        ("tk_native_linear", tk_native_linear),
        ("tk_native_linear_m2n4s4", lambda group: tk_native_linear_variant(group, "gemm_linear_native_m2n4s4")),
        ("tk_native_linear_m2n4s6", lambda group: tk_native_linear_variant(group, "gemm_linear_native_m2n4s6")),
        ("tk_native_linear_m2n4s8", lambda group: tk_native_linear_variant(group, "gemm_linear_native_m2n4s8")),
        ("tk_native_linear_m2n4s12", lambda group: tk_native_linear_variant(group, "gemm_linear_native_m2n4s12")),
        ("tk_native_linear_m2n4s16", lambda group: tk_native_linear_variant(group, "gemm_linear_native_m2n4s16")),
        ("tk_native_linear_m2n2s8", lambda group: tk_native_linear_variant(group, "gemm_linear_native_m2n2s8")),
        ("tk_native_linear_m1n4s8", lambda group: tk_native_linear_variant(group, "gemm_linear_native_m1n4s8")),
        ("tk_native_gelu", lambda group: tk_native_gelu_variant(group, "gemm_custom_native")),
        ("tk_native_gelu_m2n4s4", lambda group: tk_native_gelu_variant(group, "gemm_custom_native_m2n4s4")),
        ("tk_native_gelu_m2n4s6", lambda group: tk_native_gelu_variant(group, "gemm_custom_native_m2n4s6")),
        ("tk_native_gelu_m2n4s8", lambda group: tk_native_gelu_variant(group, "gemm_custom_native_m2n4s8")),
        ("tk_native_gelu_m2n4s12", lambda group: tk_native_gelu_variant(group, "gemm_custom_native_m2n4s12")),
        ("tk_native_gelu_m2n4s16", lambda group: tk_native_gelu_variant(group, "gemm_custom_native_m2n4s16")),
        ("tk_native_gelu_m2n2s8", lambda group: tk_native_gelu_variant(group, "gemm_custom_native_m2n2s8")),
        ("tk_native_gelu_m1n4s8", lambda group: tk_native_gelu_variant(group, "gemm_custom_native_m1n4s8")),
        ("tk_fused_linear_gated", lambda group: tk_fused_linear_gated(group, tokens)),
        ("tk_fused_linear_gated_m2n4s2", lambda group: (_C.gemm_linear_gated_residual_m2n4s2(group[0], group[1], group[3], group[4], group[6], group[7], group[2], tokens), group[6])),
        ("tk_fused_linear_gated_m2n4s4", lambda group: (_C.gemm_linear_gated_residual_m2n4s4(group[0], group[1], group[3], group[4], group[6], group[7], group[2], tokens), group[6])),
        ("tk_fused_linear_gated_m2n4s6", lambda group: (_C.gemm_linear_gated_residual_m2n4s6(group[0], group[1], group[3], group[4], group[6], group[7], group[2], tokens), group[6])),
        ("tk_fused_linear_gated_m2n4s8", lambda group: (_C.gemm_linear_gated_residual_m2n4s8(group[0], group[1], group[3], group[4], group[6], group[7], group[2], tokens), group[6])),
        ("tk_fused_linear_gated_m2n4s12", lambda group: (_C.gemm_linear_gated_residual_m2n4s12(group[0], group[1], group[3], group[4], group[6], group[7], group[2], tokens), group[6])),
        ("tk_fused_linear_gated_m2n4s16", lambda group: (_C.gemm_linear_gated_residual_m2n4s16(group[0], group[1], group[3], group[4], group[6], group[7], group[2], tokens), group[6])),
        ("tk_fused_linear_gated_m2n2s8", lambda group: (_C.gemm_linear_gated_residual_m2n2s8(group[0], group[1], group[3], group[4], group[6], group[7], group[2], tokens), group[6])),
        ("tk_fused_linear_gated_m1n4s8", lambda group: (_C.gemm_linear_gated_residual_m1n4s8(group[0], group[1], group[3], group[4], group[6], group[7], group[2], tokens), group[6])),
        ("tk_fused_linear_gated_out", lambda group: tk_fused_linear_gated_out(group, tokens)),
        ("tk_native_then_gated", lambda group: tk_native_then_gated(group, tokens)),
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
    gelu_ratio = results["torch_compile_linear_gelu_train"] / results["tk_native_gelu"]
    fused_vs_native = results["tk_native_linear"] / results["tk_fused_linear_gated"]
    out_vs_native = results["tk_native_linear"] / results["tk_fused_linear_gated_out"]
    print(
        f"RESULT {label}: torch={results['torch_linear']:.2f}us "
        f"torch_compile={results['torch_compile_linear']:.2f}us "
        f"torch_compile_gelu_train={results['torch_compile_linear_gelu_train']:.2f}us "
        f"torch_gated={results['torch_linear_gated']:.2f}us "
        f"torch_compile_gated={results['torch_compile_linear_gated']:.2f}us "
        f"tk_native={results['tk_native_linear']:.2f}us "
        f"tk_gelu={results['tk_native_gelu']:.2f}us "
        f"tk_fused={results['tk_fused_linear_gated']:.2f}us "
        f"tk_out_only={results['tk_fused_linear_gated_out']:.2f}us "
        f"native_vs_torch={native_ratio:.2f}x "
        f"gelu_vs_compile_gelu_train={gelu_ratio:.2f}x "
        f"fused_vs_torch_gated={fused_ratio:.2f}x "
        f"out_vs_torch_gated={out_ratio:.2f}x "
        f"fused_vs_compile_gated={compiled_fused_ratio:.2f}x "
        f"out_vs_compile_gated={compiled_out_ratio:.2f}x "
        f"fused_vs_native={fused_vs_native:.2f}x "
        f"out_vs_native={out_vs_native:.2f}x",
        flush=True,
    )
    return results


def bench_one_fwd_bwd(
    label: str,
    batch: int,
    tokens: int,
    k: int,
    n: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    m = batch * tokens
    print(f"\n=== {label} fwd+bwd B={batch} T={tokens} M={m} N={n} K={k} ===", flush=True)
    groups = make_groups(batch, tokens, k, n, 81000 + batch + tokens + k + n)
    check_fwd_bwd(label, groups[0], tokens)
    flops = 6.0 * m * n * k

    compiled_fwd_bwd: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]] = torch.compile(
        lambda group: torch_linear_gated_fwd_bwd(group, tokens),
        dynamic=False,
    )
    results = {}
    for name, fn in (
        ("torch_linear_gated_fwd_bwd", lambda group: torch_linear_gated_fwd_bwd(group, tokens)),
        ("torch_compile_linear_gated_fwd_bwd", compiled_fwd_bwd),
        ("tk_projected_fwd_bwd", lambda group: tk_projected_fwd_bwd(group, tokens)),
        ("tk_native_then_gated_fwd_bwd", lambda group: tk_native_then_gated_fwd_bwd(group, tokens)),
        ("tk_out_recompute_fwd_bwd", lambda group: tk_out_recompute_fwd_bwd(group, tokens)),
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

    compile_us = results["torch_compile_linear_gated_fwd_bwd"]
    print(
        f"RESULT {label} fwd_bwd: "
        f"torch={results['torch_linear_gated_fwd_bwd']:.2f}us "
        f"torch_compile={compile_us:.2f}us "
        f"tk_projected={results['tk_projected_fwd_bwd']:.2f}us "
        f"tk_native_then_gated={results['tk_native_then_gated_fwd_bwd']:.2f}us "
        f"tk_out_recompute={results['tk_out_recompute_fwd_bwd']:.2f}us "
        f"tk_projected_vs_compile={compile_us / results['tk_projected_fwd_bwd']:.2f}x "
        f"tk_native_then_gated_vs_compile={compile_us / results['tk_native_then_gated_fwd_bwd']:.2f}x "
        f"tk_out_recompute_vs_compile={compile_us / results['tk_out_recompute_fwd_bwd']:.2f}x",
        flush=True,
    )
    return results


def bench_one_breakdown(
    label: str,
    batch: int,
    tokens: int,
    k: int,
    n: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    m = batch * tokens
    print(f"\n=== {label} breakdown B={batch} T={tokens} M={m} N={n} K={k} ===", flush=True)
    groups = make_groups(batch, tokens, k, n, 91000 + batch + tokens + k + n)
    check_fwd_bwd(label, groups[0], tokens)
    prepare_projected_and_dprojected(groups, tokens)

    flops_gemm = 2.0 * m * n * k
    results = {}
    for name, fn, flops in (
        ("tk_forward_projected", lambda group: tk_forward_projected(group, tokens), flops_gemm),
        ("tk_forward_out_only", lambda group: tk_forward_out_only(group, tokens), flops_gemm),
        ("tk_backward_gate_db", lambda group: tk_backward_gate(group, tokens), 0.0),
        ("tk_backward_dw", tk_backward_dw, flops_gemm),
        ("tk_backward_dx", tk_backward_dx, flops_gemm),
        ("tk_backward_db_standalone", tk_backward_db, 0.0),
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

    projected_sum = (
        results["tk_forward_projected"]
        + results["tk_backward_gate_db"]
        + results["tk_backward_dw"]
        + results["tk_backward_dx"]
    )
    out_recompute_sum = (
        results["tk_forward_out_only"]
        + results["tk_forward_out_only"]
        + results["tk_backward_gate_db"]
        + results["tk_backward_dw"]
        + results["tk_backward_dx"]
    )
    print(
        f"RESULT {label} breakdown: "
        f"projected_sum={projected_sum:.2f}us "
        f"out_recompute_approx_sum={out_recompute_sum:.2f}us "
        f"fwd_projected={results['tk_forward_projected']:.2f}us "
        f"gate_db_bwd={results['tk_backward_gate_db']:.2f}us "
        f"dw={results['tk_backward_dw']:.2f}us "
        f"dx={results['tk_backward_dx']:.2f}us "
        f"db_standalone={results['tk_backward_db_standalone']:.2f}us",
        flush=True,
    )
    return results


def bench_one_gelu_bwd_breakdown(
    label: str,
    batch: int,
    tokens: int,
    k: int,
    n: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    m = batch * tokens
    print(f"\n=== {label} gelu-bwd breakdown B={batch} T={tokens} M={m} N={n} K={k} ===", flush=True)
    groups = make_groups(batch, tokens, k, n, 101000 + batch + tokens + k + n)
    check_outputs(label, groups[0], tokens)
    prepare_native_gelu(groups)

    flops_gemm = 2.0 * m * n * k
    results = {}
    for name, fn, flops in (
        ("gelu_bwd_bias", lambda group: tk_gelu_bwd_bias_variant(group, "gelu_bwd_bias"), 0.0),
        ("gelu_bwd_bias_r512", lambda group: tk_gelu_bwd_bias_variant(group, "gelu_bwd_bias_r512"), 0.0),
        ("gelu_bwd_bias_r2048", lambda group: tk_gelu_bwd_bias_variant(group, "gelu_bwd_bias_r2048"), 0.0),
        ("gelu_bwd_bias_c32r1024", lambda group: tk_gelu_bwd_bias_variant(group, "gelu_bwd_bias_c32r1024"), 0.0),
        ("dw_gemm", lambda group: tk_gelu_bwd_dw_variant(group, "dw_gemm"), flops_gemm),
        ("dw_gemm_s4", lambda group: tk_gelu_bwd_dw_variant(group, "dw_gemm_s4"), flops_gemm),
        ("dw_gemm_s16", lambda group: tk_gelu_bwd_dw_variant(group, "dw_gemm_s16"), flops_gemm),
        ("dw_gemm_g120", lambda group: tk_gelu_bwd_dw_variant(group, "dw_gemm_g120"), flops_gemm),
        ("dw_gemm_g132", lambda group: tk_gelu_bwd_dw_variant(group, "dw_gemm_g132"), flops_gemm),
        ("dw_gemm_p3", lambda group: tk_gelu_bwd_dw_variant(group, "dw_gemm_p3"), flops_gemm),
        ("dx_gemm_native", lambda group: tk_gelu_bwd_dx_variant(group, "dx_gemm_native"), flops_gemm),
        ("dx_gemm_native_s4", lambda group: tk_gelu_bwd_dx_variant(group, "dx_gemm_native_s4"), flops_gemm),
        ("dx_gemm_native_s16", lambda group: tk_gelu_bwd_dx_variant(group, "dx_gemm_native_s16"), flops_gemm),
        ("dx_gemm_native_g120", lambda group: tk_gelu_bwd_dx_variant(group, "dx_gemm_native_g120"), flops_gemm),
        ("dx_gemm_native_g132", lambda group: tk_gelu_bwd_dx_variant(group, "dx_gemm_native_g132"), flops_gemm),
        ("dx_gemm_native_p3", lambda group: tk_gelu_bwd_dx_variant(group, "dx_gemm_native_p3"), flops_gemm),
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

    best_gelu = min((v, k) for k, v in results.items() if k.startswith("gelu_bwd_bias"))
    best_dw = min((v, k) for k, v in results.items() if k.startswith("dw_gemm"))
    best_dx = min((v, k) for k, v in results.items() if k.startswith("dx_gemm_native"))
    print(
        f"RESULT {label} gelu_bwd_breakdown: "
        f"best_gelu={best_gelu[1]}:{best_gelu[0]:.2f}us "
        f"best_dw={best_dw[1]}:{best_dw[0]:.2f}us "
        f"best_dx={best_dx[1]}:{best_dx[0]:.2f}us "
        f"best_sum={best_gelu[0] + best_dw[0] + best_dx[0]:.2f}us "
        f"default_sum={results['gelu_bwd_bias'] + results['dw_gemm'] + results['dx_gemm_native']:.2f}us",
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
    parser.add_argument("--bench", choices=["fwd", "fwd_bwd", "breakdown", "gelu_bwd_breakdown", "both"], default="fwd")
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
            if args.bench in ("fwd", "both"):
                bench_one(case, batch, tokens, k, n, args.warmup, args.iters)
            if args.bench in ("fwd_bwd", "both"):
                bench_one_fwd_bwd(case, batch, tokens, k, n, args.warmup, args.iters)
            if args.bench in ("breakdown", "both"):
                bench_one_breakdown(case, batch, tokens, k, n, args.warmup, args.iters)
            if args.bench in ("gelu_bwd_breakdown", "both"):
                bench_one_gelu_bwd_breakdown(case, batch, tokens, k, n, args.warmup, args.iters)


if __name__ == "__main__":
    main()
