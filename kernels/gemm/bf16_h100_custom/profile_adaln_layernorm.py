from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Callable

import torch
import torch.nn.functional as F

import _C
from dit3d_e2e_bench import tk_layernorm_adaln_backward_op, tk_layernorm_adaln_op
from harness import reference_backward, reference_forward, run_fused_backward, run_fused_forward
from tk_bench import check_close, input_group_count, print_bench, profile_groups, uniform_bf16


def compile_or_none(fn: Callable, *, max_autotune: bool = False, fixed_shapes: bool = False):
    try:
        kwargs = {}
        if max_autotune:
            kwargs["mode"] = "max-autotune"
        if fixed_shapes:
            kwargs["dynamic"] = False
            torch._dynamo.reset()
        return torch.compile(fn, **kwargs)
    except Exception as exc:
        print(f"torch.compile unavailable for {getattr(fn, '__name__', repr(fn))}: {exc!r}", flush=True)
        return None


def torch_autograd_forward(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, tokens: int, eps: float):
    batch = shift.shape[0]
    x3 = x.reshape(batch, tokens, x.shape[1])
    out = torch.nn.functional.layer_norm(x3, (x.shape[1],), eps=eps)
    out = out * (1.0 + scale[:, None, :]) + shift[:, None, :]
    return out.reshape_as(x)


def torch_autograd_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    tokens: int,
    eps: float,
):
    with torch.enable_grad():
        out = torch_autograd_forward(x, shift, scale, tokens, eps)
        dx, dshift, dscale = torch.autograd.grad(
            out,
            (x, shift, scale),
            grad,
            retain_graph=False,
            create_graph=False,
        )
    return dx, dshift, dscale


class FusedAdaLNAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, tokens: int, eps: float):
        flat = x.contiguous()
        scale_c = scale.contiguous()
        out, mean, rstd = tk_layernorm_adaln_op(flat, shift.contiguous(), scale_c, tokens, eps)
        ctx.save_for_backward(flat, scale_c, mean, rstd)
        ctx.tokens = tokens
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, scale, mean, rstd = ctx.saved_tensors
        dx, dshift, dscale = tk_layernorm_adaln_backward_op(grad_out.contiguous(), x, scale, mean, rstd, ctx.tokens)
        return dx, dshift.to(scale.dtype), dscale.to(scale.dtype), None, None


def custom_autograd_forward(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, tokens: int, eps: float):
    return FusedAdaLNAutograd.apply(x, shift, scale, tokens, eps)


def run_fused_forward_variant(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, tokens: int, eps: float, variant: str):
    if variant == "cta":
        return run_fused_forward(x, shift, scale, tokens, eps)
    out = torch.empty_like(x)
    mean = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    if variant == "persistent":
        _C.layernorm_adaln_persistent(x, shift, scale, out, mean, rstd, tokens, eps)
    elif variant == "warp4":
        _C.layernorm_adaln_warp4(x, shift, scale, out, mean, rstd, tokens, eps)
    else:
        raise ValueError(f"unsupported forward variant: {variant}")
    return out, mean, rstd


def run_fused_backward_variant(
    grad: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    tokens: int,
    variant: str,
):
    if variant == "cta":
        return run_fused_backward(grad, x, scale, mean, rstd, tokens)
    dx = torch.empty_like(x)
    dshift = torch.empty_like(scale, dtype=torch.float32)
    dscale = torch.empty_like(scale, dtype=torch.float32)
    if variant == "warp4":
        _C.layernorm_adaln_backward_warp4(grad, x, scale, mean, rstd, dx, dshift, dscale, tokens)
    else:
        raise ValueError(f"unsupported backward variant: {variant}")
    return dx, dshift, dscale


def run_fused_fwd_bwd_variant(
    grad: torch.Tensor,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    tokens: int,
    eps: float,
    variant: str,
):
    out, mean, rstd = run_fused_forward_variant(x, shift, scale, tokens, eps, variant)
    dx, dshift, dscale = run_fused_backward_variant(grad, x, scale, mean, rstd, tokens, variant if variant == "warp4" else "cta")
    return out, dx, dshift, dscale


def make_groups(batch: int, tokens: int, dim: int, eps: float, seed: int):
    rows = batch * tokens
    group_bytes = (
        rows * dim * 2  # x
        + batch * dim * 2 * 2  # shift, scale
        + rows * dim * 2  # grad
        + rows * 4 * 2  # mean, rstd
    )
    groups_n = min(input_group_count(group_bytes), 8)
    groups = []
    for i in range(groups_n):
        x = uniform_bf16((rows, dim), seed + i * 10 + 0, -2.0, 2.0)
        shift = uniform_bf16((batch, dim), seed + i * 10 + 1, -0.5, 0.5)
        scale = uniform_bf16((batch, dim), seed + i * 10 + 2, -0.25, 0.25)
        grad = uniform_bf16((rows, dim), seed + i * 10 + 3, -1.0, 1.0)
        out, mean, rstd = run_fused_forward(x, shift, scale, tokens, eps)
        groups.append((x, shift, scale, grad, mean, rstd, out))
    return groups


def make_autograd_groups(batch: int, tokens: int, dim: int, seed: int):
    rows = batch * tokens
    group_bytes = rows * dim * 2 * 2 + batch * dim * 2 * 2
    groups_n = min(input_group_count(group_bytes), 8)
    groups = []
    for i in range(groups_n):
        x = uniform_bf16((rows, dim), seed + i * 10 + 0, -2.0, 2.0).requires_grad_(True)
        shift = uniform_bf16((batch, dim), seed + i * 10 + 1, -0.5, 0.5).requires_grad_(True)
        scale = uniform_bf16((batch, dim), seed + i * 10 + 2, -0.25, 0.25).requires_grad_(True)
        grad = uniform_bf16((rows, dim), seed + i * 10 + 3, -1.0, 1.0)
        groups.append((x, shift, scale, grad))
    return groups


def export_trace(path: Path, name: str, fn: Callable[[], object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    trace_path = path / f"{name}.json"
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        fn()
    prof.export_chrome_trace(str(trace_path))
    summarize_trace(trace_path)


def summarize_trace(trace_path: Path) -> None:
    data = json.loads(trace_path.read_text())
    events = data.get("traceEvents", [])
    kernel_names: list[str] = []
    aten_names: list[str] = []
    compiled = 0
    for event in events:
        name = str(event.get("name", ""))
        cat = str(event.get("cat", ""))
        if "kernel" in cat.lower() or name.startswith(("triton_", "void ", "ampere_", "cutlass", "sm90", "_Z")):
            kernel_names.append(name)
        if name.startswith("aten::"):
            aten_names.append(name)
        if "CompiledFunction" in name:
            compiled += 1
    unique_kernels = sorted(set(kernel_names))
    materializing_aten = [
        name for name in aten_names
        if any(token in name for token in ("empty", "zeros", "clone", "copy", "index_add", "sum", "mul", "sub", "add"))
    ]
    print(f"\nTrace {trace_path.name}:", flush=True)
    print(f"  cuda_kernel_events={len(kernel_names)} unique_cuda_kernels={len(unique_kernels)}", flush=True)
    print(f"  compiled_function_events={compiled}", flush=True)
    print(f"  aten_events={len(aten_names)} materializing_or_reduction_aten_events={len(materializing_aten)}", flush=True)
    if unique_kernels:
        print("  kernel_names:", flush=True)
        for name in unique_kernels[:20]:
            print(f"    - {name}", flush=True)
        if len(unique_kernels) > 20:
            print(f"    ... {len(unique_kernels) - 20} more", flush=True)
    if materializing_aten:
        counts: dict[str, int] = {}
        for name in materializing_aten:
            counts[name] = counts.get(name, 0) + 1
        print("  selected_aten_counts:", flush=True)
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20]:
            print(f"    - {name}: {count}", flush=True)


def make_train_groups(batch: int, tokens: int, dim: int, seed: int):
    rows = batch * tokens
    group_bytes = rows * dim * 2 * 2 + batch * dim * 2 * 2
    groups_n = min(input_group_count(group_bytes), 8)
    groups = []
    for i in range(groups_n):
        x = uniform_bf16((rows, dim), seed + i * 10 + 0, -2.0, 2.0).requires_grad_(True)
        shift = uniform_bf16((batch, dim), seed + i * 10 + 1, -0.5, 0.5).requires_grad_(True)
        scale = uniform_bf16((batch, dim), seed + i * 10 + 2, -0.25, 0.25).requires_grad_(True)
        grad = uniform_bf16((rows, dim), seed + i * 10 + 3, -1.0, 1.0)
        groups.append((x, shift, scale, grad))
    return groups


def zero_train_group(group):
    for tensor in group[:3]:
        tensor.grad = None


def train_step(fn: Callable, group, tokens: int, eps: float):
    x, shift, scale, grad = group
    out = fn(x, shift, scale, tokens, eps)
    out.backward(grad)
    zero_train_group(group)


def validate_shape(batch: int, tokens: int, dim: int, eps: float) -> bool:
    rows = batch * tokens
    x = uniform_bf16((rows, dim), 71000, -2.0, 2.0)
    shift = uniform_bf16((batch, dim), 71001, -0.5, 0.5)
    scale = uniform_bf16((batch, dim), 71002, -0.25, 0.25)
    grad = uniform_bf16((rows, dim), 71003, -1.0, 1.0)

    ref_out, ref_mean, ref_rstd = reference_forward(x, shift, scale, tokens, eps)
    tk_out, tk_mean, tk_rstd = run_fused_forward(x, shift, scale, tokens, eps)
    ok = check_close("forward out", tk_out, ref_out, atol=2e-2)
    ok = check_close("forward mean", tk_mean, ref_mean, atol=2e-3) and ok
    ok = check_close("forward rstd", tk_rstd, ref_rstd, atol=2e-3) and ok
    for variant in ("persistent", "warp4"):
        variant_out, variant_mean, variant_rstd = run_fused_forward_variant(x, shift, scale, tokens, eps, variant)
        ok = check_close(f"{variant} forward out", variant_out, ref_out, atol=2e-2) and ok
        ok = check_close(f"{variant} forward mean", variant_mean, ref_mean, atol=2e-3) and ok
        ok = check_close(f"{variant} forward rstd", variant_rstd, ref_rstd, atol=2e-3) and ok

    ref_dx, ref_dshift, ref_dscale = reference_backward(grad, x, scale, ref_mean, ref_rstd, tokens)
    tk_dx, tk_dshift, tk_dscale = run_fused_backward(grad, x, scale, tk_mean, tk_rstd, tokens)
    ok = check_close("backward dx", tk_dx, ref_dx, atol=2e-2) and ok
    ok = check_close("backward dshift", tk_dshift, ref_dshift, atol=5e-2) and ok
    ok = check_close("backward dscale", tk_dscale, ref_dscale, atol=5e-2) and ok
    warp4_dx, warp4_dshift, warp4_dscale = run_fused_backward_variant(grad, x, scale, tk_mean, tk_rstd, tokens, "warp4")
    ok = check_close("warp4 backward dx", warp4_dx, ref_dx, atol=2e-2) and ok
    ok = check_close("warp4 backward dshift", warp4_dshift, ref_dshift, atol=5e-2) and ok
    ok = check_close("warp4 backward dscale", warp4_dscale, ref_dscale, atol=5e-2) and ok
    return ok


def profile_shape(
    batch: int,
    tokens: int,
    dim: int,
    warmup: int,
    iters: int,
    eps: float,
    *,
    compile_max_autotune: bool,
    compile_fixed_shapes: bool,
    include_autograd_baseline: bool,
    trace_dir: Path | None,
    trace_shape: str,
) -> bool:
    label = f"B{batch}T{tokens}D{dim}"
    print(f"\nLayerNorm+AdaLN profile {label}", flush=True)
    ok = validate_shape(batch, tokens, dim, eps)
    groups = make_groups(batch, tokens, dim, eps, seed=72000 + batch + tokens)

    compiled_forward = compile_or_none(
        lambda x, shift, scale: reference_forward(x, shift, scale, tokens, eps)[0],
        max_autotune=compile_max_autotune,
        fixed_shapes=compile_fixed_shapes,
    )
    compiled_backward = compile_or_none(
        lambda grad, x, scale, mean, rstd: reference_backward(grad, x, scale, mean, rstd, tokens),
        max_autotune=compile_max_autotune,
        fixed_shapes=compile_fixed_shapes,
    )
    compiled_autograd_bwd = None
    compiled_autograd_fwd_bwd = None
    autograd_groups = None
    if include_autograd_baseline:
        compiled_autograd_bwd = compile_or_none(
            lambda grad, x, shift, scale: torch_autograd_backward(grad, x, shift, scale, tokens, eps),
            max_autotune=compile_max_autotune,
            fixed_shapes=compile_fixed_shapes,
        )
        compiled_autograd_fwd_bwd = compiled_autograd_bwd
        autograd_groups = make_autograd_groups(batch, tokens, dim, seed=76000 + batch + tokens)

    rows = batch * tokens
    bytes_forward = rows * dim * 2 * 2 + batch * dim * 2 * 2 + rows * 4 * 2
    bytes_backward = rows * dim * 2 * 4 + batch * dim * 2 + rows * 4 * 2 + batch * dim * 4 * 2
    bytes_fwd_bwd = bytes_forward + bytes_backward
    results = [
        profile_groups(
            f"{label} eager forward",
            groups,
            lambda g: reference_forward(g[0], g[1], g[2], tokens, eps)[0],
            warmup=warmup,
            iters=iters,
            bytes_moved=bytes_forward,
        )
    ]
    if compiled_forward is not None:
        results.append(
            profile_groups(
                f"{label} compile forward",
                groups,
                lambda g: compiled_forward(g[0], g[1], g[2]),
                warmup=warmup,
                iters=iters,
                bytes_moved=bytes_forward,
            )
        )
    results.append(
        profile_groups(
            f"{label} tk fused forward",
            groups,
            lambda g: run_fused_forward(g[0], g[1], g[2], tokens, eps),
            warmup=warmup,
            iters=iters,
            bytes_moved=bytes_forward,
        )
    )
    for variant in ("persistent", "warp4"):
        results.append(
            profile_groups(
                f"{label} tk {variant} forward",
                groups,
                lambda g, variant=variant: run_fused_forward_variant(g[0], g[1], g[2], tokens, eps, variant),
                warmup=warmup,
                iters=iters,
                bytes_moved=bytes_forward,
            )
        )

    results.append(
        profile_groups(
            f"{label} eager backward",
            groups,
            lambda g: reference_backward(g[3], g[0], g[2], g[4], g[5], tokens),
            warmup=warmup,
            iters=iters,
            bytes_moved=bytes_backward,
        )
    )
    if compiled_backward is not None:
        results.append(
            profile_groups(
                f"{label} compile backward",
                groups,
                lambda g: compiled_backward(g[3], g[0], g[2], g[4], g[5]),
                warmup=warmup,
                iters=iters,
                bytes_moved=bytes_backward,
            )
        )
    if compiled_autograd_bwd is not None and autograd_groups is not None:
        results.append(
            profile_groups(
                f"{label} compile autograd fwd+bwd",
                autograd_groups,
                lambda g: compiled_autograd_fwd_bwd(g[3], g[0], g[1], g[2]),
                warmup=warmup,
                iters=iters,
                bytes_moved=bytes_fwd_bwd,
            )
        )
    results.append(
        profile_groups(
            f"{label} tk fused backward",
            groups,
            lambda g: run_fused_backward(g[3], g[0], g[2], g[4], g[5], tokens),
            warmup=warmup,
            iters=iters,
            bytes_moved=bytes_backward,
        )
    )
    results.append(
        profile_groups(
            f"{label} tk warp4 fwd+bwd",
            groups,
            lambda g: run_fused_fwd_bwd_variant(g[3], g[0], g[1], g[2], tokens, eps, "warp4"),
            warmup=warmup,
            iters=iters,
            bytes_moved=bytes_fwd_bwd,
        )
    )

    if trace_dir is not None and label == trace_shape:
        trace_root = trace_dir / label
        if compiled_backward is not None:
            export_trace(
                trace_root,
                "compile_manual_backward",
                lambda: compiled_backward(groups[0][3], groups[0][0], groups[0][2], groups[0][4], groups[0][5]),
            )
        if compiled_autograd_fwd_bwd is not None and autograd_groups is not None:
            export_trace(
                trace_root,
                "compile_autograd_fwd_bwd",
                lambda: compiled_autograd_fwd_bwd(
                    autograd_groups[0][3],
                    autograd_groups[0][0],
                    autograd_groups[0][1],
                    autograd_groups[0][2],
                ),
            )
        export_trace(
            trace_root,
            "tk_warp4_fwd_bwd",
            lambda: run_fused_fwd_bwd_variant(groups[0][3], groups[0][0], groups[0][1], groups[0][2], tokens, eps, "warp4"),
        )
    results.append(
        profile_groups(
            f"{label} tk warp4 backward",
            groups,
            lambda g: run_fused_backward_variant(g[3], g[0], g[2], g[4], g[5], tokens, "warp4"),
            warmup=warmup,
            iters=iters,
            bytes_moved=bytes_backward,
        )
    )

    for result in results:
        print_bench(result)

    by_name = {result.name: result for result in results}
    compile_fwd = by_name.get(f"{label} compile forward")
    compile_bwd = by_name.get(f"{label} compile backward")
    tk_fwd = by_name[f"{label} tk fused forward"]
    tk_bwd = by_name[f"{label} tk fused backward"]
    tk_warp4_bwd = by_name[f"{label} tk warp4 backward"]
    tk_warp4_fwd_bwd = by_name[f"{label} tk warp4 fwd+bwd"]
    if compile_fwd is not None:
        print(f"RESULT {label} forward_vs_compile_speedup={compile_fwd.us / tk_fwd.us:.3f}x", flush=True)
    if compile_bwd is not None:
        print(f"RESULT {label} backward_vs_compile_speedup={compile_bwd.us / tk_bwd.us:.3f}x", flush=True)
        print(f"RESULT {label} backward_warp4_vs_compile_speedup={compile_bwd.us / tk_warp4_bwd.us:.3f}x", flush=True)
    compile_autograd_fwd_bwd = by_name.get(f"{label} compile autograd fwd+bwd")
    if compile_autograd_fwd_bwd is not None:
        print(
            f"RESULT {label} fwd_bwd_warp4_vs_compile_autograd_speedup="
            f"{compile_autograd_fwd_bwd.us / tk_warp4_fwd_bwd.us:.3f}x",
            flush=True,
        )
    return ok


def profile_train_shape(
    batch: int,
    tokens: int,
    dim: int,
    warmup: int,
    iters: int,
    eps: float,
    *,
    compile_max_autotune: bool,
    compile_fixed_shapes: bool,
) -> bool:
    label = f"B{batch}T{tokens}D{dim}"
    print(f"\nLayerNorm+AdaLN autograd train profile {label}", flush=True)
    ok = validate_shape(batch, tokens, dim, eps)

    eager_groups = make_train_groups(batch, tokens, dim, seed=73000 + batch + tokens)
    compile_groups = make_train_groups(batch, tokens, dim, seed=74000 + batch + tokens)
    custom_groups = make_train_groups(batch, tokens, dim, seed=75000 + batch + tokens)

    compiled_forward = compile_or_none(
        torch_autograd_forward,
        max_autotune=compile_max_autotune,
        fixed_shapes=compile_fixed_shapes,
    )
    compiled_custom_forward = compile_or_none(
        custom_autograd_forward,
        max_autotune=compile_max_autotune,
        fixed_shapes=compile_fixed_shapes,
    )

    rows = batch * tokens
    bytes_train = rows * dim * 2 * 8 + batch * dim * 2 * 4 + rows * 4 * 2 + batch * dim * 4 * 2
    results = [
        profile_groups(
            f"{label} eager autograd train",
            eager_groups,
            lambda g: train_step(torch_autograd_forward, g, tokens, eps),
            warmup=warmup,
            iters=iters,
            bytes_moved=bytes_train,
        )
    ]
    if compiled_forward is not None:
        results.append(
            profile_groups(
                f"{label} compile autograd train",
                compile_groups,
                lambda g: train_step(compiled_forward, g, tokens, eps),
                warmup=warmup,
                iters=iters,
                bytes_moved=bytes_train,
            )
        )
    results.append(
        profile_groups(
            f"{label} custom autograd train",
            custom_groups,
            lambda g: train_step(custom_autograd_forward, g, tokens, eps),
            warmup=warmup,
            iters=iters,
            bytes_moved=bytes_train,
        )
    )
    if compiled_custom_forward is not None:
        results.append(
            profile_groups(
                f"{label} custom+compile autograd train",
                custom_groups,
                lambda g: train_step(compiled_custom_forward, g, tokens, eps),
                warmup=warmup,
                iters=iters,
                bytes_moved=bytes_train,
            )
        )

    for result in results:
        print_bench(result)

    by_name = {result.name: result for result in results}
    eager = by_name[f"{label} eager autograd train"]
    custom = by_name[f"{label} custom autograd train"]
    compile_train = by_name.get(f"{label} compile autograd train")
    custom_compile = by_name.get(f"{label} custom+compile autograd train")
    print(f"RESULT {label} custom_vs_eager_train_speedup={eager.us / custom.us:.3f}x", flush=True)
    if compile_train is not None:
        print(f"RESULT {label} custom_vs_compile_train_speedup={compile_train.us / custom.us:.3f}x", flush=True)
    if custom_compile is not None and compile_train is not None:
        print(f"RESULT {label} custom_compile_vs_compile_train_speedup={compile_train.us / custom_compile.us:.3f}x", flush=True)
    return ok


def parse_shape(text: str) -> tuple[int, int]:
    batch, tokens = text.lower().replace("b", "").replace("t", "").split("x")
    return int(batch), int(tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=["64x1024", "80x1024", "16x4096", "20x4096"])
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--compile-max-autotune", action="store_true")
    parser.add_argument("--compile-fixed-shapes", action="store_true")
    parser.add_argument("--autograd-baseline", action="store_true")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-shape", default="B64T1024D1024")
    args = parser.parse_args()

    ok = True
    for shape in args.shapes:
        batch, tokens = parse_shape(shape)
        if args.train:
            ok = profile_train_shape(
                batch,
                tokens,
                args.dim,
                args.warmup,
                args.iters,
                args.eps,
                compile_max_autotune=args.compile_max_autotune,
                compile_fixed_shapes=args.compile_fixed_shapes,
            ) and ok
        else:
            ok = profile_shape(
                batch,
                tokens,
                args.dim,
                args.warmup,
                args.iters,
                args.eps,
                compile_max_autotune=args.compile_max_autotune,
                compile_fixed_shapes=args.compile_fixed_shapes,
                include_autograd_baseline=args.autograd_baseline,
                trace_dir=args.trace_dir,
                trace_shape=args.trace_shape,
            ) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
