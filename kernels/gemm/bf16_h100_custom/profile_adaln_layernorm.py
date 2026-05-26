from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from dit3d_e2e_bench import tk_layernorm_adaln_backward_op, tk_layernorm_adaln_op
from harness import reference_backward, reference_forward, run_fused_backward, run_fused_forward
from tk_bench import check_close, input_group_count, print_bench, profile_groups, uniform_bf16


def compile_or_none(fn: Callable):
    try:
        return torch.compile(fn)
    except Exception as exc:
        print(f"torch.compile unavailable for {getattr(fn, '__name__', repr(fn))}: {exc!r}", flush=True)
        return None


def torch_autograd_forward(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, tokens: int, eps: float):
    batch = shift.shape[0]
    x3 = x.reshape(batch, tokens, x.shape[1])
    out = torch.nn.functional.layer_norm(x3, (x.shape[1],), eps=eps)
    out = out * (1.0 + scale[:, None, :]) + shift[:, None, :]
    return out.reshape_as(x)


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

    ref_dx, ref_dshift, ref_dscale = reference_backward(grad, x, scale, ref_mean, ref_rstd, tokens)
    tk_dx, tk_dshift, tk_dscale = run_fused_backward(grad, x, scale, tk_mean, tk_rstd, tokens)
    ok = check_close("backward dx", tk_dx, ref_dx, atol=2e-2) and ok
    ok = check_close("backward dshift", tk_dshift, ref_dshift, atol=5e-2) and ok
    ok = check_close("backward dscale", tk_dscale, ref_dscale, atol=5e-2) and ok
    return ok


def profile_shape(batch: int, tokens: int, dim: int, warmup: int, iters: int, eps: float) -> bool:
    label = f"B{batch}T{tokens}D{dim}"
    print(f"\nLayerNorm+AdaLN profile {label}", flush=True)
    ok = validate_shape(batch, tokens, dim, eps)
    groups = make_groups(batch, tokens, dim, eps, seed=72000 + batch + tokens)

    compiled_forward = compile_or_none(lambda x, shift, scale: reference_forward(x, shift, scale, tokens, eps)[0])
    compiled_backward = compile_or_none(
        lambda grad, x, scale, mean, rstd: reference_backward(grad, x, scale, mean, rstd, tokens)
    )

    rows = batch * tokens
    bytes_forward = rows * dim * 2 * 2 + batch * dim * 2 * 2 + rows * 4 * 2
    bytes_backward = rows * dim * 2 * 4 + batch * dim * 2 + rows * 4 * 2 + batch * dim * 4 * 2
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
                warmup=max(2, min(warmup, 10)),
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
                warmup=max(2, min(warmup, 10)),
                iters=iters,
                bytes_moved=bytes_backward,
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

    for result in results:
        print_bench(result)

    by_name = {result.name: result for result in results}
    compile_fwd = by_name.get(f"{label} compile forward")
    compile_bwd = by_name.get(f"{label} compile backward")
    tk_fwd = by_name[f"{label} tk fused forward"]
    tk_bwd = by_name[f"{label} tk fused backward"]
    if compile_fwd is not None:
        print(f"RESULT {label} forward_vs_compile_speedup={compile_fwd.us / tk_fwd.us:.3f}x", flush=True)
    if compile_bwd is not None:
        print(f"RESULT {label} backward_vs_compile_speedup={compile_bwd.us / tk_bwd.us:.3f}x", flush=True)
    return ok


def profile_train_shape(batch: int, tokens: int, dim: int, warmup: int, iters: int, eps: float) -> bool:
    label = f"B{batch}T{tokens}D{dim}"
    print(f"\nLayerNorm+AdaLN autograd train profile {label}", flush=True)
    ok = validate_shape(batch, tokens, dim, eps)

    eager_groups = make_train_groups(batch, tokens, dim, seed=73000 + batch + tokens)
    compile_groups = make_train_groups(batch, tokens, dim, seed=74000 + batch + tokens)
    custom_groups = make_train_groups(batch, tokens, dim, seed=75000 + batch + tokens)

    compiled_forward = compile_or_none(torch_autograd_forward)
    compiled_custom_forward = compile_or_none(custom_autograd_forward)

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
                warmup=max(2, min(warmup, 10)),
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
                warmup=max(2, min(warmup, 10)),
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
    args = parser.parse_args()

    ok = True
    for shape in args.shapes:
        batch, tokens = parse_shape(shape)
        if args.train:
            ok = profile_train_shape(batch, tokens, args.dim, args.warmup, args.iters, args.eps) and ok
        else:
            ok = profile_shape(batch, tokens, args.dim, args.warmup, args.iters, args.eps) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
