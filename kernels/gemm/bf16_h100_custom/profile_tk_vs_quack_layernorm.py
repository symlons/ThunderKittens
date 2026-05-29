from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import _C
from liger_kernel.ops.layer_norm import layer_norm_backward as liger_layer_norm_backward
from liger_kernel.ops.layer_norm import layer_norm_forward as liger_layer_norm_forward
from quack.rmsnorm import layernorm_fwd
from tk_bench import check_close, input_group_count, print_bench, profile_groups, uniform_bf16


def parse_shape(text: str) -> tuple[int, int]:
    batch, tokens = text.lower().replace("b", "").replace("t", "").split("x")
    return int(batch), int(tokens)


def tk_layernorm(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, tokens: int, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    mean = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    _C.layernorm_adaln(x, shift, scale, out, mean, rstd, tokens, eps)
    return out


def tk_layernorm_variant(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    tokens: int,
    eps: float,
    variant: str,
) -> torch.Tensor:
    out, _, _ = tk_layernorm_variant_with_stats(x, shift, scale, tokens, eps, variant)
    return out


def tk_layernorm_variant_with_stats(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    tokens: int,
    eps: float,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if variant == "cta":
        out = torch.empty_like(x)
        mean = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        _C.layernorm_adaln(x, shift, scale, out, mean, rstd, tokens, eps)
        return out, mean, rstd
    out = torch.empty_like(x)
    mean = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    if variant == "persistent":
        _C.layernorm_adaln_persistent(x, shift, scale, out, mean, rstd, tokens, eps)
    elif variant == "warp4":
        _C.layernorm_adaln_warp4(x, shift, scale, out, mean, rstd, tokens, eps)
    else:
        raise ValueError(f"unsupported TK variant: {variant}")
    return out, mean, rstd


def make_groups(batch: int, tokens: int, dim: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    rows = batch * tokens
    group_bytes = rows * dim * 2 * 2 + batch * dim * 2 * 2
    groups_n = min(input_group_count(group_bytes), 8)
    groups = []
    for i in range(groups_n):
        x = uniform_bf16((rows, dim), seed + i, -2.0, 2.0)
        shift = torch.zeros((batch, dim), device="cuda", dtype=torch.bfloat16)
        scale = torch.zeros_like(shift)
        groups.append((x, shift, scale))
    return groups


def tk_layernorm_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    tokens: int,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(x)
    dshift = torch.empty_like(scale, dtype=torch.float32)
    dscale = torch.empty_like(scale, dtype=torch.float32)
    if variant == "cta":
        _C.layernorm_adaln_backward(grad, x, scale, mean, rstd, dx, dshift, dscale, tokens)
    elif variant == "warp4":
        _C.layernorm_adaln_backward_warp4(grad, x, scale, mean, rstd, dx, dshift, dscale, tokens)
    else:
        raise ValueError(f"unsupported TK backward variant: {variant}")
    return dx, dshift, dscale


def make_backward_groups(
    forward_groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    weight: torch.Tensor,
    bias: torch.Tensor,
    tokens: int,
    eps: float,
    seed: int,
):
    groups = []
    for i, (x, shift, scale) in enumerate(forward_groups):
        grad = uniform_bf16(x.shape, seed + i, -1.0, 1.0)
        _, tk_mean, tk_rstd = tk_layernorm_variant_with_stats(x, shift, scale, tokens, eps, "warp4")
        _, liger_x, liger_mean, liger_rstd, _, _ = liger_layer_norm_forward(x, weight, bias, eps)
        groups.append((x, shift, scale, grad, tk_mean, tk_rstd, liger_x, liger_mean, liger_rstd))
    return groups


def profile_shape(batch: int, tokens: int, dim: int, warmup: int, iters: int, eps: float) -> bool:
    label = f"B{batch}T{tokens}D{dim}"
    rows = batch * tokens
    groups = make_groups(batch, tokens, dim, seed=91000 + batch + tokens)
    weight = torch.ones((dim,), device="cuda", dtype=torch.float32)
    bias = torch.zeros_like(weight)

    x, shift, scale = groups[0]
    ref = F.layer_norm(x.float(), (dim,), weight, bias, eps).to(torch.bfloat16)
    tk_out = tk_layernorm(x, shift, scale, tokens, eps)
    _, tk_mean, tk_rstd = tk_layernorm_variant_with_stats(x, shift, scale, tokens, eps, "warp4")
    quack_out = layernorm_fwd(x, weight, bias, eps=eps)
    liger_out, liger_x, liger_mean, liger_rstd, _, _ = liger_layer_norm_forward(x, weight, bias, eps)

    print(f"\nTK vs QuACK LayerNorm fwd {label}", flush=True)
    ok = check_close("tk vs torch", tk_out, ref, atol=2e-2)
    ok = check_close("quack vs torch", quack_out, ref, atol=2e-2) and ok
    ok = check_close("tk vs quack", tk_out, quack_out, atol=2e-2) and ok
    ok = check_close("liger vs torch", liger_out, ref, atol=2e-2) and ok
    ok = check_close("tk vs liger", tk_out, liger_out, atol=2e-2) and ok
    for variant in ("persistent", "warp4"):
        variant_out = tk_layernorm_variant(x, shift, scale, tokens, eps, variant)
        ok = check_close(f"tk {variant} vs torch", variant_out, ref, atol=2e-2) and ok
        ok = check_close(f"tk {variant} vs quack", variant_out, quack_out, atol=2e-2) and ok
        ok = check_close(f"tk {variant} vs liger", variant_out, liger_out, atol=2e-2) and ok

    grad = uniform_bf16(x.shape, 93000 + batch + tokens, -1.0, 1.0)
    ref_x = x.detach().float().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_bias = bias.detach().clone().requires_grad_(True)
    F.layer_norm(ref_x, (dim,), ref_weight, ref_bias, eps).backward(grad.float())
    tk_dx, tk_dshift, tk_dscale = tk_layernorm_backward(grad, x, scale, tk_mean, tk_rstd, tokens, "warp4")
    liger_dx, liger_dw, liger_db = liger_layer_norm_backward(grad, liger_x, weight, bias, liger_mean, liger_rstd)
    ok = check_close("tk warp4 backward dx vs torch", tk_dx, ref_x.grad.to(torch.bfloat16), atol=2e-2) and ok
    ok = check_close("liger backward dx vs torch", liger_dx, ref_x.grad.to(torch.bfloat16), atol=2e-2) and ok
    ok = check_close("liger backward dw vs torch", liger_dw.float(), ref_weight.grad, atol=1.0) and ok
    ok = check_close("liger backward db vs torch", liger_db.float(), ref_bias.grad, atol=5e-2) and ok
    ok = check_close("tk warp4 backward dx vs liger", tk_dx, liger_dx, atol=2e-2) and ok
    ok = check_close("tk dscale sum vs liger dw", tk_dscale.sum(dim=0), liger_dw.float(), atol=1.0) and ok
    ok = check_close("tk dshift sum vs liger db", tk_dshift.sum(dim=0), liger_db.float(), atol=8e-2) and ok

    elem_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    quack_bytes = rows * dim * elem_bytes * 2 + dim * 4 * 2
    tk_bytes = rows * dim * elem_bytes * 2 + batch * dim * elem_bytes * 2 + rows * 4 * 2
    liger_forward_bytes = rows * dim * elem_bytes * 2 + dim * 4 * 2 + rows * elem_bytes * 2
    liger_backward_bytes = rows * dim * elem_bytes * 3 + dim * 4 + rows * elem_bytes * 2 + dim * 4 * 2
    tk_backward_bytes = rows * dim * elem_bytes * 4 + batch * dim * elem_bytes + rows * 4 * 2 + batch * dim * 4 * 2

    tk_result = profile_groups(
        f"{label} tk fused layernorm_adaln zero-shift",
        groups,
        lambda g: tk_layernorm(g[0], g[1], g[2], tokens, eps),
        warmup=warmup,
        iters=iters,
        bytes_moved=tk_bytes,
    )
    tk_variant_results = []
    for variant in ("persistent", "warp4"):
        tk_variant_results.append(
            (
                variant,
                profile_groups(
                    f"{label} tk {variant} layernorm_adaln zero-shift",
                    groups,
                    lambda g, variant=variant: tk_layernorm_variant(g[0], g[1], g[2], tokens, eps, variant),
                    warmup=warmup,
                    iters=iters,
                    bytes_moved=tk_bytes,
                ),
            )
        )
    quack_result = profile_groups(
        f"{label} quack layernorm_fwd",
        groups,
        lambda g: layernorm_fwd(g[0], weight, bias, eps=eps),
        warmup=warmup,
        iters=iters,
        bytes_moved=quack_bytes,
    )
    liger_forward_result = profile_groups(
        f"{label} liger layer_norm_forward",
        groups,
        lambda g: liger_layer_norm_forward(g[0], weight, bias, eps)[0],
        warmup=warmup,
        iters=iters,
        bytes_moved=liger_forward_bytes,
    )

    backward_groups = make_backward_groups(groups, weight, bias, tokens, eps, seed=94000 + batch + tokens)
    tk_backward_result = profile_groups(
        f"{label} tk warp4 layernorm_adaln backward zero-shift",
        backward_groups,
        lambda g: tk_layernorm_backward(g[3], g[0], g[2], g[4], g[5], tokens, "warp4"),
        warmup=warmup,
        iters=iters,
        bytes_moved=tk_backward_bytes,
    )
    liger_backward_result = profile_groups(
        f"{label} liger layer_norm_backward",
        backward_groups,
        lambda g: liger_layer_norm_backward(g[3], g[6], weight, bias, g[7], g[8]),
        warmup=warmup,
        iters=iters,
        bytes_moved=liger_backward_bytes,
    )

    print_bench(tk_result)
    for _, result in tk_variant_results:
        print_bench(result)
    print_bench(quack_result)
    print_bench(liger_forward_result)
    print_bench(tk_backward_result)
    print_bench(liger_backward_result)
    print(f"RESULT {label} tk_vs_quack_speedup={quack_result.us / tk_result.us:.3f}x", flush=True)
    for variant, result in tk_variant_results:
        print(f"RESULT {label} tk_{variant}_vs_quack_speedup={quack_result.us / result.us:.3f}x", flush=True)
    print(f"RESULT {label} tk_warp4_vs_liger_fwd_speedup={liger_forward_result.us / dict(tk_variant_results)['warp4'].us:.3f}x", flush=True)
    print(f"RESULT {label} tk_warp4_vs_liger_bwd_speedup={liger_backward_result.us / tk_backward_result.us:.3f}x", flush=True)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=["64x1024", "80x1024", "16x4096", "20x4096"])
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    ok = True
    for shape in args.shapes:
        batch, tokens = parse_shape(shape)
        ok = profile_shape(batch, tokens, args.dim, args.warmup, args.iters, args.eps) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
