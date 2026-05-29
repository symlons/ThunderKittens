from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import _C
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


def profile_shape(batch: int, tokens: int, dim: int, warmup: int, iters: int, eps: float) -> bool:
    label = f"B{batch}T{tokens}D{dim}"
    rows = batch * tokens
    groups = make_groups(batch, tokens, dim, seed=91000 + batch + tokens)
    weight = torch.ones((dim,), device="cuda", dtype=torch.float32)
    bias = torch.zeros_like(weight)

    x, shift, scale = groups[0]
    ref = F.layer_norm(x.float(), (dim,), weight, bias, eps).to(torch.bfloat16)
    tk_out = tk_layernorm(x, shift, scale, tokens, eps)
    quack_out = layernorm_fwd(x, weight, bias, eps=eps)

    print(f"\nTK vs QuACK LayerNorm fwd {label}", flush=True)
    ok = check_close("tk vs torch", tk_out, ref, atol=2e-2)
    ok = check_close("quack vs torch", quack_out, ref, atol=2e-2) and ok
    ok = check_close("tk vs quack", tk_out, quack_out, atol=2e-2) and ok

    elem_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    quack_bytes = rows * dim * elem_bytes * 2 + dim * 4 * 2
    tk_bytes = rows * dim * elem_bytes * 2 + batch * dim * elem_bytes * 2 + rows * 4 * 2

    tk_result = profile_groups(
        f"{label} tk fused layernorm_adaln zero-shift",
        groups,
        lambda g: tk_layernorm(g[0], g[1], g[2], tokens, eps),
        warmup=warmup,
        iters=iters,
        bytes_moved=tk_bytes,
    )
    quack_result = profile_groups(
        f"{label} quack layernorm_fwd",
        groups,
        lambda g: layernorm_fwd(g[0], weight, bias, eps=eps),
        warmup=warmup,
        iters=iters,
        bytes_moved=quack_bytes,
    )

    print_bench(tk_result)
    print_bench(quack_result)
    print(f"RESULT {label} tk_vs_quack_speedup={quack_result.us / tk_result.us:.3f}x", flush=True)
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
