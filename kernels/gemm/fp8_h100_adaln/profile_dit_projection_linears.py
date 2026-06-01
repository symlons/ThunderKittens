from __future__ import annotations

import statistics

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

from profile_dit_block_fp8_linear_e2e import fp8_linear


def event_bench(fn, *, warmup: int, iters: int) -> dict[str, float]:
    times: list[float] = []
    for idx in range(warmup + iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        if idx >= warmup:
            times.append(start.elapsed_time(end))
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }


class TkLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16) * 0.02)
        self.bias = nn.Parameter(torch.randn((out_features,), device="cuda", dtype=torch.bfloat16) * 0.02)
        self.register_buffer("quant_scale", torch.ones((1,), device="cuda", dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_linear(x, self.weight, self.bias, self.quant_scale)


def zero_grads(module: nn.Module, x: torch.Tensor) -> None:
    x.grad = None
    for param in module.parameters():
        param.grad = None


def step_tk(module: nn.Module, x: torch.Tensor, grad: torch.Tensor) -> None:
    out = module(x)
    out.backward(grad)
    zero_grads(module, x)


def step_te(module: nn.Module, x: torch.Tensor, grad: torch.Tensor, recipe: DelayedScaling) -> None:
    with te.autocast(enabled=True, recipe=recipe):
        out = module(x)
    out.backward(grad)
    zero_grads(module, x)


def copy_to_te(src: TkLinear, dst: te.ops.Linear) -> None:
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        dst.bias.copy_(src.bias)


def run_case(name: str, rows: int, in_features: int, out_features: int, warmup: int, iters: int) -> None:
    x0 = torch.randn((rows, in_features), device="cuda", dtype=torch.bfloat16)
    grad = torch.randn((rows, out_features), device="cuda", dtype=torch.bfloat16)
    tk = TkLinear(in_features, out_features).train()
    tel = te.ops.Linear(in_features, out_features, bias=True, dtype=torch.bfloat16).cuda().train()
    copy_to_te(tk, tel)
    recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16)

    xt = x0.detach().clone().requires_grad_(True)
    xe = x0.detach().clone().requires_grad_(True)
    step_tk(tk, xt, grad)
    step_te(tel, xe, grad, recipe)
    torch.cuda.synchronize()

    tk_stats = event_bench(lambda: step_tk(tk, xt, grad), warmup=warmup, iters=iters)
    te_stats = event_bench(lambda: step_te(tel, xe, grad, recipe), warmup=warmup, iters=iters)
    print(
        f"{name}: M={rows} K={in_features} N={out_features} "
        f"tk_mean={tk_stats['mean']:.6f} tk_med={tk_stats['median']:.6f} "
        f"te_mean={te_stats['mean']:.6f} te_med={te_stats['median']:.6f} "
        f"tk_vs_te={te_stats['mean'] / tk_stats['mean']:.3f}x",
        flush=True,
    )


def main() -> None:
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    warmup = 10
    iters = 20
    hidden = 1024
    mlp_hidden = 4096
    tokens = 1024
    cases = [
        ("qkv", hidden, 3 * hidden),
        ("attn_out", hidden, hidden),
        ("fc1", hidden, mlp_hidden),
        ("fc2", mlp_hidden, hidden),
    ]
    for batch in [4, 8, 16, 32, 64]:
        rows = batch * tokens
        print(f"\nB={batch} T={tokens}", flush=True)
        for name, in_features, out_features in cases:
            run_case(name, rows, in_features, out_features, warmup, iters)


if __name__ == "__main__":
    main()
