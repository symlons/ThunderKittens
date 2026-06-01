from __future__ import annotations

import sys

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
from torch.profiler import ProfilerActivity, profile

from profile_dit_block_fp8_linear_e2e import fp8_linear


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


def main() -> None:
    kind = sys.argv[1]
    rows = int(sys.argv[2])
    in_features = int(sys.argv[3])
    out_features = int(sys.argv[4])
    x = torch.randn((rows, in_features), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn((rows, out_features), device="cuda", dtype=torch.bfloat16)
    recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16)
    if kind == "tk":
        module = TkLinear(in_features, out_features).train()
        fn = lambda: step_tk(module, x, grad)
    elif kind == "te":
        module = te.ops.Linear(in_features, out_features, bias=True, dtype=torch.bfloat16).cuda().train()
        fn = lambda: step_te(module, x, grad, recipe)
    else:
        raise ValueError(kind)

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))


if __name__ == "__main__":
    main()
