from __future__ import annotations

import argparse
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dit3d_e2e_bench import modulate
from tk_bench import profile_groups, uniform_bf16


class MlpOnlyBranch(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.ada(c).chunk(3, dim=1)
        h = modulate(self.norm(x), shift, scale)
        h = self.fc2(F.gelu(self.fc1(h), approximate="tanh"))
        return x + gate.unsqueeze(1) * h


class PreProjection(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.proj(modulate(self.norm(x), shift, scale))


class PostProjectionResidual(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim, bias=True))
        self.proj = nn.Linear(dim, dim)

    def forward(self, residual: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gate = self.gate(c)
        return residual + gate.unsqueeze(1) * self.proj(h)


def clone_module(module: nn.Module) -> nn.Module:
    import copy

    cloned = copy.deepcopy(module).cuda().to(torch.bfloat16)
    cloned.train()
    return cloned


def make_inputs(name: str, batch: int, tokens: int, dim: int) -> tuple[torch.Tensor, ...]:
    x = uniform_bf16((batch, tokens, dim), 100, -2.0, 2.0).requires_grad_(True)
    c = uniform_bf16((batch, dim), 101, -1.0, 1.0).requires_grad_(True)
    if name == "post_projection_residual":
        h = uniform_bf16((batch, tokens, dim), 102, -2.0, 2.0).requires_grad_(True)
        return x, h, c
    return x, c


def make_module(name: str, dim: int) -> nn.Module:
    if name == "mlp_branch":
        return MlpOnlyBranch(dim)
    if name == "pre_qkv_projection":
        return PreProjection(dim, 3 * dim)
    if name == "post_projection_residual":
        return PostProjectionResidual(dim)
    raise ValueError(name)


def train_step(module: nn.Module, inputs: tuple[torch.Tensor, ...], grad: torch.Tensor) -> None:
    out = module(*inputs)
    out.backward(grad)
    module.zero_grad(set_to_none=True)
    for tensor in inputs:
        tensor.grad = None


def explain_compile(module: nn.Module, inputs: tuple[torch.Tensor, ...], grad: torch.Tensor) -> None:
    def step(*args):
        *actual_inputs, actual_grad = args
        out = module(*actual_inputs)
        grads = torch.autograd.grad(out, tuple(module.parameters()) + tuple(actual_inputs), actual_grad, allow_unused=True)
        return tuple(g for g in grads if g is not None)

    print("\nDynamo explain:", flush=True)
    try:
        explanation = torch._dynamo.explain(step)(*inputs, grad)
        print(f"  graph_count={getattr(explanation, 'graph_count', 'NA')}", flush=True)
        print(f"  graph_break_count={getattr(explanation, 'graph_break_count', 'NA')}", flush=True)
        print(f"  op_count={getattr(explanation, 'op_count', 'NA')}", flush=True)
        break_reasons = getattr(explanation, "break_reasons", [])
        if break_reasons:
            print("  break_reasons:", flush=True)
            for reason in break_reasons:
                print(f"    - {reason}", flush=True)
        ops_per_graph = getattr(explanation, "ops_per_graph", [])
        for idx, ops in enumerate(ops_per_graph):
            print(f"  graph_{idx}_ops:", flush=True)
            for op in ops:
                print(f"    - {op}", flush=True)
    except Exception as exc:
        print(f"torch._dynamo.explain failed: {exc!r}", flush=True)


def profile_step(label: str, step: Callable[[], None], warmup: int, iters: int) -> None:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(iters):
            step()
    torch.cuda.synchronize()
    print(f"\nProfiler {label}:", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40), flush=True)


def enable_inductor_debug() -> None:
    os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    try:
        import torch._inductor.config as inductor_config

        inductor_config.debug = True
        if hasattr(inductor_config, "trace"):
            inductor_config.trace.enabled = True
            if hasattr(inductor_config.trace, "output_code"):
                inductor_config.trace.output_code = True
    except Exception as exc:
        print(f"Could not enable Inductor debug config: {exc!r}", flush=True)


def _latest_output_code_files(start_time: float) -> list[Path]:
    roots = [Path.cwd() / "torch_compile_debug", Path("/tmp")]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("output_code.py")
            if path.is_file() and path.stat().st_mtime >= start_time - 1.0
        )
    return sorted(files, key=lambda path: path.stat().st_mtime)


def _summarize_source_comments(lines: list[str], def_index: int) -> str:
    comments: list[str] = []
    for line in reversed(lines[max(0, def_index - 12):def_index]):
        stripped = line.strip()
        if stripped.startswith("# Topologically Sorted Source Nodes:"):
            comments.append(stripped.removeprefix("# ").strip())
            break
        if stripped.startswith("# Original ATen:"):
            comments.append(stripped.removeprefix("# ").strip())
    return " | ".join(reversed(comments))


def summarize_inductor_debug(case_name: str, start_time: float) -> None:
    files = _latest_output_code_files(start_time)
    print(f"\nInductor generated-code summary for {case_name}:", flush=True)
    if not files:
        print("  no output_code.py files found for this run", flush=True)
        return

    for path in files[-4:]:
        rel = path
        try:
            rel = path.relative_to(Path.cwd())
        except ValueError:
            pass
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        triton_defs: list[tuple[str, str]] = []
        for idx, line in enumerate(lines):
            match = re.search(r"def (triton_[a-zA-Z0-9_]+)\(", line)
            if match:
                triton_defs.append((match.group(1), _summarize_source_comments(lines, idx)))
        extern_calls = sorted(set(re.findall(r"extern_kernels\.([a-zA-Z0-9_]+)\(", text)))
        aten_calls = sorted(set(re.findall(r"torch\.ops\.aten\.([a-zA-Z0-9_]+)\.", text)))

        print(f"  {rel}", flush=True)
        if extern_calls:
            print(f"    extern kernels: {', '.join(extern_calls)}", flush=True)
        if aten_calls:
            print(f"    aten calls: {', '.join(aten_calls[:20])}", flush=True)
        for kernel_name, source in triton_defs[:30]:
            suffix = f" [{source}]" if source else ""
            print(f"    triton: {kernel_name}{suffix}", flush=True)
        if len(triton_defs) > 30:
            print(f"    ... {len(triton_defs) - 30} more Triton kernels", flush=True)


def run_case(
    name: str,
    batch: int,
    tokens: int,
    dim: int,
    warmup: int,
    iters: int,
    dump_inductor: bool,
    skip_profiler: bool,
) -> None:
    torch.cuda.empty_cache()
    base = clone_module(make_module(name, dim))
    debug_start = time.time()
    compiled = torch.compile(clone_module(make_module(name, dim)), fullgraph=False)
    inputs_eager = make_inputs(name, batch, tokens, dim)
    inputs_compile = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in inputs_eager)
    out_dim = 3 * dim if name == "pre_qkv_projection" else dim
    grad = uniform_bf16((batch, tokens, out_dim), 103, -1.0, 1.0)

    print(f"\n=== {name} B={batch} T={tokens} D={dim} ===", flush=True)
    explain_compile(clone_module(make_module(name, dim)), inputs_compile, grad)

    eager_result = profile_groups(
        f"{name} eager train",
        [(inputs_eager, grad)],
        lambda g: train_step(base, g[0], g[1]),
        warmup=warmup,
        iters=iters,
        cooldown_s=0.0,
    )
    compile_result = profile_groups(
        f"{name} torch.compile train",
        [(inputs_compile, grad)],
        lambda g: train_step(compiled, g[0], g[1]),
        warmup=max(2, min(warmup, 5)),
        iters=iters,
        cooldown_s=0.0,
    )
    print(f"Timing: eager={eager_result.us:.2f} us compile={compile_result.us:.2f} us speedup={eager_result.us / compile_result.us:.2f}x", flush=True)

    if dump_inductor:
        summarize_inductor_debug(name, debug_start)

    if skip_profiler:
        return
    profile_step("eager", lambda: train_step(base, inputs_eager, grad), max(1, min(warmup, 3)), iters)
    profile_step("torch.compile", lambda: train_step(compiled, inputs_compile, grad), max(1, min(warmup, 3)), iters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--cases", nargs="+", default=["mlp_branch", "pre_qkv_projection", "post_projection_residual"])
    parser.add_argument("--dump-inductor", action="store_true")
    parser.add_argument("--skip-profiler", action="store_true")
    args = parser.parse_args()

    if args.dump_inductor:
        enable_inductor_debug()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.init()
    for name in args.cases:
        run_case(name, args.batch, args.tokens, args.dim, args.warmup, args.iters, args.dump_inductor, args.skip_profiler)


if __name__ == "__main__":
    main()
