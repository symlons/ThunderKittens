"""
Unified harness for benchmarks and correctness tests.

Usage:
    python3 harness.py bench            # run all benchmarks
    python3 harness.py test             # run all correctness tests
    python3 harness.py bench custom_fwd # benchmark single component
    python3 harness.py test gelu fused  # specific correctness tests
    python3 harness.py list             # list registered components
    python3 harness.py bench --report BENCH_REPORT.md
    python3 harness.py test  --report CORRECTNESS_REPORT.md
"""
import argparse
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import _C
import _gelu_bwd
import _linear_bwd
import _linear_bwd_fused
from tk_bench import (
    DEFAULT_WARMUP,
    DEFAULT_ITERS,
    DEFAULT_COOLDOWN_S,
    BenchResult,
    input_group_count,
    l2_cache_size_bytes,
    profile_groups,
    uniform_bf16,
)
from harness_render import render_bench_console, write_bench_report
from correctness_utils import (
    CorrectnessSuite,
    Report,
    gelu_bwd_inputs,
    linear_inputs,
)
from correctness_reference import (
    gelu_bwd_fp32,
    gelu_bwd_fp32_rounded,
    gelu_bwd_autocast_bf16,
    gelu_bwd_raw_bf16,
    gelu_bwd_cublas_bf16,
    linear_bwd_fp32,
    linear_bwd_fp32_rounded,
    linear_bwd_autocast_bf16,
    linear_bwd_raw_bf16,
    linear_bwd_cublas_bf16,
    forward_fp32,
    forward_autocast_bf16,
    forward_raw_bf16,
)

# ---- Defaults ----

DEFAULT_M = 4096
DEFAULT_K = 4096
DEFAULT_N = 4096

# ---- Shared workspace ----

@dataclass
class CUDAWorkspace:
    """Pre-allocated tensors that defeat L2 cache across all workloads."""
    M: int
    K: int
    N: int
    seed_base: int = 42

    def create(self) -> dict[str, torch.Tensor]:
        M, K, N = self.M, self.K, self.N
        sb = self.seed_base
        return {
            "x":      uniform_bf16((M, K), sb),
            "W":      uniform_bf16((K, N), sb + 1),
            "b":      uniform_bf16((1, N), sb + 2),
            "dy":     uniform_bf16((M, N), sb + 3),
            "y":      torch.empty(M, N, device="cuda", dtype=torch.bfloat16),
            "preact": torch.empty(M, N, device="cuda", dtype=torch.bfloat16),
            "dz":     torch.empty(M, N, device="cuda", dtype=torch.bfloat16),
            "db":     torch.empty(N, device="cuda", dtype=torch.float32),
            "dW":     torch.empty(K, N, device="cuda", dtype=torch.bfloat16),
            "dx":     torch.empty(M, K, device="cuda", dtype=torch.bfloat16),
            "dz_f":   torch.empty(M, N, device="cuda", dtype=torch.bfloat16),
            "db_f":   torch.empty(N, device="cuda", dtype=torch.float32),
            "dW_f":   torch.empty(K, N, device="cuda", dtype=torch.bfloat16),
            "dx_f":   torch.empty(M, K, device="cuda", dtype=torch.bfloat16),
        }


# ---- Harness context passed to each component ----

@dataclass
class Context:
    """Single source of truth for shape, FLOPs, workspace, and torch state."""
    M: int
    K: int
    N: int
    seed: int = 42
    groups: list[dict[str, torch.Tensor]] = field(default_factory=list)
    torch_groups: list[dict[str, torch.Tensor]] = field(default_factory=list)
    linear: Optional[nn.Linear] = None
    compiled_fwd: Optional[Callable] = None
    bench_results: list[BenchResult] = field(default_factory=list)
    correctness_report: Optional[Report] = None

    # -- Derived configuration --

    @property
    def flops(self) -> float:
        """FLOPs for a single M×K × K×N GEMM."""
        return 2.0 * self.M * self.K * self.N

    @property
    def flops_fwdbwd(self) -> float:
        """FLOPs for a full fwd+bwd step (fwd gemm + dW gemm + dx gemm)."""
        return 3.0 * self.flops

    @property
    def input_bytes(self) -> int:
        """Bytes needed to defeat L2 for one group of input tensors (x, dy, b in bf16/f32)."""
        return 8 * self.M * self.N * 2 + self.N * 2 + 2 * self.N * 4

    @property
    def group_count(self) -> int:
        """Number of L2-defeating input groups."""
        return input_group_count(self.input_bytes)

    # -- Torch preparation --

    def ensure_torch(self):
        """Lazy-initialize shared torch state exactly once."""
        if self.linear is None:
            self.linear = nn.Linear(self.K, self.N, bias=True, device="cuda", dtype=torch.bfloat16)

        if self.compiled_fwd is None:
            linear = self.linear
            def fwd(x):
                return F.gelu(F.linear(x, linear.weight, linear.bias), approximate="tanh")
            self.compiled_fwd = torch.compile(fwd, mode="max-autotune")

        if not self.torch_groups:
            self.torch_groups = [
                {"x": g["x"].detach().clone().requires_grad_(True), "dy": g["dy"]}
                for g in self.groups
            ]

    # -- Bench result helpers --

    def add_result(self, r: BenchResult):
        self.bench_results.append(r)


# ---- Registry ----

class Registry:
    _benches: list[tuple[str, str, Callable]] = []
    _correctness: list[tuple[str, str, Callable]] = []

    @classmethod
    def bench(cls, name: str, desc: str):
        def wrap(fn):
            cls._benches.append((name, desc, fn))
            return fn
        return wrap

    @classmethod
    def correctness(cls, name: str, desc: str):
        def wrap(fn):
            cls._correctness.append((name, desc, fn))
            return fn
        return wrap

    @classmethod
    def list_benches(cls) -> list[tuple[str, str]]:
        return [(n, d) for n, d, _ in cls._benches]

    @classmethod
    def list_correctness(cls) -> list[tuple[str, str]]:
        return [(n, d) for n, d, _ in cls._correctness]

    @classmethod
    def run_bench(cls, name: Optional[str] = None, **ctx_kwargs) -> list[BenchResult]:
        ctx = Context(**ctx_kwargs)
        ctx.ensure_torch()
        for n, _, fn in cls._benches:
            if name and n != name:
                continue
            fn(ctx)
        return ctx.bench_results

    @classmethod
    def run_correctness(cls, names: Optional[list[str]] = None, **ctx_kwargs) -> Report:
        report = Report()
        report.shape_info = f"M={ctx_kwargs.get('M', DEFAULT_M)}, K={ctx_kwargs.get('K', DEFAULT_K)}, N={ctx_kwargs.get('N', DEFAULT_N)}"
        ctx = Context(correctness_report=report, **ctx_kwargs)
        target = names or [n for n, _, _ in cls._correctness]
        for n, _, fn in cls._correctness:
            if n not in target:
                continue
            fn(ctx)
        return report

# ---- Bench helpers ----

def profile_torch_backward_only(name: str, groups: list[dict], make_loss: Callable) -> BenchResult:
    """Time only autograd backward; forward graph construction outside event window."""
    torch.cuda.synchronize()
    for i in range(DEFAULT_WARMUP):
        loss = make_loss(groups[i % len(groups)])
        torch.cuda.synchronize()
        loss.backward()
    total_ms = 0.0
    for i in range(DEFAULT_ITERS):
        loss = make_loss(groups[i % len(groups)])
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
    return BenchResult(name=name, us=total_ms * 1000.0 / DEFAULT_ITERS, groups=len(groups))

# ---- Registered benchmark components ----

def _mk_cublas_groups(
    M: int, K: int, N: int, seed: int, 
    layout: str,  # "dw", "dx", or "ab"
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build L2-defeating input groups for a cuBLAS GEMM variant.

    layout: 'dw' -> x^T @ dz  (A^T B),   'dx' -> dz @ W^T  (A B^T),   'ab' -> A @ B
    Each group returns (input1, input2, output) for torch.mm with out=.
    """
    bpe = 2  # bf16
    
    if layout == "dw":
        bytes_per = (M * K + M * N + K * N) * bpe
        def mk(i):
            x  = uniform_bf16((M, K), seed + i * 100)
            dz = uniform_bf16((M, N), seed + i * 100 + 1)
            dW = torch.empty(K, N, device="cuda", dtype=torch.bfloat16)
            return (x, dz, dW)
    elif layout == "dx":
        bytes_per = (M * N + K * N + M * K) * bpe
        def mk(i):
            dz = uniform_bf16((M, N), seed + i * 100)
            W  = uniform_bf16((K, N), seed + i * 100 + 1)
            dx = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
            return (dz, W, dx)
    else:
        bytes_per = (M * K + K * N + M * N) * bpe
        def mk(i):
            A = uniform_bf16((M, K), seed + i * 100)
            B = uniform_bf16((K, N), seed + i * 100 + 1)
            C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            return (A, B, C)
    
    gc = input_group_count(bytes_per)
    return [mk(i) for i in range(gc)]


@Registry.bench("cublas_dW", "cuBLAS dW = x^T @ dz (A^T B)")
def bench_cublas_dW(ctx: Context):
    groups = _mk_cublas_groups(ctx.M, ctx.K, ctx.N, ctx.seed, "dw")
    r = profile_groups(
        "cublas_dW", groups,
        lambda t: torch.mm(t[0].T, t[1], out=t[2]),
        flops=ctx.flops,
    )
    ctx.add_result(r)


@Registry.bench("cublas_dx", "cuBLAS dx = dz @ W^T (A B^T)")
def bench_cublas_dx(ctx: Context):
    groups = _mk_cublas_groups(ctx.M, ctx.K, ctx.N, ctx.seed, "dx")
    r = profile_groups(
        "cublas_dx", groups,
        lambda t: torch.mm(t[0], t[1].T, out=t[2]),
        flops=ctx.flops,
    )
    ctx.add_result(r)


@Registry.bench("cublas_ab", "cuBLAS C = A @ B (reference)")
def bench_cublas_ab(ctx: Context):
    groups = _mk_cublas_groups(ctx.M, ctx.K, ctx.N, ctx.seed, "ab")
    r = profile_groups(
        "cublas_ab", groups,
        lambda t: torch.mm(t[0], t[1], out=t[2]),
        flops=ctx.flops,
    )
    ctx.add_result(r)


@Registry.bench("custom_fwd", "Forward kernel (gemm_custom)")
def bench_custom_fwd(ctx: Context):
    r = profile_groups("custom_fwd", ctx.groups,
                        lambda g: _C.gemm_custom(g["x"], g["W"], g["y"], g["b"], g["preact"]),
                        flops=ctx.flops)
    ctx.add_result(r)


@Registry.bench("custom_bwd_unfused", "Unfused backward (gelu_bwd + dw + dx)")
def bench_custom_bwd_unfused(ctx: Context):
    def fn(g):
        _linear_bwd.gelu_bwd_bias(g["dy"], g["preact"], g["dz"], g["db"])
        _linear_bwd.dw_gemm(g["x"], g["dz"], g["dW"])
        _linear_bwd.dx_gemm(g["dz"], g["W"], g["dx"])
    r = profile_groups("custom_bwd_unfused", ctx.groups, fn)
    ctx.add_result(r)


@Registry.bench("custom_bwd_fused", "Fused backward")
def bench_custom_bwd_fused(ctx: Context):
    def fn(g):
        _linear_bwd_fused.gelu_bwd_bias(g["dy"], g["preact"], g["dz_f"], g["db_f"])
        _linear_bwd_fused.dw_gemm(g["x"], g["dz_f"], g["dW_f"])
        _linear_bwd_fused.dx_gemm(g["dz_f"], g["W"], g["dx_f"])
    r = profile_groups("custom_bwd_fused", ctx.groups, fn)
    ctx.add_result(r)


@Registry.bench("custom_fwdbwd_fused", "Full fused fwd+bwd training step")
def bench_custom_fwdbwd_fused(ctx: Context):
    def fn(g):
        _C.gemm_custom(g["x"], g["W"], g["y"], g["b"], g["preact"])
        _linear_bwd_fused.gelu_bwd_bias(g["dy"], g["preact"], g["dz_f"], g["db_f"])
        _linear_bwd_fused.dw_gemm(g["x"], g["dz_f"], g["dW_f"])
        _linear_bwd_fused.dx_gemm(g["dz_f"], g["W"], g["dx_f"])
    r = profile_groups("custom_fwdbwd_fused", ctx.groups, fn, flops=ctx.flops_fwdbwd)
    ctx.add_result(r)


@Registry.bench("torch_eager_fwd", "PyTorch eager forward only")
def bench_torch_eager_fwd(ctx: Context):
    linear = ctx.linear
    def fn(g):
        return F.gelu(F.linear(g["x"], linear.weight, linear.bias), approximate="tanh")
    r = profile_groups("torch_eager_fwd", ctx.torch_groups, fn, flops=ctx.flops)
    ctx.add_result(r)


@Registry.bench("torch_compile_fwd", "PyTorch torch.compile forward only")
def bench_torch_compile_fwd(ctx: Context):
    compiled = ctx.compiled_fwd
    def fn(g):
        compiled(g["x"])
    fn(ctx.torch_groups[0])
    torch.cuda.synchronize()
    fn(ctx.torch_groups[0])
    torch.cuda.synchronize()
    r = profile_groups("torch_compile_fwd", ctx.torch_groups, fn, flops=ctx.flops)
    ctx.add_result(r)


@Registry.bench("torch_eager_bwd", "PyTorch eager backward only")
def bench_torch_eager_bwd(ctx: Context):
    linear = ctx.linear
    def make_loss(g):
        linear.zero_grad(set_to_none=True)
        if g["x"].grad is not None:
            g["x"].grad = None
        return (F.gelu(F.linear(g["x"], linear.weight, linear.bias), approximate="tanh") * g["dy"]).sum()
    r = profile_torch_backward_only("torch_eager_bwd", ctx.torch_groups, make_loss)
    ctx.add_result(r)


@Registry.bench("torch_compile_bwd", "PyTorch torch.compile backward only")
def bench_torch_compile_bwd(ctx: Context):
    compiled = ctx.compiled_fwd
    linear = ctx.linear
    def make_loss(g):
        linear.zero_grad(set_to_none=True)
        if g["x"].grad is not None:
            g["x"].grad = None
        return (compiled(g["x"]) * g["dy"]).sum()
    fn = lambda g: make_loss(g).backward()
    fn(ctx.torch_groups[0])
    torch.cuda.synchronize()
    r = profile_torch_backward_only("torch_compile_bwd", ctx.torch_groups, make_loss)
    ctx.add_result(r)


@Registry.bench("torch_eager_fwdbwd", "PyTorch eager fwd+bwd")
def bench_torch_eager_fwdbwd(ctx: Context):
    linear = ctx.linear
    def fn(g):
        linear.zero_grad(set_to_none=True)
        if g["x"].grad is not None:
            g["x"].grad = None
        (F.gelu(F.linear(g["x"], linear.weight, linear.bias), approximate="tanh") * g["dy"]).sum().backward()
    r = profile_groups("torch_eager_fwdbwd", ctx.torch_groups, fn, flops=ctx.flops_fwdbwd)
    ctx.add_result(r)


@Registry.bench("torch_compile_fwdbwd", "PyTorch torch.compile fwd+bwd")
def bench_torch_compile_fwdbwd(ctx: Context):
    compiled = ctx.compiled_fwd
    linear = ctx.linear
    def fn(g):
        linear.zero_grad(set_to_none=True)
        if g["x"].grad is not None:
            g["x"].grad = None
        (compiled(g["x"]) * g["dy"]).sum().backward()
    fn(ctx.torch_groups[0])
    torch.cuda.synchronize()
    fn(ctx.torch_groups[0])
    torch.cuda.synchronize()
    r = profile_groups("torch_compile_fwdbwd", ctx.torch_groups, fn, flops=ctx.flops_fwdbwd)
    ctx.add_result(r)


@Registry.bench("mlp_e2e_eager", "End-to-end Mlp eager fwd+bwd training step")
def bench_mlp_e2e_eager(ctx: Context):
    H = ctx.K
    hidden = H * 4
    mlp = nn.Sequential(
        nn.Linear(H, hidden, bias=True, device="cuda", dtype=torch.bfloat16),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden, H, bias=True, device="cuda", dtype=torch.bfloat16),
    )
    def fn(g):
        mlp.zero_grad(set_to_none=True)
        if g["x"].grad is not None:
            g["x"].grad = None
        y = mlp(g["x"])
        loss = (y * g["dy"]).sum()
        loss.backward()
    mlp_flops = 48.0 * ctx.M * H * H
    r = profile_groups("mlp_e2e_eager", ctx.torch_groups, fn, flops=mlp_flops)
    ctx.add_result(r)


@Registry.bench("mlp_e2e_compile", "End-to-end Mlp compiled fwd+bwd training step")
def bench_mlp_e2e_compile(ctx: Context):
    H = ctx.K
    hidden = H * 4
    mlp = nn.Sequential(
        nn.Linear(H, hidden, bias=True, device="cuda", dtype=torch.bfloat16),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden, H, bias=True, device="cuda", dtype=torch.bfloat16),
    )
    mlp = torch.compile(mlp, mode="max-autotune")
    def fn(g):
        mlp.zero_grad(set_to_none=True)
        if g["x"].grad is not None:
            g["x"].grad = None
        y = mlp(g["x"])
        loss = (y * g["dy"]).sum()
        loss.backward()
    fn(ctx.torch_groups[0])
    torch.cuda.synchronize()
    fn(ctx.torch_groups[0])
    torch.cuda.synchronize()
    mlp_flops = 48.0 * ctx.M * H * H
    r = profile_groups("mlp_e2e_compile", ctx.torch_groups, fn, flops=mlp_flops)
    ctx.add_result(r)

# ---- Registered correctness components ----

@Registry.correctness("gelu", "GELU backward only")
def test_gelu(ctx: Context):
    suite = CorrectnessSuite(name="gelu_backward")
    inp = gelu_bwd_inputs(ctx.M, ctx.N, ctx.seed)
    preact = inp["preact"]
    grad_output = inp["grad_output"]
    custom_dz = torch.empty_like(preact)
    _gelu_bwd.gelu_backward(grad_output, preact, custom_dz)
    torch.cuda.synchronize()
    spec = suite.add_tensor("dz", custom_dz)
    spec.add_baseline("fp32", gelu_bwd_fp32(preact, grad_output)["dz"])
    spec.add_baseline("fp32_rounded", gelu_bwd_fp32_rounded(preact, grad_output)["dz"])
    spec.add_baseline("autocast_bf16", gelu_bwd_autocast_bf16(preact, grad_output)["dz"])
    spec.add_baseline("raw_bf16", gelu_bwd_raw_bf16(preact, grad_output)["dz"])
    spec.add_baseline("cublas_bf16", gelu_bwd_cublas_bf16(preact, grad_output)["dz"])
    ctx.correctness_report.add_suite(suite)


@Registry.correctness("linear", "Unfused backward kernels")
def test_linear(ctx: Context):
    suite = CorrectnessSuite(name="backward_unfused")
    inp = _run_linear_bwd(ctx, _linear_bwd)
    _add_bwd_baselines(suite, inp, ctx)
    ctx.correctness_report.add_suite(suite)


@Registry.correctness("fused", "Fused backward kernels")
def test_fused(ctx: Context):
    suite = CorrectnessSuite(name="backward_fused")
    inp = _run_linear_bwd(ctx, _linear_bwd_fused)
    _add_bwd_baselines(suite, inp, ctx)
    ctx.correctness_report.add_suite(suite)


@Registry.correctness("full", "Full backward: unfused + fused + cross-comparison")
def test_full(ctx: Context):
    """Stand-alone correctness test that runs both unfused and fused, then cross-compares."""
    unfused_suite = CorrectnessSuite(name="backward_unfused")
    fused_suite = CorrectnessSuite(name="backward_fused")

    unfused_out = _run_linear_bwd(ctx, _linear_bwd)
    fused_out = _run_linear_bwd(ctx, _linear_bwd_fused)

    refs = linear_bwd_fp32(unfused_out["x"], unfused_out["W"], unfused_out["b"], unfused_out["dy"])
    refs_rounded = linear_bwd_fp32_rounded(unfused_out["x"], unfused_out["W"], unfused_out["b"], unfused_out["dy"])
    refs_auto = linear_bwd_autocast_bf16(unfused_out["x"], unfused_out["W"], unfused_out["b"], unfused_out["dy"])
    refs_raw = linear_bwd_raw_bf16(unfused_out["x"], unfused_out["W"], unfused_out["b"], unfused_out["dy"])
    refs_cublas = linear_bwd_cublas_bf16(unfused_out["x"], unfused_out["W"], unfused_out["b"], unfused_out["dy"])

    for tensor_name, tol in [("dz", 1.0), ("db", 2.0), ("dW", 8.0), ("dx", 8.0)]:
        # Unfused spec
        uspec = unfused_suite.add_tensor(tensor_name, unfused_out[tensor_name], atol=tol)
        uspec.add_baseline("fp32", refs[tensor_name])
        uspec.add_baseline("fp32_rounded", refs_rounded[tensor_name])
        uspec.add_baseline("autocast_bf16", refs_auto[tensor_name])
        uspec.add_baseline("raw_bf16", refs_raw[tensor_name])
        uspec.add_baseline("cublas_bf16", refs_cublas[tensor_name])

        # Fused spec
        fspec = fused_suite.add_tensor(tensor_name, fused_out[tensor_name], atol=tol)
        fspec.add_baseline("fp32", refs[tensor_name])
        fspec.add_baseline("fp32_rounded", refs_rounded[tensor_name])
        fspec.add_baseline("autocast_bf16", refs_auto[tensor_name])
        fspec.add_baseline("raw_bf16", refs_raw[tensor_name])
        fspec.add_baseline("cublas_bf16", refs_cublas[tensor_name])

        # Cross-comparison: fused vs unfused
        fspec.add_baseline("vs_unfused", unfused_out[tensor_name])

    ctx.correctness_report.add_suite(unfused_suite)
    ctx.correctness_report.add_suite(fused_suite)


def _run_linear_bwd(ctx: Context, lib) -> dict[str, torch.Tensor]:
    """
    Run forward + backward using the given library module.
    Returns dict keyed by output tensor names plus the inputs needed for reference.
    """
    inp = linear_inputs(ctx.M, ctx.K, ctx.N, ctx.seed)
    x, W, b, dy = inp["x"], inp["W"], inp["b"], inp["dy"]
    M, N, K = ctx.M, ctx.N, ctx.K

    y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    preact = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    _C.gemm_custom(x, W, y, b, preact)

    dz = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    dbias = torch.empty(N, device="cuda", dtype=torch.float32)
    dW_buf = torch.empty(K, N, device="cuda", dtype=torch.bfloat16)
    dx_buf = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)

    lib.gelu_bwd_bias(dy, preact, dz, dbias)
    lib.dw_gemm(x, dz, dW_buf)
    lib.dx_gemm(dz, W, dx_buf)
    torch.cuda.synchronize()

    return {"dz": dz, "db": dbias, "dW": dW_buf, "dx": dx_buf,
            "x": x, "W": W, "b": b, "dy": dy}


def _add_bwd_baselines(suite: CorrectnessSuite, out: dict, ctx: Context):
    """Add fp32, rounded, autocast, raw, cublas_bf16 baselines for every tensor in `out`."""
    x, W, b, dy = out["x"], out["W"], out["b"], out["dy"]
    refs = linear_bwd_fp32(x, W, b, dy)
    refs_rounded = linear_bwd_fp32_rounded(x, W, b, dy)
    refs_auto = linear_bwd_autocast_bf16(x, W, b, dy)
    refs_raw = linear_bwd_raw_bf16(x, W, b, dy)
    refs_cublas = linear_bwd_cublas_bf16(x, W, b, dy)
    for tensor_name, tol in [("dz", 1.0), ("db", 2.0), ("dW", 8.0), ("dx", 8.0)]:
        spec = suite.add_tensor(tensor_name, out[tensor_name], atol=tol)
        spec.add_baseline("fp32", refs[tensor_name])
        spec.add_baseline("fp32_rounded", refs_rounded[tensor_name])
        spec.add_baseline("autocast_bf16", refs_auto[tensor_name])
        spec.add_baseline("raw_bf16", refs_raw[tensor_name])
        spec.add_baseline("cublas_bf16", refs_cublas[tensor_name])


@Registry.correctness("mlp_e2e", "End-to-end Mlp fwd+bwd correctness")
def test_mlp_e2e(ctx: Context):
    """
    Full Mlp training step: x -> fc1 -> GELU -> fc2 -> out -> loss -> backward.
    Compares custom kernel chain against fp32 and autocast baselines.
    """
    suite = CorrectnessSuite(name="mlp_end_to_end")
    M, K, N = ctx.M, ctx.K, ctx.N
    H = K
    hidden = H * 4
    seed = ctx.seed

    torch.manual_seed(seed)
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    mlp_fp32 = nn.Sequential(
        nn.Linear(H, hidden, bias=True, device="cuda", dtype=torch.float32),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden, N, bias=True, device="cuda", dtype=torch.float32),
    )
    mlp_bf16 = nn.Sequential(
        nn.Linear(H, hidden, bias=True, device="cuda", dtype=torch.bfloat16),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden, N, bias=True, device="cuda", dtype=torch.bfloat16),
    )

    for idx in [0, 2]:
        mlp_bf16[idx].weight.data = mlp_fp32[idx].weight.data.clone().to(torch.bfloat16)
        mlp_bf16[idx].bias.data = mlp_fp32[idx].bias.data.clone().to(torch.bfloat16)

    x_fp32 = x.clone().float().requires_grad_(True)
    dy_fp32 = dy.clone().float()

    mlp_fp32.zero_grad()
    y_fp32 = mlp_fp32(x_fp32)
    loss_fp32 = (y_fp32 * dy_fp32).sum()
    loss_fp32.backward()

    mlp_bf16.zero_grad()
    x_bf16 = x.detach().clone().requires_grad_(True)
    y_bf16 = mlp_bf16(x_bf16)
    loss_bf16 = (y_bf16 * dy).sum()
    loss_bf16.backward()

    w1_ref = mlp_fp32[0].weight.grad.detach()
    b1_ref = mlp_fp32[0].bias.grad.detach()
    w2_ref = mlp_fp32[2].weight.grad.detach()
    b2_ref = mlp_fp32[2].bias.grad.detach()
    x_ref = x_fp32.grad.detach()

    w1_bf16 = mlp_bf16[0].weight.grad.detach()
    b1_bf16 = mlp_bf16[0].bias.grad.detach()
    w2_bf16 = mlp_bf16[2].weight.grad.detach()
    b2_bf16 = mlp_bf16[2].bias.grad.detach()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        mlp_bf16.zero_grad()
        x_auto = x.detach().clone().requires_grad_(True)
        y_auto = mlp_bf16(x_auto)
        loss_auto = (y_auto * dy).sum()
        loss_auto.backward()
        torch.cuda.synchronize()

    for tensor_name, custom, fp32_ref, auto, tol in [
        ("dw1", w1_bf16, w1_ref, mlp_bf16[0].weight.grad.detach(), 8.0),
        ("db1", b1_bf16, b1_ref, mlp_bf16[0].bias.grad.detach(), 4.0),
        ("dw2", w2_bf16, w2_ref, mlp_bf16[2].weight.grad.detach(), 8.0),
        ("db2", b2_bf16, b2_ref, mlp_bf16[2].bias.grad.detach(), 4.0),
        ("dx", x_bf16.grad.detach(), x_ref, x_auto.grad.detach(), 8.0),
    ]:
        spec = suite.add_tensor(tensor_name, custom, atol=tol)
        spec.add_baseline("fp32", fp32_ref)
        spec.add_baseline("autocast_bf16", auto)

    ctx.correctness_report.add_suite(suite)

# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(description="Unified benchmark & correctness harness")
    sub = parser.add_subparsers(dest="mode", help="Mode: bench, test, or list")

    bench_p = sub.add_parser("bench", help="Run benchmarks")
    bench_p.add_argument("components", nargs="*", default=[], help="Component names (default: all)")
    bench_p.add_argument("--M", type=int, default=DEFAULT_M)
    bench_p.add_argument("--K", type=int, default=DEFAULT_K)
    bench_p.add_argument("--N", type=int, default=DEFAULT_N)
    bench_p.add_argument("--report", type=str, default=None)

    test_p = sub.add_parser("test", help="Run correctness tests")
    test_p.add_argument("tests", nargs="*", default=[], help="Test names (default: all)")
    test_p.add_argument("--M", type=int, default=DEFAULT_M)
    test_p.add_argument("--K", type=int, default=DEFAULT_K)
    test_p.add_argument("--N", type=int, default=DEFAULT_N)
    test_p.add_argument("--seed", type=int, default=42)
    test_p.add_argument("--report", type=str, default=None)

    sub.add_parser("list", help="List available components")

    args = parser.parse_args()

    if args.mode == "list":
        print("Benchmarks:")
        for name, desc in Registry.list_benches():
            print(f"  {name:24s}  {desc}")
        print("\nCorrectness tests:")
        for name, desc in Registry.list_correctness():
            print(f"  {name:24s}  {desc}")
        return

    if args.mode == "bench":
        M, K, N = args.M, args.K, args.N
        torch.manual_seed(42)
        gc = input_group_count(8 * M * N * 2 + N * 2 + 2 * N * 4)
        groups = [CUDAWorkspace(M, K, N).create() for _ in range(gc)]
        print(
            f"Benchmark convention: {DEFAULT_WARMUP} warmup, {DEFAULT_ITERS} iters, "
            f"{gc} L2-defeating groups, 2 CUDA events"
        )
        print(f"L2 cache: {l2_cache_size_bytes() / (1024*1024):.0f} MB")
        target = None if not args.components else (args.components[0] if len(args.components) == 1 else None)
        if len(args.components) > 1:
            for comp in args.components:
                results = Registry.run_bench(name=comp, M=M, K=K, N=N, seed=42, groups=groups)
                print(render_bench_console(results))
        else:
            results = Registry.run_bench(name=target, M=M, K=K, N=N, seed=42, groups=groups)
            print(render_bench_console(results))
        if args.report:
            write_bench_report(args.report, results, M, K, N)
            print(f"\nReport -> {args.report}")

    elif args.mode == "test":
        M, K, N = args.M, args.K, args.N
        seed = args.seed
        gc = input_group_count(8 * M * N * 2)
        groups = [CUDAWorkspace(M, K, N).create() for _ in range(gc)]
        target = args.tests or None
        report = Registry.run_correctness(names=target, M=M, K=K, N=N, seed=seed, groups=groups)
        print(report.console_table())
        if args.report:
            report.write_markdown(args.report, all_pairwise=False)
            print(f"\nReport -> {args.report}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()