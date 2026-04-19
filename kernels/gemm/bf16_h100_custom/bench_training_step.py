"""
End-to-end training step benchmark: forward + backward for Linear+GELU.
Compares: unfused custom, fused custom (bias into dx GEMM), PyTorch eager/compiled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler as prof
import _C
import _linear_bwd
import _linear_bwd_fused

torch.manual_seed(42)

M, K, N = 4096, 4096, 4096
WARMUP, ITERS = 200, 200

print(f"{'='*70}")
print(f"Training step benchmark: M={M}, K={K}, N={N}, bf16")
print(f"{'='*70}")

# ============================================================
# Setup
# ============================================================
x  = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
W  = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
b  = torch.randn(1, N, device="cuda", dtype=torch.bfloat16)
dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

# Buffers for unfused
y_custom     = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
preact       = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
dz           = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
dbias        = torch.empty(N, device="cuda", dtype=torch.float32)
dW           = torch.empty(K, N, device="cuda", dtype=torch.bfloat16)
dx           = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)

# Buffers for fused
dz_f         = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
dbias_f      = torch.empty(N, device="cuda", dtype=torch.float32)
dW_f         = torch.empty(K, N, device="cuda", dtype=torch.bfloat16)
dx_f         = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)

def profile_batch(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters

# ============================================================
# Custom kernels — unfused (3 bwd kernels)
# ============================================================
def custom_fwd_bwd():
    _C.gemm_custom(x, W, y_custom, b, preact)
    _linear_bwd.gelu_bwd_bias(dy, preact, dz, dbias)
    _linear_bwd.dw_gemm(x, dz, dW)
    _linear_bwd.dx_gemm(dz, W, dx)

def custom_fwd():
    _C.gemm_custom(x, W, y_custom, b, preact)

def custom_bwd():
    _linear_bwd.gelu_bwd_bias(dy, preact, dz, dbias)
    _linear_bwd.dw_gemm(x, dz, dW)
    _linear_bwd.dx_gemm(dz, W, dx)

# ============================================================
# Custom kernels — fused (gelu+bias combined, same GEMM kernels)
# ============================================================
def fused_bwd():
    _linear_bwd_fused.gelu_bwd_bias(dy, preact, dz_f, dbias_f)
    _linear_bwd_fused.dw_gemm(x, dz_f, dW_f)
    _linear_bwd_fused.dx_gemm(dz_f, W, dx_f)

def fused_fwd_bwd():
    _C.gemm_custom(x, W, y_custom, b, preact)
    _linear_bwd_fused.gelu_bwd_bias(dy, preact, dz_f, dbias_f)
    _linear_bwd_fused.dw_gemm(x, dz_f, dW_f)
    _linear_bwd_fused.dx_gemm(dz_f, W, dx_f)

# ============================================================
# PyTorch model
# ============================================================
linear_pt = nn.Linear(K, N, bias=True, device="cuda", dtype=torch.bfloat16)
x_pt = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dy_pt = dy.clone()

def torch_forward(x):
    return F.gelu(F.linear(x, linear_pt.weight, linear_pt.bias), approximate="tanh")

torch_forward_compiled = torch.compile(torch_forward, mode="max-autotune")

def torch_fwd_bwd():
    linear_pt.zero_grad(set_to_none=True)
    if x_pt.grad is not None:
        x_pt.grad = None
    out = torch_forward(x_pt)
    loss = (out * dy_pt).sum()
    loss.backward()

def torch_fwd_bwd_compiled():
    linear_pt.zero_grad(set_to_none=True)
    if x_pt.grad is not None:
        x_pt.grad = None
    out = torch_forward_compiled(x_pt)
    loss = (out * dy_pt).sum()
    loss.backward()

def torch_fwd_only():
    torch_forward(x_pt)

def torch_bwd_only():
    linear_pt.zero_grad(set_to_none=True)
    if x_pt.grad is not None:
        x_pt.grad = None
    out = torch_forward(x_pt)
    loss = (out * dy_pt).sum()
    loss.backward()

# ============================================================
# Correctness check: fused vs unfused
# ============================================================
_C.gemm_custom(x, W, y_custom, b, preact)
_linear_bwd.gelu_bwd_bias(dy, preact, dz, dbias)
_linear_bwd.dw_gemm(x, dz, dW)
_linear_bwd.dx_gemm(dz, W, dx)

_linear_bwd_fused.gelu_bwd_bias(dy, preact, dz_f, dbias_f)
_linear_bwd_fused.dw_gemm(x, dz_f, dW_f)
_linear_bwd_fused.dx_gemm(dz_f, W, dx_f)
torch.cuda.synchronize()

def check(name, a, b, atol=8.0):
    d = (a.float() - b.float()).abs()
    ok = d.max().item() < atol
    print(f"  {name}: max_diff={d.max().item():.4f} {'PASS' if ok else 'FAIL'}")
    return ok

print("\nFused vs Unfused correctness:")
ok_dz = check("dz",    dz_f, dz)
ok_dw = check("dW",    dW_f, dW)
ok_dx = check("dx",    dx_f, dx)
ok_db = check("dbias", dbias_f, dbias, 2.0)
print(f"  {'ALL PASS' if ok_dz and ok_dw and ok_dx and ok_db else 'SOME FAILED'}")

# ============================================================
# Benchmark
# ============================================================
t_custom_fwdbwd = profile_batch(custom_fwd_bwd)
t_custom_fwd    = profile_batch(custom_fwd)
t_custom_bwd    = profile_batch(custom_bwd)

t_fused_bwd     = profile_batch(fused_bwd)
t_fused_fwdbwd  = profile_batch(fused_fwd_bwd)

t_torch_fwdbwd  = profile_batch(torch_fwd_bwd)
t_torch_fwd     = profile_batch(torch_fwd_only)
t_torch_bwd     = profile_batch(torch_bwd_only)

t_torch_fwdbwd_comp = profile_batch(torch_fwd_bwd_compiled)

# ============================================================
# Results
# ============================================================
flops_gemm = 2.0 * M * K * N

def tflops(time_us):
    return flops_gemm / (time_us * 1e-6) / 1e12

print(f"\n{'='*70}")
print(f"Performance")
print(f"{'='*70}")

print(f"\nCustom (unfused — gelu+bias, dW, dx):")
print(f"  Forward:   {t_custom_fwd:7.1f} us  ({tflops(t_custom_fwd):.0f} TFLOPS)")
print(f"  Backward:  {t_custom_bwd:7.1f} us")
print(f"  Fwd+Bwd:   {t_custom_fwdbwd:7.1f} us")

print(f"\nCustom (fused gelu+bias, same GEMMs):")
print(f"  Backward:  {t_fused_bwd:7.1f} us")
print(f"  Fwd+Bwd:   {t_fused_fwdbwd:7.1f} us")

print(f"\nPyTorch eager:")
print(f"  Forward:   {t_torch_fwd:7.1f} us  ({tflops(t_torch_fwd):.0f} TFLOPS)")
print(f"  Backward:  {t_torch_bwd:7.1f} us")
print(f"  Fwd+Bwd:   {t_torch_fwdbwd:7.1f} us")

print(f"\nPyTorch compiled:")
print(f"  Fwd+Bwd:   {t_torch_fwdbwd_comp:7.1f} us")

# ============================================================
# Speedups
# ============================================================
fused_vs_unfused = (t_custom_bwd - t_fused_bwd) / t_custom_bwd * 100.0
speedup_eager_unfused = (t_torch_fwdbwd - t_custom_fwdbwd) / t_torch_fwdbwd * 100.0
speedup_eager = (t_torch_fwdbwd - t_fused_fwdbwd) / t_torch_fwdbwd * 100.0
speedup_comp  = (t_torch_fwdbwd_comp - t_fused_fwdbwd) / t_torch_fwdbwd_comp * 100.0

print(f"\n{'='*70}")
print(f"Speedup")
print(f"{'='*70}")

print(f"Fused vs unfused bwd:     {fused_vs_unfused:+.2f}%  ({t_fused_bwd:.1f} vs {t_custom_bwd:.1f} us)")
print(f"Unfused vs PyTorch eager:  {speedup_eager_unfused:+.2f}%")
print(f"Fused vs PyTorch eager:    {speedup_eager:+.2f}%")
print(f"Fused vs torch.compile:    {speedup_comp:+.2f}%")

# ============================================================
# Kernel inspection
# ============================================================
def show_kernels(fn, name):
    print(f"\n{'='*70}")
    print(f"KERNEL TRACE: {name}")
    print(f"{'='*70}")

    with prof.profile(
        activities=[prof.ProfilerActivity.CUDA],
        record_shapes=False
    ) as p:
        for _ in range(10):
            fn()

    table = p.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    )
    print(table)

show_kernels(torch_fwd_bwd, "PyTorch eager")
show_kernels(torch_fwd_bwd_compiled, "PyTorch compiled")
