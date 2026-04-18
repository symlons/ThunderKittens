import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
import _linear_bwd

M, K, N = 4096, 4096, 4096

print(f"{'='*60}")
print(f"Linear backward benchmark: M={M}, K={K}, N={N}, bf16")
print(f"Forward:  y = GELU(x @ W + b)")
print(f"Backward: dz, db, dW, dx from dy")
print(f"{'='*60}")

# ========== Setup ==========
# Common data (same for both)
x  = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
W  = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
b  = torch.randn(N,    device="cuda", dtype=torch.bfloat16)
preact = x @ W + b  # saved from forward
dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

# ========== Correctness ==========
# PyTorch reference (float32 for accuracy)
preact_f = preact.float().requires_grad_(True)
y_ref = F.gelu(preact_f, approximate="tanh")
y_ref.backward(dy.float())
dz_ref = preact_f.grad.bfloat16()
dW_ref = (x.float().T @ preact_f.grad).bfloat16()
dx_ref = (preact_f.grad @ W.float().T).bfloat16()
db_ref = preact_f.grad.sum(0).float()

# Custom kernels
dz    = torch.empty_like(preact)
dbias = torch.empty(N, device="cuda", dtype=torch.float32)
dW    = torch.empty(K, N, device="cuda", dtype=torch.bfloat16)
dx    = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)

_linear_bwd.gelu_bwd_bias(dy, preact, dz, dbias)
_linear_bwd.dw_gemm(x, dz, dW)
_linear_bwd.dx_gemm(dz, W, dx)
torch.cuda.synchronize()

def cmp(name, a, b, atol=1.0):
    d = (a.float() - b.float()).abs()
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=5e-2)
    print(f"  {name}: max_diff={d.max().item():.4f} {'PASS' if ok else 'FAIL'}")
    return ok

print("\nCorrectness (vs float32 reference):")
ok = cmp("dz", dz, dz_ref) & cmp("db", dbias, db_ref) & cmp("dW", dW, dW_ref, 8.0) & cmp("dx", dx, dx_ref, 8.0)
print(f"  {'ALL PASS' if ok else 'FAIL'}")

# ========== Profiling ==========
# Use identical methodology: batch timing with cuda events, many iterations

WARMUP = 500
ITERS  = 1000

def profile_batch(fn, warmup=WARMUP, iters=ITERS):
    """Time a function using batch cuda events — same method for both."""
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
    return s.elapsed_time(e) * 1000.0 / iters  # microseconds

# ----- Custom backward -----
# All 3 kernels launched from Python, same as how a user would call them
def custom_backward():
    _linear_bwd.gelu_bwd_bias(dy, preact, dz, dbias)  # dz + dbias
    _linear_bwd.dw_gemm(x, dz, dW)                     # dW = x^T @ dz
    _linear_bwd.dx_gemm(dz, W, dx)                     # dx = dz @ W^T

# ----- PyTorch backward -----
# Use torch.autograd.grad with retain_graph=True to time ONLY backward.
# No forward subtraction needed — forward runs once, backward is timed directly.
linear_pt = nn.Linear(K, N, bias=True, device="cuda", dtype=torch.bfloat16)
x_pt = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dy_pt = dy.clone()

# Build graph once; retain_graph lets us re-run backward repeatedly
out_pt = F.gelu(F.linear(x_pt, linear_pt.weight, linear_pt.bias), approximate="tanh")

# Verify PyTorch computes all 3 gradients
grads = torch.autograd.grad(out_pt, [x_pt, linear_pt.weight, linear_pt.bias], dy_pt,
                            retain_graph=True)
assert grads[0] is not None, "PyTorch didn't compute dx"
assert grads[1] is not None, "PyTorch didn't compute dW"
assert grads[2] is not None, "PyTorch didn't compute db"

def torch_bwd_only():
    torch.autograd.grad(out_pt, [x_pt, linear_pt.weight, linear_pt.bias], dy_pt,
                        retain_graph=True)

# Also time fwd+bwd for reference
def torch_fwd_bwd():
    out = F.gelu(F.linear(x_pt, linear_pt.weight, linear_pt.bias), approximate="tanh")
    torch.autograd.grad(out, [x_pt, linear_pt.weight, linear_pt.bias], dy_pt)

# ----- Profile everything -----
t_custom      = profile_batch(custom_backward)
t_gelu        = profile_batch(lambda: _linear_bwd.gelu_bwd_bias(dy, preact, dz, dbias))
t_dw          = profile_batch(lambda: _linear_bwd.dw_gemm(x, dz, dW))
t_dx          = profile_batch(lambda: _linear_bwd.dx_gemm(dz, W, dx))
t_torch_bwd   = profile_batch(torch_bwd_only)
t_torch_fwdbwd = profile_batch(torch_fwd_bwd)

# ----- Bare cuBLAS GEMMs (no autograd overhead) -----
# Replicate exactly what PyTorch backward launches:
#   1) gelu_bwd elementwise: dz = dy * gelu'(preact)
#   2) dW = x^T @ dz  (torch.mm(x.T, dz))
#   3) dx = dz @ W^T  (torch.mm(dz, W.T))
#   4) db = dz.sum(0)
dz_torch = torch.empty_like(preact)
dW_torch = torch.empty(K, N, device="cuda", dtype=torch.bfloat16)
dx_torch = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
db_torch = torch.empty(N, device="cuda", dtype=torch.bfloat16)

def torch_gelu_bwd():
    torch.ops.aten.gelu_backward(dy, preact, approximate="tanh")

def torch_dw_gemm():
    torch.mm(x.T, dz_torch, out=dW_torch)

def torch_dx_gemm():
    torch.mm(dz_torch, W.T, out=dx_torch)

def torch_bias_reduce():
    torch.sum(dz_torch, dim=0, out=db_torch)

def torch_bare_backward():
    torch.ops.aten.gelu_backward(dy, preact, approximate="tanh")
    torch.mm(x.T, dz_torch, out=dW_torch)
    torch.mm(dz_torch, W.T, out=dx_torch)
    torch.sum(dz_torch, dim=0, out=db_torch)

t_torch_gelu    = profile_batch(torch_gelu_bwd)
t_torch_dw      = profile_batch(torch_dw_gemm)
t_torch_dx      = profile_batch(torch_dx_gemm)
t_torch_bias    = profile_batch(torch_bias_reduce)
t_torch_bare    = profile_batch(torch_bare_backward)

# ========== Results ==========
flops = 2 * M * N * K
print(f"\n{'='*60}")
print(f"Performance ({ITERS} iters, batch-timed)")
print(f"{'='*60}")

print(f"\nCustom backward (3 kernel launches):")
print(f"  gelu'+bias:        {t_gelu:7.1f} us")
print(f"  dW = x^T @ dz:    {t_dw:7.1f} us  ({flops/t_dw/1e6:.0f} TFLOPS)")
print(f"  dx = dz @ Wt:     {t_dx:7.1f} us  ({flops/t_dx/1e6:.0f} TFLOPS)")
print(f"  total (measured):  {t_custom:7.1f} us")

print(f"\nBare cuBLAS (same ops, no autograd):")
print(f"  gelu_backward:     {t_torch_gelu:7.1f} us")
print(f"  dW = x^T @ dz:    {t_torch_dw:7.1f} us  ({flops/t_torch_dw/1e6:.0f} TFLOPS)")
print(f"  dx = dz @ W^T:    {t_torch_dx:7.1f} us  ({flops/t_torch_dx/1e6:.0f} TFLOPS)")
print(f"  bias reduce:       {t_torch_bias:7.1f} us")
print(f"  total (measured):  {t_torch_bare:7.1f} us")

print(f"\nPyTorch autograd (nn.Linear + GELU):")
print(f"  bwd only:          {t_torch_bwd:7.1f} us")
print(f"  fwd+bwd:           {t_torch_fwdbwd:7.1f} us")

print(f"\n{'='*60}")
print(f"COMPARISON")
print(f"{'='*60}")
# vs bare cuBLAS
r_bare = t_torch_bare / t_custom
if r_bare > 1:
    print(f"  vs bare cuBLAS:    Custom {r_bare:.2f}x FASTER  ({t_custom:.1f} vs {t_torch_bare:.1f} us)")
else:
    print(f"  vs bare cuBLAS:    cuBLAS {1/r_bare:.2f}x faster  ({t_custom:.1f} vs {t_torch_bare:.1f} us)")
# vs autograd
r_auto = t_torch_bwd / t_custom
if r_auto > 1:
    print(f"  vs autograd bwd:   Custom {r_auto:.2f}x FASTER  ({t_custom:.1f} vs {t_torch_bwd:.1f} us)")
else:
    print(f"  vs autograd bwd:   PyTorch {1/r_auto:.2f}x faster  ({t_custom:.1f} vs {t_torch_bwd:.1f} us)")

print(f"\n  Kernel-by-kernel (custom vs bare cuBLAS):")
print(f"    gelu+bias:  {t_gelu:7.1f} vs {t_torch_gelu + t_torch_bias:7.1f} us  (torch gelu {t_torch_gelu:.1f} + reduce {t_torch_bias:.1f})")
print(f"    dW GEMM:    {t_dw:7.1f} vs {t_torch_dw:7.1f} us")
print(f"    dx GEMM:    {t_dx:7.1f} vs {t_torch_dx:7.1f} us")
print(f"{'='*60}")
