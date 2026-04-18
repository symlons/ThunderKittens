"""
End-to-end training step benchmark: forward + backward for Linear+GELU.
Measures the full cost including transposes needed by our custom kernels.

Custom:  fwd(x,W,b) → y,preact  |  Xt=x.T, Wt=W.T  |  bwd(dy,preact,Xt,Wt) → dz,db,dW,dx
PyTorch: fwd+bwd via autograd
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import _C
import _linear_bwd

torch.manual_seed(42)

M, K, N = 4096, 4096, 4096
WARMUP, ITERS = 500, 1000

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

# Custom outputs
y_custom     = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
preact       = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
dz           = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
dbias        = torch.empty(N, device="cuda", dtype=torch.float32)
dW           = torch.empty(K, N, device="cuda", dtype=torch.bfloat16)
dx           = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)

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
    return s.elapsed_time(e) * 1000.0 / iters  # us

# ============================================================
# Custom fwd + bwd  (transposes included in the timing)
# ============================================================
def custom_fwd_bwd():
    # Forward: y = GELU(x @ W + b), saves preact
    _C.gemm_custom(x, W, y_custom, b, preact)
    # Backward — no transposes needed!
    _linear_bwd.gelu_bwd_bias(dy, preact, dz, dbias)
    _linear_bwd.dw_gemm(x, dz, dW)
    _linear_bwd.dx_gemm(dz, W, dx)

# Custom fwd only
def custom_fwd():
    _C.gemm_custom(x, W, y_custom, b, preact)

# Custom bwd only
def custom_bwd():
    _linear_bwd.gelu_bwd_bias(dy, preact, dz, dbias)
    _linear_bwd.dw_gemm(x, dz, dW)
    _linear_bwd.dx_gemm(dz, W, dx)

# ============================================================
# PyTorch fwd + bwd
# ============================================================
linear_pt = nn.Linear(K, N, bias=True, device="cuda", dtype=torch.bfloat16)
x_pt = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dy_pt = dy.clone()

def torch_fwd_bwd():
    out = F.gelu(F.linear(x_pt, linear_pt.weight, linear_pt.bias), approximate="tanh")
    torch.autograd.grad(out, [x_pt, linear_pt.weight, linear_pt.bias], dy_pt)

def torch_fwd_only():
    F.gelu(F.linear(x_pt, linear_pt.weight, linear_pt.bias), approximate="tanh")

out_pt = F.gelu(F.linear(x_pt, linear_pt.weight, linear_pt.bias), approximate="tanh")
def torch_bwd_only():
    torch.autograd.grad(out_pt, [x_pt, linear_pt.weight, linear_pt.bias], dy_pt, retain_graph=True)

# ============================================================
# Profile everything
# ============================================================
t_custom_fwdbwd      = profile_batch(custom_fwd_bwd)
t_custom_fwd         = profile_batch(custom_fwd)
t_custom_bwd         = profile_batch(custom_bwd)
t_torch_fwdbwd       = profile_batch(torch_fwd_bwd)
t_torch_fwd          = profile_batch(torch_fwd_only)
t_torch_bwd          = profile_batch(torch_bwd_only)

# ============================================================
# Results
# ============================================================
flops_gemm = 2.0 * M * K * N

print(f"\n{'='*70}")
print(f"Performance ({ITERS} iters, batch-timed)")
print(f"{'='*70}")

print(f"\nCustom (TK kernels, zero-transpose):")
print(f"  Forward:              {t_custom_fwd:7.1f} us  ({flops_gemm/t_custom_fwd/1e6:.0f} TFLOPS)")
print(f"  Backward:             {t_custom_bwd:7.1f} us")
print(f"  Fwd+Bwd (measured):   {t_custom_fwdbwd:7.1f} us")

print(f"\nPyTorch:")
print(f"  Forward:              {t_torch_fwd:7.1f} us  ({flops_gemm/t_torch_fwd/1e6:.0f} TFLOPS)")
print(f"  Backward:             {t_torch_bwd:7.1f} us")
print(f"  Fwd+Bwd (measured):   {t_torch_fwdbwd:7.1f} us")

print(f"\n{'='*70}")
print(f"TRAINING STEP COMPARISON (fwd + bwd)")
print(f"{'='*70}")
speedup = t_torch_fwdbwd / t_custom_fwdbwd
if speedup > 1:
    print(f"  Custom:   {t_custom_fwdbwd:7.1f} us")
    print(f"  PyTorch:  {t_torch_fwdbwd:7.1f} us")
    print(f"  >>> CUSTOM {speedup:.2f}x FASTER <<<")
else:
    print(f"  Custom:   {t_custom_fwdbwd:7.1f} us")
    print(f"  PyTorch:  {t_torch_fwdbwd:7.1f} us")
    print(f"  >>> PyTorch {1/speedup:.2f}x faster <<<")
print(f"{'='*70}")
