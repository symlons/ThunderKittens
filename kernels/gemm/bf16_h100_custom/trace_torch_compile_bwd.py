"""
Trace exactly what CUDA kernels torch.compile launches for Linear+GELU backward.
Run: python3 trace_torch_compile_bwd.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

M, K, N = 4096, 4096, 4096

# Setup
linear = nn.Linear(K, N, bias=True, device="cuda", dtype=torch.bfloat16)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

# ============================================================
# Compiled forward+backward
# ============================================================
def fwd_bwd(x, weight, bias, dy):
    out = F.gelu(F.linear(x, weight, bias), approximate="tanh")
    grads = torch.autograd.grad(out, [x, weight, bias], dy)
    return grads

compiled_fwd_bwd = torch.compile(fwd_bwd, mode="max-autotune")

# Warmup (triggers compilation)
print("Compiling... (this may take a minute)")
for _ in range(5):
    compiled_fwd_bwd(x, linear.weight, linear.bias, dy)
torch.cuda.synchronize()
print("Compilation done.\n")

# ============================================================
# Trace compiled forward+backward
# ============================================================
print("=" * 80)
print("TORCH.COMPILE TRACE (forward + backward)")
print("=" * 80)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof_compile_fwdbwd:
    compiled_fwd_bwd(x, linear.weight, linear.bias, dy)
    torch.cuda.synchronize()

print("\nCUDA kernels launched:")
print(f"  {'Kernel':<90} {'Duration (us)':>14}")
print("-" * 110)
for evt in sorted(prof_compile_fwdbwd.events(), key=lambda e: e.time_range.start):
    if evt.device_type == torch.autograd.DeviceType.CUDA:
        name = evt.name[:88]
        dur = evt.device_time_total
        print(f"  {name:<88} {dur:>12.1f} us")

print("\n" + prof_compile_fwdbwd.key_averages().table(sort_by="cuda_time_total", row_limit=25))

# ============================================================
# Also trace compiled backward-only via retain_graph
# ============================================================
print("\n" + "=" * 80)
print("TORCH.COMPILE TRACE (backward only, retain_graph)")
print("=" * 80)

def bwd_only(x, weight, bias, dy, out):
    grads = torch.autograd.grad(out, [x, weight, bias], dy, retain_graph=True)
    return grads

compiled_bwd = torch.compile(bwd_only, mode="max-autotune")

out = F.gelu(F.linear(x, linear.weight, linear.bias), approximate="tanh")

# Warmup
for _ in range(5):
    compiled_bwd(x, linear.weight, linear.bias, dy, out)
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof_compile_bwd:
    compiled_bwd(x, linear.weight, linear.bias, dy, out)
    torch.cuda.synchronize()

print("\nCUDA kernels launched:")
print(f"  {'Kernel':<90} {'Duration (us)':>14}")
print("-" * 110)
for evt in sorted(prof_compile_bwd.events(), key=lambda e: e.time_range.start):
    if evt.device_type == torch.autograd.DeviceType.CUDA:
        name = evt.name[:88]
        dur = evt.device_time_total
        print(f"  {name:<88} {dur:>12.1f} us")

print("\n" + prof_compile_bwd.key_averages().table(sort_by="cuda_time_total", row_limit=25))

# ============================================================
# Compare: eager vs compiled timing
# ============================================================
print("\n" + "=" * 80)
print("TIMING COMPARISON (1000 iters, batch-timed)")
print("=" * 80)

WARMUP, ITERS = 500, 1000

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

# Eager
out_eager = F.gelu(F.linear(x, linear.weight, linear.bias), approximate="tanh")
def eager_bwd():
    torch.autograd.grad(out_eager, [x, linear.weight, linear.bias], dy, retain_graph=True)
def eager_fwdbwd():
    o = F.gelu(F.linear(x, linear.weight, linear.bias), approximate="tanh")
    torch.autograd.grad(o, [x, linear.weight, linear.bias], dy)

# Compiled
def compiled_bwd_fn():
    compiled_bwd(x, linear.weight, linear.bias, dy, out_eager)
def compiled_fwdbwd_fn():
    compiled_fwd_bwd(x, linear.weight, linear.bias, dy)

t_eager_bwd = profile_batch(eager_bwd)
t_eager_fwdbwd = profile_batch(eager_fwdbwd)
t_compile_bwd = profile_batch(compiled_bwd_fn)
t_compile_fwdbwd = profile_batch(compiled_fwdbwd_fn)

print(f"\n  Eager    bwd only:   {t_eager_bwd:7.1f} us")
print(f"  Eager    fwd+bwd:    {t_eager_fwdbwd:7.1f} us")
print(f"  Compiled bwd only:   {t_compile_bwd:7.1f} us")
print(f"  Compiled fwd+bwd:    {t_compile_fwdbwd:7.1f} us")

# Export traces
prof_compile_fwdbwd.export_chrome_trace("torch_compile_fwdbwd_trace.json")
prof_compile_bwd.export_chrome_trace("torch_compile_bwd_trace.json")
print(f"\nTraces exported to torch_compile_*_trace.json")
