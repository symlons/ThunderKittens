"""
Trace exactly what CUDA kernels PyTorch launches for Linear+GELU backward.
Run: python3 trace_torch_bwd.py
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

# Warmup
for _ in range(5):
    out = F.gelu(F.linear(x, linear.weight, linear.bias), approximate="tanh")
    torch.autograd.grad(out, [x, linear.weight, linear.bias], dy, retain_graph=True)
torch.cuda.synchronize()

# ============================================================
# Method 1: torch.profiler — shows kernel names + durations
# ============================================================
print("=" * 80)
print("TORCH PROFILER TRACE (backward only)")
print("=" * 80)

# Build graph
out = F.gelu(F.linear(x, linear.weight, linear.bias), approximate="tanh")

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    grads = torch.autograd.grad(out, [x, linear.weight, linear.bias], dy, retain_graph=True)
    torch.cuda.synchronize()

# Print all CUDA kernels
print("\nCUDA kernels launched during backward:")
print(f"{'Kernel':<90} {'Duration (us)':>14} {'Grid':>20} {'Block':>15}")
print("-" * 145)

events = prof.key_averages()
for evt in sorted(prof.events(), key=lambda e: e.time_range.start):
    if evt.device_type == torch.autograd.DeviceType.CUDA:
        name = evt.name[:88]
        dur = evt.device_time_total
        print(f"  {name:<88} {dur:>12.1f} us")

# ============================================================
# Method 2: Detailed table
# ============================================================
print("\n" + "=" * 80)
print("KERNEL SUMMARY (sorted by total CUDA time)")
print("=" * 80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# ============================================================
# Method 3: Also trace forward+backward to see the difference
# ============================================================
print("\n" + "=" * 80)
print("TORCH PROFILER TRACE (forward + backward)")
print("=" * 80)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof2:
    out2 = F.gelu(F.linear(x, linear.weight, linear.bias), approximate="tanh")
    grads2 = torch.autograd.grad(out2, [x, linear.weight, linear.bias], dy)
    torch.cuda.synchronize()

print(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=25))

# ============================================================
# Method 4: Export chrome trace for visual inspection
# ============================================================
trace_path = "torch_bwd_trace.json"
prof.export_chrome_trace(trace_path)
print(f"\nChrome trace exported to: {trace_path}")
print("Open in chrome://tracing or https://ui.perfetto.dev/")
