"""
Isolate and benchmark the exact cuBLAS GEMMs that PyTorch backward uses.
Follows the ThunderKittens benchmarking convention exactly:
  - Uniform random inputs (bitwise identical across groups via manual_seed)
  - L2-defeating input groups: num_groups = (input < 3*L2) ? (3*L2/input)+1 : 1
  - 500 warmup iterations (no sync after)
  - 100 profiling iterations, back-to-back, no intermediate sync
  - 2 CUDA events around all 100 iters
  - 500ms sleep between benchmarks for thermal cooldown
Run: python3 bench_cublas_gemms.py
"""
import torch
import time

torch.manual_seed(42)

M, K, N = 4096, 4096, 4096
WARMUP = 500
ITERS  = 100
flops = 2.0 * M * K * N

# Query actual L2 cache size from device
l2_cache_size = torch.cuda.get_device_properties(0).L2_cache_size  # bytes
print(f"L2 cache size: {l2_cache_size / 1024 / 1024:.0f} MB")


def bench_gemm(name, make_inputs_fn, gemm_fn, input_bytes):
    """Benchmark a GEMM following TK convention."""
    num_groups = 1 if input_bytes >= l2_cache_size * 3 else (l2_cache_size * 3 // input_bytes) + 1

    # Allocate all buffer groups with bitwise-identical random data per group
    all_inputs = [make_inputs_fn(i) for i in range(num_groups)]
    torch.cuda.synchronize()

    # 500 warmup — no sync after
    for i in range(WARMUP):
        gemm_fn(*all_inputs[i % num_groups])

    # 2 events around all 100 iters, back-to-back, no intermediate sync
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(ITERS):
        gemm_fn(*all_inputs[i % num_groups])
    end.record()
    torch.cuda.synchronize()

    us = start.elapsed_time(end) * 1000.0 / ITERS
    tflops = flops / us / 1e6
    print(f"\n{name}:")
    print(f"  input_bytes: {input_bytes / 1e6:.0f} MB,  groups: {num_groups}")
    print(f"  Time:    {us:.1f} us")
    print(f"  TFLOPS:  {tflops:.0f}")

    # 500ms thermal cooldown
    torch.cuda.synchronize()
    time.sleep(0.5)

    return us, tflops


print(f"{'='*70}")
print(f"cuBLAS GEMM benchmark (TK convention): M={M}, K={K}, N={N}, bf16")
print(f"FLOPs per GEMM: {flops/1e12:.3f} TFLOP")
print(f"Warmup: {WARMUP}, Iters: {ITERS}")
print(f"{'='*70}")

bpe = 2  # bf16

# ---- dW = x^T @ dz  (A^T B): inputs x[M,K] + dz[M,N] + output dW[K,N] ----
dw_bytes = (M * K + M * N + K * N) * bpe

def make_dw_inputs(seed):
    torch.manual_seed(42 + seed * 100)
    x  = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    dz = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dW = torch.empty(K, N, device="cuda", dtype=torch.bfloat16)
    return (x, dz, dW)

def dw_gemm(x, dz, dW):
    torch.mm(x.T, dz, out=dW)

t_dw, tf_dw = bench_gemm("dW = x^T @ dz (A^T B)", make_dw_inputs, dw_gemm, dw_bytes)

# ---- dx = dz @ W^T  (A B^T): inputs dz[M,N] + W[K,N] + output dx[M,K] ----
dx_bytes = (M * N + K * N + M * K) * bpe

def make_dx_inputs(seed):
    torch.manual_seed(42 + seed * 100)
    dz = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    W  = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    dx = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    return (dz, W, dx)

def dx_gemm(dz, W, dx):
    torch.mm(dz, W.T, out=dx)

t_dx, tf_dx = bench_gemm("dx = dz @ W^T (A B^T)", make_dx_inputs, dx_gemm, dx_bytes)

# ---- Plain A @ B reference ----
ab_bytes = (M * K + K * N + M * N) * bpe

def make_ab_inputs(seed):
    torch.manual_seed(42 + seed * 100)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    return (A, B, C)

def ab_gemm(A, B, C):
    torch.mm(A, B, out=C)

t_ab, tf_ab = bench_gemm("C = A @ B (no transpose, reference)", make_ab_inputs, ab_gemm, ab_bytes)

# ---- Profiler trace to confirm kernel names ----
print(f"\n{'='*70}")
print("Profiler trace (verifying kernel names)")
print(f"{'='*70}")

x  = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
dz = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
W  = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
A  = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
B  = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
dW = torch.empty(K, N, device="cuda", dtype=torch.bfloat16)
dx = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
C  = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
) as prof:
    torch.mm(x.T, dz, out=dW)
    torch.mm(dz, W.T, out=dx)
    torch.mm(A, B, out=C)
    torch.cuda.synchronize()

print(f"\n  {'Kernel':<90} {'Duration (us)':>14}")
print("-" * 110)
for evt in sorted(prof.events(), key=lambda e: e.time_range.start):
    if evt.device_type == torch.autograd.DeviceType.CUDA:
        print(f"  {evt.name[:88]:<88} {evt.device_time_total:>12.1f} us")

print(f"\n{'='*70}")
print(f"SUMMARY (TK convention: {WARMUP} warmup, {ITERS} iters, L2-defeating)")
print(f"{'='*70}")
print(f"  dW (A^T B):  {t_dw:7.1f} us  ({tf_dw:.0f} TFLOPS)")
print(f"  dx (A B^T):  {t_dx:7.1f} us  ({tf_dx:.0f} TFLOPS)")
print(f"  ref (A B):   {t_ab:7.1f} us  ({tf_ab:.0f} TFLOPS)")
