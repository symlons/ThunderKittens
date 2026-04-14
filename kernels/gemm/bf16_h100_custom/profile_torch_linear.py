import torch
import torch.nn as nn

M, K, N = 8129, 1056, 4424

A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)

act = nn.GELU(approximate="tanh")

def run(A, W, bias):
    out = A @ W.t()
    out = out + bias
    out = act(out)
    return out

compiled_run = torch.compile(run, mode="reduce-overhead")

def profile(fn, name):
    for _ in range(100):
        fn(A, W, bias)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    iters = 500

    start.record()
    for _ in range(iters):
        fn(A, W, bias)
    end.record()

    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    us_per_iter = (ms / iters) * 1000.0

    flops = 2 * M * N * K
    tflops = (flops / us_per_iter) / 1e6

    print(f"{name}:")
    print(f"  time per iter: {us_per_iter:.2f} us")
    print(f"  tflops: {tflops:.2f}")
    print()

profile(run, "eager")
profile(compiled_run, "compiled")
