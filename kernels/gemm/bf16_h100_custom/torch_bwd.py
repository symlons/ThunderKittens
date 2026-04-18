import torch
import torch.nn as nn
import time

device = "cuda"

M, K, N = 4096, 4096, 4096
iters = 20

def make_model():
    linear = nn.Linear(K, N, bias=True, device=device, dtype=torch.bfloat16)
    return linear

def run_eager():
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    linear = make_model()

    def step():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = torch.nn.functional.gelu(linear(x))
            loss = y.sum()
        loss.backward()

    for _ in range(10):
        x.grad = None
        linear.zero_grad(set_to_none=True)
        step()

    torch.cuda.synchronize()

    start = time.time()

    for _ in range(iters):
        x.grad = None
        linear.zero_grad(set_to_none=True)
        step()

    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / iters


def run_compiled():
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    linear = make_model()

    def step():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = torch.nn.functional.gelu(linear(x))
            loss = y.sum()
        loss.backward()

    compiled_step = torch.compile(step, mode="max-autotune")

    for _ in range(10):
        x.grad = None
        linear.zero_grad(set_to_none=True)
        compiled_step()

    torch.cuda.synchronize()

    start = time.time()

    for _ in range(iters):
        x.grad = None
        linear.zero_grad(set_to_none=True)
        compiled_step()

    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / iters


flops = 4 * M * N * K


eager_time = run_eager()
compiled_time = run_compiled()

eager_tflops = flops / eager_time / 1e12
compiled_tflops = flops / compiled_time / 1e12

print("=== EAGER BF16 + AUTOCAST ===")
print(f"time per iter (ms): {eager_time * 1000:.3f}")
print(f"TFLOPs: {eager_tflops:.2f}")

print("\n=== TORCH COMPILE + AUTOCAST ===")
print(f"time per iter (ms): {compiled_time * 1000:.3f}")
print(f"TFLOPs: {compiled_tflops:.2f}")

print("\n=== SPEEDUP ===")
print(f"x{eager_time / compiled_time:.2f}")
