import torch
import torch.nn.functional as F

import _gelu_bwd

M, N = 4096, 4096

torch.manual_seed(42)
preact = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
grad_output = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

# Reference: PyTorch GELU backward
preact_f = preact.float().requires_grad_(True)
y = F.gelu(preact_f, approximate="tanh")
y.backward(grad_output.float())
ref = preact_f.grad.to(torch.bfloat16)

# Custom kernel
grad_input = torch.empty_like(preact)
_gelu_bwd.gelu_backward(grad_output, preact, grad_input)

diff = (grad_input.float() - ref.float()).abs()
print(f"max diff:  {diff.max().item()}")
print(f"mean diff: {diff.mean().item()}")
print(f"match:     {torch.allclose(grad_input, ref, rtol=5e-2, atol=5e-2)}")

# Benchmark
def profile(fn, name, warmup=500, iters=1000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000.0 / iters
    bytes_moved = 3 * M * N * 2
    bw = bytes_moved / (us * 1e-6) / 1e12
    print(f"\n{name}:")
    print(f"  time: {us:.2f} us")
    print(f"  BW:   {bw:.3f} TB/s")

profile(lambda: _gelu_bwd.gelu_backward(grad_output, preact, grad_input), "custom_kernel")

# PyTorch eager baseline
def torch_gelu_bwd():
    preact_f.grad = None
    y = F.gelu(preact_f, approximate="tanh")
    y.backward(grad_output.float())

profile(torch_gelu_bwd, "torch_eager")

# torch.compile baseline
compiled_bwd = torch.compile(torch_gelu_bwd, mode="max-autotune")
profile(compiled_bwd, "torch_compile")
