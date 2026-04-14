import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import _C

M, K, N = 8129, 1056, 4424

A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
C = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
bias = torch.zeros(1, N, device="cuda", dtype=torch.bfloat16)

def run_custom(A, B, C, bias):
    _C.gemm_custom(A, B, C, bias)
    return C

act = nn.GELU(approximate="tanh")

def run_torch(A, B, bias):
    out = A @ B# + bias
    out = act(out)
    return out

def profile(fn, name, iters=500):
    for _ in range(100):
        fn(A, B, C, bias) if name == "custom" else fn(A, B, bias)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        if name == "custom":
            fn(A, B, C, bias)
        else:
            fn(A, B, bias)
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

profile(run_custom, "custom")
profile(run_torch, "torch")

# correctness check
_C.gemm_custom(A, B, C, bias)

torch_out = run_torch(A, B, bias)

diff = (C - torch_out).abs()
diff_f = diff.float()

max_diff = diff_f.max().item()
min_diff = diff_f.min().item()
mean_diff = diff_f.mean().item()

rtol = 5e-2
atol = 5e-2

matches = torch.allclose(C, torch_out, rtol=rtol, atol=atol)

print(f"match: {matches}")
print(f"max diff: {max_diff}")
print(f"min diff: {min_diff}")
print(f"mean diff: {mean_diff}")

bad = diff_f > (atol + rtol * torch_out.abs().float())
print(f"num mismatches: {bad.sum().item()}")

diff_img = diff_f.cpu()

plt.figure(figsize=(6, 6))
plt.imshow(diff_img)
plt.colorbar()
plt.title("Absolute Difference (bf16)")
plt.axis('off')
plt.savefig("diff_map.png", dpi=200, bbox_inches='tight')
plt.close()

plt.figure(figsize=(6, 6))
plt.imshow(torch.log1p(diff_img))
plt.colorbar()
plt.title("Log Diff")
plt.axis('off')
plt.savefig("diff_map_log.png", dpi=200, bbox_inches='tight')
plt.close()
