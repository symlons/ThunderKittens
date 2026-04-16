import torch
import torch.nn as nn

torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_gemm_backends = "CUTLASS,TRITON"

import _C

M, K, N = 4096, 4096, 4096

l2_cache_size = 50 * 1024 * 1024

def make_group(seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    gen2 = torch.Generator(device="cuda")
    gen2.manual_seed(seed + 1)

    g = {}
    g["A"] = (2 * torch.rand(M, K, device="cuda", dtype=torch.float32, generator=gen) - 1).to(torch.bfloat16)
    g["B"] = (2 * torch.rand(K, N, device="cuda", dtype=torch.float32, generator=gen2) - 1).to(torch.bfloat16)
    g["C"] = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    g["bias"] = torch.zeros(1, N, device="cuda", dtype=torch.bfloat16)
    g["preact"] = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    return g

arg_size = 2 * (M * K + N * K + M * N) * 2
ideal_arg_size = l2_cache_size * 3
arg_group_count = (ideal_arg_size // arg_size) + 1
arg_group_count = max(1, int(arg_group_count))

groups = [make_group(42 + i * 100) for i in range(arg_group_count)]

act = nn.GELU(approximate="tanh").cuda()

def run_custom(A, B, C, bias, preact):
    _C.gemm_custom(A, B, C, bias, preact)
    return C

def run_torch_gelu(A, B, bias):
    return act(torch.matmul(A, B) + bias)

compiled_gelu = torch.compile(run_torch_gelu, mode="max-autotune")

def profile(fn, name, iters=100, warmup=500):
    is_custom = name == "custom"

    for i in range(warmup):
        g = groups[i % arg_group_count]
        if is_custom:
            fn(g["A"], g["B"], g["C"], g["bias"], g["preact"])
        else:
            fn(g["A"], g["B"], g["bias"])

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for i in range(iters):
        g = groups[i % arg_group_count]
        if is_custom:
            fn(g["A"], g["B"], g["C"], g["bias"], g["preact"])
        else:
            fn(g["A"], g["B"], g["bias"])

    end.record()
    torch.cuda.synchronize()

    us_per_iter = (start.elapsed_time(end) * 1000.0) / iters
    tflops = (2 * M * N * K / us_per_iter) / 1e6

    print(name)
    print("time per iter (us):", us_per_iter)
    print("tflops:", tflops)
    print()

profile(run_custom, "custom")
profile(run_torch_gelu, "torch_eager_gelu")
profile(compiled_gelu, "torch_compile_gelu")

g = groups[0]
g["C"].zero_()
g["preact"].zero_()

_C.gemm_custom(g["A"], g["B"], g["C"], g["bias"], g["preact"])
torch_out = run_torch_gelu(g["A"], g["B"], g["bias"])

diff = (g["C"] - torch_out).abs().float()

print("match:", torch.allclose(g["C"], torch_out, rtol=5e-2, atol=5e-2))
print("max diff:", diff.max().item())
print("mean diff:", diff.mean().item())

torch_preact = torch.matmul(g["A"], g["B"]) + g["bias"]
preact_diff = (g["preact"] - torch_preact).abs().float()
print("preact match:", (preact_diff > 1.0).sum().item() <= 10)
print("preact max diff:", preact_diff.max().item())
print("preact mean diff:", preact_diff.mean().item())
