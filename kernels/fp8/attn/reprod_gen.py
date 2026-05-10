import torch
import numpy as np

SEED = 2024
def splitmix_bf16(count: int, seed: int, min_val: float, max_val: float) -> torch.Tensor:
    """Splitmix64 hash -> uniform bf16, bitwise identical to common.cuh fill_kernel."""
    idx = np.arange(count, dtype=np.uint64)
    x = np.uint64(seed) + idx
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    u = (x >> np.uint64(40)).astype(np.float32) * np.float32(1.0 / 16777216.0)
    vals = u * np.float32(max_val - min_val) + np.float32(min_val)
    return torch.from_numpy(vals).to(torch.bfloat16).cuda()

def create_inputs(B, S, H, Dqk, Dv, seed):
    """Create a single set of (Q, K, V, O, LSE) tensors."""
    count_qk = B * S * H * Dqk
    count_v = B * S * H * Dv
    Q = splitmix_bf16(count_qk, seed, -1.0, 1.0).view(B, S, H, Dqk)
    K = splitmix_bf16(count_qk, seed + 1, -1.0, 1.0).view(B, S, H, Dqk)
    V = splitmix_bf16(count_v, seed + 2, -1.0, 1.0).view(B, S, H, Dv)
    O = torch.zeros(B, S, H, Dv, dtype=torch.bfloat16, device="cuda")
    LSE = torch.zeros(B, H, 1, S, dtype=torch.float32, device="cuda")
    return Q, K, V, O, LSE

