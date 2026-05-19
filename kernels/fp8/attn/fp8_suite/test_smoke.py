import argparse

import torch

from .kernel_api import fp8_forward, require_extension
from .metrics import fmt_forward, tensor_metrics
from .recipe import prepare_forward_inputs
from .references import fp8_quant_reference, reference_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d64", action="store_true")
    args = parser.parse_args()

    require_extension("fp8_mha_forward")
    B, H, N, D = 1, 8, 1536, 64 if args.d64 else 128
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    fwd = prepare_forward_inputs(Q, K, V)
    O, _ = fp8_forward(fwd)
    O = O.to(torch.float32)
    qref = fp8_quant_reference(fwd.Qq, fwd.Kq, fwd.sq, fwd.sk, fwd.K_mean, V, causal=False)
    fp32 = reference_attention(Q, K, V, causal=False)

    print(f"B={B} H={H} N={N} D={D} (non-causal)")
    print("kernel output stats:")
    print(f"  O kern : mean={O.mean().item():.4f}  max={O.abs().max().item():.4f}  has_nan={torch.isnan(O).any().item()}")
    print("\nerror vs references:")
    print(f"  {'vs fp8-quant reference':<30} {fmt_forward(tensor_metrics(O, qref))}")
    print(f"  {'vs fp32 reference':<30} {fmt_forward(tensor_metrics(O, fp32))}")


if __name__ == "__main__":
    main()
