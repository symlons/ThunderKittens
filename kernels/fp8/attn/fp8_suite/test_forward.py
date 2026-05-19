import argparse
import sys
from pathlib import Path

import torch

from .cases import forward_cases
from .kernel_api import fp8_forward, require_extension
from .metrics import fmt_forward, tensor_metrics
from .recipe import prepare_forward_inputs
from .references import fp8_quant_reference, reference_attention


def kernel_attention(Q, K, V, causal, *, smooth_k=True, smooth_v=True):
    if causal:
        raise NotImplementedError("FP8 kernel is non-causal / bidirectional only")
    fwd = prepare_forward_inputs(Q, K, V, smooth_k=smooth_k, smooth_v=smooth_v)
    O, _ = fp8_forward(fwd)
    return O.to(torch.float32), fwd


def simulation_attention(Q, K, V, causal, granularity="per_token"):
    if causal:
        return None
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from smooth_core import quantized_attention

    B, H, _, _ = Q.shape
    out = torch.empty_like(Q)
    for b in range(B):
        for h in range(H):
            out[b, h] = quantized_attention(
                Q[b, h],
                K[b, h],
                V[b, h],
                qk_mode="fp8_e4m3",
                smooth_q=False,
                smooth_k=True,
                smooth_v=True,
                granularity=granularity,
                pv_mode="fp8_e4m3",
            )
    return out.to(torch.float32)


def print_comparison(label, out, ref):
    print(f"  {label:<12} {fmt_forward(tensor_metrics(out, ref))}")


def run_one(B, H, N, D, causal, seed, *, run_sim=False):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    O_ref_fp32 = reference_attention(Q, K, V, causal=causal)
    O_ker, fwd = kernel_attention(Q, K, V, causal)
    O_ref_quant = fp8_quant_reference(fwd.Qq, fwd.Kq, fwd.sq, fwd.sk, fwd.K_mean, V, causal)

    print(f"\n[B={B} H={H} N={N} D={D} causal={causal} seed={seed}]")
    print_comparison("vs ref-fp32", O_ker, O_ref_fp32)
    print_comparison("vs ref-quant", O_ker, O_ref_quant)
    if run_sim:
        O_sim = simulation_attention(Q, K, V, causal)
        if O_sim is not None:
            print_comparison("vs ref-sim", O_ker, O_sim)
            print_comparison("sim vs fp32", O_sim, O_ref_fp32)


def run_smoothing_ablation(B, H, N, D, seed):
    print(f"\n[smoothing ablation: B={B} H={H} N={N} D={D} seed={seed}, non-causal]")
    torch.manual_seed(seed)
    Q0 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K0 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V0 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    bias_K = torch.randn(1, 1, 1, D, device="cuda") * 4.0
    bias_V = torch.randn(1, 1, 1, D, device="cuda") * 4.0

    families = [
        ("zero-mean   K,V", Q0, K0, V0),
        ("biased K,V (+/-4 sigma)", Q0, K0 + bias_K, V0 + bias_V),
    ]
    for tag, Q, K, V in families:
        print(f"  -- {tag} -- (K|max|={K.abs().max().item():.2f}, V|max|={V.abs().max().item():.2f})")
        O_ref = reference_attention(Q, K, V, causal=False)
        for sk_on, sv_on in [(False, False), (True, False), (False, True), (True, True)]:
            O_ker, _ = kernel_attention(Q, K, V, causal=False, smooth_k=sk_on, smooth_v=sv_on)
            label = f"smooth K={int(sk_on)} V={int(sv_on)}"
            print(f"     {label:<22} {fmt_forward(tensor_metrics(O_ker, O_ref))}")


def run_cogvideox():
    print("\n[CogVideoX-2b real Q,K,V]")
    bundle = torch.load(
        Path(__file__).parent.parent / "captures" / "cogvideox.pt",
        map_location="cuda",
        weights_only=False,
    )
    Q_full = bundle["Q"]
    K_full = bundle["K"]
    V_full = bundle["V"]
    if isinstance(Q_full, list):
        Q_full, K_full, V_full = Q_full[0], K_full[0], V_full[0]
    Q = Q_full[:1, :8, :1536, :].to(torch.float32).contiguous()
    K = K_full[:1, :8, :1536, :].to(torch.float32).contiguous()
    V = V_full[:1, :8, :1536, :].to(torch.float32).contiguous()
    O_ker, _ = kernel_attention(Q, K, V, causal=False)
    print(f"  shape {tuple(Q.shape)}  V|max|={V.abs().max().item():.2f}  K|max|={K.abs().max().item():.2f}")
    print_comparison("vs ref-fp32", O_ker, reference_attention(Q, K, V, causal=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--cogvideox", action="store_true")
    args = parser.parse_args()

    require_extension("fp8_mha_forward")
    for cfg in forward_cases(args.quick):
        run_one(*cfg, run_sim=not args.no_sim)
    run_smoothing_ablation(1, 8, 1536, 128, seed=0)
    if args.cogvideox:
        run_cogvideox()


if __name__ == "__main__":
    main()
