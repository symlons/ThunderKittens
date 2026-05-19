import argparse
from pathlib import Path

import torch

from .cases import backward_ref_cases
from .kernel_api import fp8_forward, require_extension
from .metrics import fmt_grad, tensor_metrics
from .recipe import natural_lse_from_kernel_l, prepare_forward_inputs
from .references import BwdRecipe, fp8_attention_backward, reference_backward


def kernel_forward_for_backward(Q, K, V, causal, *, smooth_k=True, smooth_v=True):
    if causal:
        raise NotImplementedError("FP8 kernel is non-causal / bidirectional only")
    fwd = prepare_forward_inputs(Q, K, V, smooth_k=smooth_k, smooth_v=smooth_v)
    O, L_raw = fp8_forward(fwd)
    return fwd, O.to(torch.float32), natural_lse_from_kernel_l(L_raw, Q.shape[-1])


def print_grads(label, got, ref):
    print(" ", fmt_grad(label, *(tensor_metrics(a, b) for a, b in zip(got, ref))))


def run_one(B, H, N, D, causal, seed, *, ablate=True):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    dO = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    print(f"\n[B={B} H={H} N={N} D={D} causal={causal} seed={seed}]")
    ref = reference_backward(Q, K, V, dO, causal=causal)
    fwd, _, L = kernel_forward_for_backward(Q, K, V, causal)

    recipe = fp8_attention_backward(fwd.Q_eff, fwd.K_eff_centered, fwd.V_centered, L, dO, causal=causal)
    print_grads("recipe (e4m3, per_row, SR)", recipe, ref)
    if not ablate:
        return

    sweeps = [
        ("no grad-quant", BwdRecipe(quant_grads=False, quant_p=False, quant_v=False)),
        ("grad e4m3 RTNE per_row", BwdRecipe(grad_mode="fp8_e4m3", grad_granularity="per_row", stochastic=False)),
        ("grad e4m3 SR  per_row", BwdRecipe(grad_mode="fp8_e4m3", grad_granularity="per_row", stochastic=True)),
        ("grad e4m3 SR  per_tensor", BwdRecipe(grad_mode="fp8_e4m3", grad_granularity="per_tensor", stochastic=True)),
        ("grad e5m2 SR  per_row", BwdRecipe(grad_mode="fp8_e5m2", grad_granularity="per_row", stochastic=True)),
        ("grad e5m2 SR  per_tensor", BwdRecipe(grad_mode="fp8_e5m2", grad_granularity="per_tensor", stochastic=True)),
        ("e4m3 SR per_row, V fp32", BwdRecipe(quant_v=False)),
        ("e4m3 SR per_row, P fp32", BwdRecipe(quant_p=False)),
    ]
    for label, bwd_recipe in sweeps:
        got = fp8_attention_backward(
            fwd.Q_eff,
            fwd.K_eff_centered,
            fwd.V_centered,
            L,
            dO,
            causal=causal,
            recipe=bwd_recipe,
        )
        print_grads(label, got, ref)


def run_smoothing_ablation(B, H, N, D, seed):
    print(f"\n[bwd smoothing ablation B={B} H={H} N={N} D={D} seed={seed}]")
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    dO = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = K + torch.randn(1, 1, 1, D, device="cuda") * 4.0
    V = V + torch.randn(1, 1, 1, D, device="cuda") * 4.0
    ref = reference_backward(Q, K, V, dO)
    for sk_on, sv_on in [(False, False), (True, False), (False, True), (True, True)]:
        fwd, _, L = kernel_forward_for_backward(Q, K, V, False, smooth_k=sk_on, smooth_v=sv_on)
        got = fp8_attention_backward(fwd.Q_eff, fwd.K_eff_centered, fwd.V_centered, L, dO)
        print_grads(f"K={int(sk_on)} V={int(sv_on)}", got, ref)


def run_cogvideox():
    print("\n[CogVideoX-2b real Q,K,V - backward]")
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
    dO = torch.randn_like(Q)
    ref = reference_backward(Q, K, V, dO)
    for sk_on, sv_on in [(False, False), (True, True)]:
        fwd, _, L = kernel_forward_for_backward(Q, K, V, False, smooth_k=sk_on, smooth_v=sv_on)
        got = fp8_attention_backward(fwd.Q_eff, fwd.K_eff_centered, fwd.V_centered, L, dO)
        print_grads(f"smooth K={int(sk_on)} V={int(sv_on)}", got, ref)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cogvideox", action="store_true")
    args = parser.parse_args()

    require_extension("fp8_mha_forward")
    for cfg in backward_ref_cases(args.quick):
        run_one(*cfg)
    run_smoothing_ablation(1, 8, 1536, 128, seed=0)
    if args.cogvideox:
        run_cogvideox()


if __name__ == "__main__":
    main()
