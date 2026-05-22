import argparse

import torch

from .cases import backward_kernel_cases
from .kernel_api import (
    cuda_quantize_per_channel_out,
    cuda_quantize_per_token_int8,
    cuda_quantize_per_token_out,
    fp8_backward,
    fp8_forward,
    require_extension,
)
from .metrics import check_grad_metrics, fmt_grad, tensor_metrics
from .profiling import benchmark_ms, print_profile_line, recommended_group_count, uniform_tensor
from .quant import quantize_per_channel_fp8, quantize_per_row_fp8, quantize_per_row_int8
from .recipe import prepare_backward_inputs, prepare_forward_inputs
from .references import reference_backward, sdpa_backward


def check_quant_kernel(label, xq, scale, xq_ref, scale_ref, *, granularity):
    scale_m = tensor_metrics(scale, scale_ref)
    if scale_m["rel_L1"] > 1e-6 or scale_m["max"] > 1e-6:
        raise AssertionError(f"{label} scale mismatch: max={scale_m['max']:.3e}, relL1={scale_m['rel_L1']:.3e}")
    if granularity == "token":
        got = xq.to(torch.float32) * scale.unsqueeze(-1)
        ref = xq_ref.to(torch.float32) * scale_ref.unsqueeze(-1)
    else:
        got = xq.to(torch.float32) * scale.unsqueeze(-2)
        ref = xq_ref.to(torch.float32) * scale_ref.unsqueeze(-2)
    m = tensor_metrics(got, ref)
    if m["qsnr_dB"] < 50.0 or m["rel_L1"] > 1e-4 or m["cos"] < 0.99999:
        raise AssertionError(
            f"{label} dequant mismatch: QSNR={m['qsnr_dB']:.2f}, relL1={m['rel_L1']:.3e}, cos={m['cos']:.6f}"
        )


def build_case(Q, K, V, dO, *, sr_dO=True, sdp_descale_mode="estimate"):
    fwd = prepare_forward_inputs(Q, K, V, use_cuda_quant=True)
    bwd_seed = prepare_backward_inputs(fwd, torch.empty_like(Q, dtype=torch.bfloat16), torch.empty((*Q.shape[:-1], 1), device=Q.device), dO, sr_dO=sr_dO, sdp_descale_mode=sdp_descale_mode)
    fwd.Vbf = (bwd_seed.Vq_ch.to(torch.float32) * bwd_seed.sv_ch.unsqueeze(-2)).to(torch.bfloat16).contiguous()
    O, L = fp8_forward(fwd)
    return prepare_backward_inputs(fwd, O, L, dO, sr_dO=sr_dO, sdp_descale_mode=sdp_descale_mode), O, L


def run_kernel(Q, K, V, dO, *, sr_dO=True, fp8_dS_mode=0):
    prepared, O, L = build_case(Q, K, V, dO, sr_dO=sr_dO)
    dQ, dK, dV = fp8_backward(prepared, fp8_dS_mode=fp8_dS_mode)
    return prepared, O.to(torch.float32), L, (dQ.to(torch.float32), dK.to(torch.float32), dV.to(torch.float32))


def print_grads(label, got, ref):
    print(" ", fmt_grad(label, *(tensor_metrics(a, b) for a, b in zip(got, ref))))


def make_profile_groups(shape, *, seed, group_count, sdp_descale_mode="estimate"):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    groups = []
    for _ in range(group_count):
        Q = uniform_tensor(shape, generator=generator)
        K = uniform_tensor(shape, generator=generator)
        V = uniform_tensor(shape, generator=generator)
        dO = uniform_tensor(shape, generator=generator)
        K_s = K - K.mean(dim=-2, keepdim=True)
        V_s = V - V.mean(dim=-2, keepdim=True)
        prepared, O, _ = build_case(Q, K, V, dO, sr_dO=True, sdp_descale_mode=sdp_descale_mode)
        groups.append(
            {
                "Q": Q,
                "K_s": K_s,
                "V_s": V_s,
                "prepared": prepared,
                "q_out": torch.empty_like(Q, dtype=torch.float8_e4m3fn),
                "q_scale": torch.empty(shape[:3], device=Q.device, dtype=torch.float32),
                "k_out": torch.empty_like(K_s, dtype=torch.float8_e4m3fn),
                "k_scale": torch.empty(shape[:3], device=Q.device, dtype=torch.float32),
                "v_out": torch.empty_like(V_s, dtype=torch.float8_e4m3fn),
                "v_scale": torch.empty((shape[0], shape[1], shape[3]), device=Q.device, dtype=torch.float32),
            }
        )
    torch.cuda.synchronize()
    return groups


def profile_kernels(Q=None, K=None, V=None, dO=None, *, shape=None, seed=0, bench_iters=100, warmup=500, group_count=None, cooldown_s=0.2, profile_fwd=True, profile_bwd=True, sdp_descale_mode="estimate"):
    if shape is None:
        shape = tuple(Q.shape)
    B, H, N, D = shape
    bytes_per_tensor = B * H * N * D * 4
    bytes_per_group = 4 * bytes_per_tensor
    groups_n = group_count or recommended_group_count(bytes_per_group)
    print(
        f"  benchmark protocol: uniform[-1,1], groups={groups_n}, "
        f"warmup={warmup}, iters={bench_iters}, cooldown={cooldown_s:.2f}s"
    )
    groups = make_profile_groups(shape, seed=seed, group_count=groups_n, sdp_descale_mode=sdp_descale_mode)

    token_bytes = B * H * N * D * 5 + B * H * N * 4
    channel_bytes = B * H * N * D * 5 + B * H * D * 4
    print_profile_line(
        "quant Q token",
        q_ms := benchmark_ms(
            lambda i: cuda_quantize_per_token_out(groups[i]["Q"], groups[i]["q_out"], groups[i]["q_scale"]),
            groups_n,
            warmup=warmup,
            iters=bench_iters,
            cooldown_s=cooldown_s,
        ),
        bytes_moved=token_bytes,
    )
    print_profile_line(
        "quant K token",
        k_ms := benchmark_ms(
            lambda i: cuda_quantize_per_token_out(groups[i]["K_s"], groups[i]["k_out"], groups[i]["k_scale"]),
            groups_n,
            warmup=warmup,
            iters=bench_iters,
            cooldown_s=cooldown_s,
        ),
        bytes_moved=token_bytes,
    )
    print_profile_line(
        "quant V channel",
        v_ms := benchmark_ms(
            lambda i: cuda_quantize_per_channel_out(groups[i]["V_s"], groups[i]["v_out"], groups[i]["v_scale"]),
            groups_n,
            warmup=warmup,
            iters=bench_iters,
            cooldown_s=cooldown_s,
        ),
        bytes_moved=channel_bytes,
    )
    result = {
        "groups": groups_n,
        "quant_q_ms": q_ms,
        "quant_k_ms": k_ms,
        "quant_v_ms": v_ms,
    }
    if profile_fwd:
        print_profile_line(
            "fp8 attention fwd",
            fwd_ms := benchmark_ms(lambda i: fp8_forward(groups[i]["prepared"].fwd), groups_n, warmup=warmup, iters=bench_iters, cooldown_s=cooldown_s),
            flops_shape=shape,
            kind="fwd",
        )
        result["fwd_ms"] = fwd_ms
    if profile_bwd:
        print_profile_line(
            "fp8 attention bwd",
            bwd_ms := benchmark_ms(lambda i: fp8_backward(groups[i]["prepared"], fp8_dS_mode=2), groups_n, warmup=warmup, iters=bench_iters, cooldown_s=cooldown_s),
            flops_shape=shape,
            kind="bwd",
        )
        result["bwd_ms"] = bwd_ms
    return result


def run_one(B, H, N, D, seed, *, bench_iters, bench_warmup=500, bench_groups=None, bench_cooldown=0.2):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    Q = uniform_tensor((B, H, N, D), generator=generator)
    K = uniform_tensor((B, H, N, D), generator=generator)
    V = uniform_tensor((B, H, N, D), generator=generator)
    dO = uniform_tensor((B, H, N, D), generator=generator)

    print(f"\n[B={B} H={H} N={N} D={D} seed={seed}]")
    O_sdpa, dQ_ref, dK_ref, dV_ref = sdpa_backward(Q, K, V, dO, causal=False)
    alg = reference_backward(Q, K, V, dO, causal=False)
    alg_m = tuple(tensor_metrics(a, b) for a, b in zip(alg, (dQ_ref, dK_ref, dV_ref)))
    check_grad_metrics("manual fp32 reference vs torch SDPA", *alg_m, min_qsnr=55.0, max_rel_l1=1e-3, min_cos=0.99999)
    print_grads("manual ref vs torch SDPA", alg, (dQ_ref, dK_ref, dV_ref))

    prepared, O, _, recipe = run_kernel(Q, K, V, dO, sr_dO=True, fp8_dS_mode=2)
    Qq_ref, sq_ref = quantize_per_row_fp8(Q)
    Kq_ref, sk_ref = quantize_per_row_fp8(K - K.mean(dim=-2, keepdim=True))
    Vq_ref, sv_ref = quantize_per_channel_fp8(V - V.mean(dim=-2, keepdim=True))
    check_quant_kernel("Q token quant CUDA", prepared.Qq, prepared.sq, Qq_ref, sq_ref, granularity="token")
    check_quant_kernel("K token quant CUDA", prepared.Kq, prepared.sk, Kq_ref, sk_ref, granularity="token")
    check_quant_kernel("V channel quant CUDA", prepared.Vq_ch, prepared.sv_ch, Vq_ref, sv_ref, granularity="channel")
    Qq_i8, sq_i8 = cuda_quantize_per_token_int8(Q)
    Qq_i8_ref, sq_i8_ref = quantize_per_row_int8(Q)
    check_quant_kernel("Q token INT8 quant CUDA", Qq_i8, sq_i8, Qq_i8_ref, sq_i8_ref, granularity="token")

    _, _, _, rtne = run_kernel(Q, K, V, dO, sr_dO=True, fp8_dS_mode=1)
    _, _, _, sr_dO_only = run_kernel(Q, K, V, dO, sr_dO=True, fp8_dS_mode=0)
    _, _, _, rtne_dO = run_kernel(Q, K, V, dO, sr_dO=False, fp8_dS_mode=0)

    ref = (dQ_ref, dK_ref, dV_ref)
    recipe_m = tuple(tensor_metrics(a, b) for a, b in zip(recipe, ref))
    rtne_m = tuple(tensor_metrics(a, b) for a, b in zip(rtne, ref))
    sr_dO_m = tuple(tensor_metrics(a, b) for a, b in zip(sr_dO_only, ref))
    rtne_dO_m = tuple(tensor_metrics(a, b) for a, b in zip(rtne_dO, ref))
    for label, ms in [
        ("kernel recipe SR-dO+SR-dS vs torch SDPA", recipe_m),
        ("kernel SR-dO+RTNE-dS vs torch SDPA", rtne_m),
        ("kernel SR-dO+bf16-dS vs torch SDPA", sr_dO_m),
        ("kernel RTNE-dO+bf16-dS vs torch SDPA", rtne_dO_m),
    ]:
        check_grad_metrics(label, *ms, min_qsnr=18.0, max_rel_l1=0.13, min_cos=0.985)

    o_m = tensor_metrics(O, O_sdpa)
    if o_m["qsnr_dB"] < 20.0 or o_m["rel_L1"] > 0.1 or o_m["cos"] < 0.99:
        raise AssertionError(f"kernel forward O vs torch SDPA failed: QSNR={o_m['qsnr_dB']:.2f}")
    print(f"  {'kernel O vs torch SDPA':<28} QSNR={o_m['qsnr_dB']:5.2f} relL1={o_m['rel_L1']:.2e} RMSE={o_m['rmse']:.2e} cos={o_m['cos']:.5f}")
    print_grads("kernel recipe SR-dO+SR-dS", recipe, ref)
    print_grads("kernel SR-dO + RTNE-dS", rtne, ref)
    print_grads("kernel SR-dO + bf16 dS", sr_dO_only, ref)
    print_grads("kernel RTNE-dO + bf16 dS", rtne_dO, ref)

    if bench_iters > 0:
        profile_kernels(
            shape=(B, H, N, D),
            seed=seed,
            bench_iters=bench_iters,
            warmup=bench_warmup,
            group_count=bench_groups,
            cooldown_s=bench_cooldown,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--bench-iters", type=int, default=0)
    parser.add_argument("--bench-warmup", type=int, default=500)
    parser.add_argument("--bench-groups", type=int, default=None)
    parser.add_argument("--bench-cooldown", type=float, default=0.2)
    args = parser.parse_args()

    require_extension("fp8_mha_forward", "fp8_mha_backward", "int8_quantize_per_token")
    for cfg in backward_kernel_cases(args.quick):
        run_one(
            *cfg,
            bench_iters=args.bench_iters,
            bench_warmup=args.bench_warmup,
            bench_groups=args.bench_groups,
            bench_cooldown=args.bench_cooldown,
        )


if __name__ == "__main__":
    main()
