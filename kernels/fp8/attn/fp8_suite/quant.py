import torch


FP8_E4M3_MAX = 448.0


def quantize_per_row_fp8(x):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    xq = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return xq.contiguous(), scale.squeeze(-1).contiguous()


def quantize_per_channel_fp8(x):
    amax = x.abs().amax(dim=-2, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    xq = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return xq.contiguous(), scale.squeeze(-2).contiguous()


def quantize_per_tensor_fp8(x):
    descale = (x.abs().amax().clamp_min(1e-12) / FP8_E4M3_MAX).to(torch.float32)
    xq = (x / descale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return xq.contiguous(), descale


def stochastic_round_to_fp8e4m3fn_grid(y, *, gen=None):
    min_normal = 2.0 ** -6
    sub_step = 2.0 ** -9

    sign = torch.where(y < 0, -1.0, 1.0).to(y.dtype)
    ay = y.abs().clamp(max=FP8_E4M3_MAX)

    lo_sub = torch.floor(ay / sub_step) * sub_step
    step = 2.0 ** (torch.floor(torch.log2(ay.clamp_min(min_normal))) - 3.0)
    lo_norm = torch.floor(ay / step) * step
    lo = torch.where(ay < min_normal, lo_sub, lo_norm)
    lo = torch.where(ay <= 0, torch.zeros_like(lo), lo).clamp(max=FP8_E4M3_MAX)

    step_lo = 2.0 ** (torch.floor(torch.log2(lo.clamp_min(min_normal))) - 3.0)
    hi = torch.where(lo < min_normal, lo + sub_step, lo + step_lo).clamp(max=FP8_E4M3_MAX)
    p_hi = ((ay - lo) / (hi - lo).clamp_min(torch.finfo(torch.float32).tiny)).clamp(0.0, 1.0)
    if gen is None:
        u = torch.rand_like(y)
    else:
        u = torch.rand(y.shape, generator=gen, device=y.device, dtype=y.dtype)
    rounded = torch.where(u < p_hi, hi, lo)
    rounded = torch.where(torch.isnan(y), torch.zeros_like(rounded), rounded)
    return (rounded * sign).to(torch.float8_e4m3fn)


def quantize_per_tensor_fp8_sr(x, *, gen=None):
    descale = (x.abs().amax().clamp_min(1e-12) / FP8_E4M3_MAX).to(torch.float32)
    xq = stochastic_round_to_fp8e4m3fn_grid(x / descale, gen=gen)
    return xq.contiguous(), descale


def quantize_with_descale_fp8_sr(x, descale, *, gen=None):
    return stochastic_round_to_fp8e4m3fn_grid(x / descale, gen=gen).contiguous()


def quantize_per_channel_fp8_sr(x, *, gen=None):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    xq = stochastic_round_to_fp8e4m3fn_grid(x / scale, gen=gen)
    return xq.contiguous(), scale.squeeze(-1).contiguous()


def broadcast_descale(descale, like_2d):
    return torch.full_like(like_2d, float(descale), dtype=torch.float32)

