def require_extension(*symbols):
    try:
        import _C
    except ImportError as exc:
        raise SystemExit(
            "Build the FP8 extension first:\n"
            "  cd attn && make BUILD_MODE=torch KERNEL=fp8"
        ) from exc
    missing = [name for name in symbols if not hasattr(_C, name)]
    if missing:
        raise SystemExit(
            "Loaded _C is missing "
            + ", ".join(missing)
            + ". Rebuild with `make BUILD_MODE=torch KERNEL=fp8`."
        )
    return _C


def cuda_quantize_per_token(x):
    return require_extension("fp8_quantize_per_token").fp8_quantize_per_token(x.contiguous())


def cuda_quantize_per_channel(x):
    return require_extension("fp8_quantize_per_channel").fp8_quantize_per_channel(x.contiguous())


def cuda_quantize_per_token_int8(x):
    return require_extension("int8_quantize_per_token").int8_quantize_per_token(x.contiguous())


def int8_forward(prepared):
    return require_extension("int8_mha_forward").int8_mha_forward(
        prepared.Qq,
        prepared.Kq,
        prepared.Vbf,
        prepared.sq.to("cuda").contiguous().to(prepared.sq.dtype),
        prepared.sk.to("cuda").contiguous().to(prepared.sk.dtype),
        prepared.vm,
    )


def cuda_quantize_per_token_out(x, xq, scale):
    require_extension("fp8_quantize_per_token_out").fp8_quantize_per_token_out(x.contiguous(), xq, scale)


def cuda_quantize_per_channel_out(x, xq, scale):
    require_extension("fp8_quantize_per_channel_out").fp8_quantize_per_channel_out(x.contiguous(), xq, scale)


def fp8_forward(prepared):
    return require_extension("fp8_mha_forward").fp8_mha_forward(
        prepared.Qq,
        prepared.Kq,
        prepared.Vbf,
        prepared.sq.to("cuda").contiguous().to(prepared.sq.dtype),
        prepared.sk.to("cuda").contiguous().to(prepared.sk.dtype),
        prepared.vm,
    )


def fp8_backward(prepared, *, fp8_dS_mode=0):
    return require_extension("fp8_mha_backward").fp8_mha_backward(
        prepared.Qq,
        prepared.Kq,
        prepared.Vq,
        prepared.dOq,
        prepared.Qq_t,
        prepared.dOq_t,
        prepared.K_bf,
        prepared.O_bf,
        prepared.dO_bf,
        prepared.L_raw,
        prepared.sq.contiguous(),
        prepared.sk.contiguous(),
        prepared.sv.contiguous(),
        prepared.sdo_row.contiguous(),
        prepared.sdp_row.contiguous(),
        prepared.sq_ch.contiguous(),
        prepared.sdo_ch.contiguous(),
        fp8_dS_mode=fp8_dS_mode,
    )

