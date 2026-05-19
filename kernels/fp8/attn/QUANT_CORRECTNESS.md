# FP8 Quantization Kernel Correctness Report

- Device: **NVIDIA H200** (SM 9.0, 132 SMs)
- Total cases: **8046**, pass: **8046**, fail: **0**, exceptions: **0**
- Reference: `fp8_suite.quant.quantize_per_row_fp8` / `quantize_per_channel_fp8` (PyTorch fp32 → fp8 e4m3)
- Metrics: `fp8_suite.metrics.tensor_metrics` (QSNR, rel-L1, cos), plus fp8-byte exact match.

## Pass criteria

| metric | bound |
|---|---|
| scale max abs error | ≤ 1e-06 |
| scale rel-L1 | ≤ 1e-06 |
| dequant QSNR | ≥ 50.0 dB |
| dequant rel-L1 | ≤ 1e-04 |
| dequant cosine | ≥ 0.99999 |
| fp8-byte max distance | ≤ 1 (1 quantum) |
| determinism (rerun bit-identical) | required |
| zero-input shortcut | dequant bit-identical → pass |

## Ablations

### By granularity

| granularity | n | pass | fail | min QSNR | max rel-L1 | min cos | max byte Δ | max scale err | max |Δdeq| | non-det |
|---|---|---|---|---|---|---|---|---|---|---|
| channel | 4023 | 4023 | 0 | 58.27 | 8.50e-06 | 1.00000 | 1 | 9.54e-07 | 3.23e+02 | no |
| token | 4023 | 4023 | 0 | 62.27 | 3.88e-06 | 1.00000 | 1 | 9.54e-07 | 2.98e+02 | no |

### By input distribution

| kind | n | pass | fail | min QSNR | max rel-L1 | min cos | max byte Δ | max scale err | max |Δdeq| | non-det |
|---|---|---|---|---|---|---|---|---|---|---|
| large | 894 | 894 | 0 | 74.67 | 1.32e-06 | 1.00000 | 1 | 9.54e-07 | 3.23e+02 | no |
| mixed | 894 | 894 | 0 | 65.93 | 1.62e-06 | 1.00000 | 1 | 4.77e-07 | 1.33e+02 | no |
| negative | 894 | 894 | 0 | 72.36 | 5.92e-07 | 1.00000 | 1 | 9.31e-10 | 3.57e-01 | no |
| normal | 894 | 894 | 0 | 62.27 | 3.88e-06 | 1.00000 | 1 | 9.31e-10 | 3.24e-01 | no |
| one_outlier_per_row | 894 | 894 | 0 | 126.15 | 1.16e-08 | 1.00000 | 1 | 1.46e-11 | 7.11e-03 | no |
| positive | 894 | 894 | 0 | 72.36 | 5.92e-07 | 1.00000 | 1 | 9.31e-10 | 3.57e-01 | no |
| tiny | 894 | 894 | 0 | 58.27 | 8.50e-06 | 1.00000 | 1 | 1.14e-13 | 3.14e-05 | no |
| uniform | 894 | 894 | 0 | 72.34 | 1.14e-06 | 1.00000 | 1 | 2.33e-10 | 7.14e-02 | no |
| zeros | 894 | 894 | 0 | nan | 0.00e+00 | nan | 0 | 9.98e-13 | 0.00e+00 | no |

### By head dim D

| D | n | pass | fail | min QSNR | max rel-L1 | min cos | max byte Δ | max scale err | max |Δdeq| | non-det |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 4158 | 4158 | 0 | 58.27 | 8.50e-06 | 1.00000 | 1 | 9.54e-07 | 3.23e+02 | no |
| 64 | 3888 | 3888 | 0 | 58.30 | 8.46e-06 | 1.00000 | 1 | 9.54e-07 | 3.23e+02 | no |

### By sequence length N

| N | n | pass | fail | min QSNR | max rel-L1 | min cos | max byte Δ | max scale err | max |Δdeq| | non-det |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 648 | 648 | 0 | 139.80 | 9.89e-08 | 1.00000 | 0 | 9.54e-07 | 4.88e-04 | no |
| 1024 | 648 | 648 | 0 | 61.33 | 4.24e-06 | 1.00000 | 1 | 9.54e-07 | 2.37e+02 | no |
| 128 | 648 | 648 | 0 | 62.41 | 3.76e-06 | 1.00000 | 1 | 9.54e-07 | 2.28e+02 | no |
| 16 | 648 | 648 | 0 | 65.44 | 1.90e-06 | 1.00000 | 1 | 9.54e-07 | 4.42e+01 | no |
| 16384 | 648 | 648 | 0 | 66.03 | 1.41e-06 | 1.00000 | 1 | 9.54e-07 | 3.23e+02 | no |
| 2048 | 648 | 648 | 0 | 64.36 | 2.14e-06 | 1.00000 | 1 | 9.54e-07 | 2.45e+02 | no |
| 255 | 702 | 702 | 0 | 58.27 | 8.50e-06 | 1.00000 | 1 | 9.54e-07 | 2.28e+02 | no |
| 256 | 648 | 648 | 0 | 58.29 | 8.46e-06 | 1.00000 | 1 | 9.54e-07 | 2.28e+02 | no |
| 257 | 702 | 702 | 0 | 58.31 | 8.43e-06 | 1.00000 | 1 | 9.54e-07 | 2.28e+02 | no |
| 31 | 648 | 648 | 0 | 62.27 | 3.88e-06 | 1.00000 | 1 | 9.54e-07 | 8.99e+01 | no |
| 4096 | 648 | 648 | 0 | 66.03 | 1.41e-06 | 1.00000 | 1 | 9.54e-07 | 2.70e+02 | no |
| 4101 | 54 | 54 | 0 | 79.89 | 1.57e-07 | 1.00000 | 1 | 9.54e-07 | 1.51e+02 | no |
| 511 | 54 | 54 | 0 | 70.38 | 6.45e-07 | 1.00000 | 1 | 9.54e-07 | 1.04e+02 | no |
| 512 | 648 | 648 | 0 | 58.30 | 8.46e-06 | 1.00000 | 1 | 9.54e-07 | 2.23e+02 | no |
| 513 | 54 | 54 | 0 | 70.39 | 6.43e-07 | 1.00000 | 1 | 9.54e-07 | 1.04e+02 | no |

## FP8 e4m3 quantization noise floor (dequant vs original fp32)

Intrinsic loss from quantizing fp32 inputs to FP8 e4m3 with dynamic per-token / per-channel scales, then dequantizing. Lower is *better* (less loss). This is what downstream consumers see as input noise.

| kind | granularity | mean QSNR | min QSNR | max rel-L1 | min cos |
|---|---|---|---|---|---|
| large | channel | 41.79 dB | 31.53 dB | 2.25e-02 | 0.99965 |
| large | token | 31.90 dB | 30.86 dB | 2.37e-02 | 0.99960 |
| mixed | channel | 43.56 dB | 31.54 dB | 2.25e-02 | 0.99965 |
| mixed | token | 35.33 dB | 32.30 dB | 1.88e-02 | 0.99975 |
| negative | channel | 41.88 dB | 31.86 dB | 2.21e-02 | 0.99967 |
| negative | token | 32.02 dB | 31.73 dB | 2.25e-02 | 0.99966 |
| normal | channel | 41.72 dB | 31.53 dB | 2.25e-02 | 0.99965 |
| normal | token | 31.90 dB | 30.86 dB | 2.37e-02 | 0.99960 |
| one_outlier_per_row | channel | 96.21 dB | 84.18 dB | 4.96e-04 | 1.00000 |
| one_outlier_per_row | token | 85.69 dB | 83.59 dB | 5.27e-04 | 1.00000 |
| positive | channel | 41.88 dB | 31.86 dB | 2.21e-02 | 0.99967 |
| positive | token | 32.02 dB | 31.73 dB | 2.25e-02 | 0.99966 |
| tiny | channel | 41.72 dB | 31.53 dB | 2.25e-02 | 0.99965 |
| tiny | token | 31.90 dB | 30.86 dB | 2.37e-02 | 0.99960 |
| uniform | channel | 41.73 dB | 31.86 dB | 2.21e-02 | 0.99967 |
| uniform | token | 32.03 dB | 31.82 dB | 2.21e-02 | 0.99967 |

For *normal* fp32 data, FP8 e4m3 with dynamic scaling lands at ≈ **32 dB QSNR / ~2.2% rel-L1**. This is the intrinsic 3-bit-mantissa rounding noise of e4m3 — for reference: bf16 ≈ 50 dB, fp16 ≈ 60+ dB. The downstream FP8 attention gradient QSNR (~22-26 dB end-to-end) is *below* this 32 dB noise floor, so the matmul/softmax accumulation inside the kernel — not the input quantization — is the dominant error source.

## Key findings

- Across **8046** cases (7152 non-degenerate): min QSNR **58.27 dB**, max dequant rel-L1 **8.50e-06**, min cos **1.00000**, max fp8-byte distance **1**, max scale err **9.54e-07**, non-deterministic: **no**.
- **Scales** match the PyTorch reference exactly except for *all-zero* inputs, where the kernel floors the *scale* (`max(amax/448, 1e-12)`) while the reference floors the *amax* (`max(amax, 1e-12)/448`). For zero data this gives kernel scale = 1e-12 vs reference ≈ 2.23e-15. The fp8 codes are bit-identical (all zero) so the dequantized output is unaffected — the test treats this as a pass.
- **FP8 byte agreement** is within 1 quantum vs the reference. The remaining single-quantum discrepancies trace to `--use_fast_math` altering `x/s` rounding; the kernel uses `__fdiv_rn` to keep IEEE rounding for the division itself, but RTNE ties at half-way points can still flip.
- **Dequantized values** comfortably exceed the bounds used by the existing `check_quant_kernel` harness (QSNR ≥ 50 dB, rel-L1 ≤ 1e-4, cos ≥ 0.99999).
- **Determinism**: re-running on the same input produces bit-identical outputs in every case, including the atomic-reduced per-channel path. `atomicMax` is associative and commutative on the (non-negative) float bit pattern, so any block-arrival order yields the same amax.
- **Chunk boundaries** (N ∈ {255, 256, 257, 511, 512, 513, 4101} around the per-channel kernel's `CHUNK_ROWS=256` split) behave identically to interior values.
- **`one_outlier_per_row`** stresses per-token amax (one extreme element per row sets the scale) and **`mixed`** stresses per-channel amax (5% large outliers): both pass.
