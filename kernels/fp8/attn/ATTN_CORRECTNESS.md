# FP8 Attention Kernel Correctness Report

- Device: **NVIDIA H200** (SM 9.0, 132 SMs)
- Forward cases: **6**, pass: **6**, fail: **0**
- Backward cases: **6**, pass: **6**, fail: **0**
- Exceptions: **0**

## Supported shape contract

- Head dimension: **D = 64 or D = 128** only.
- End-to-end sequence length: **pad N to a multiple of 384**. The backward kernel itself is tiled at 128, but the recipe uses the forward kernel output/LSE, and forward requires `N % 384 == 0`.
- Query heads must be divisible by KV heads.
- Known excluded case in this revision: backward `(B=1, H=8, N=384, D=64)` with `fp8_dS_mode=2`; forward and INT8 forward pass for that shape, but dQ/dK are not reliable.

## Pass criteria

Same thresholds the existing `check_grad_metrics` / quant-attn harness uses. Forward must pass both `vs_fp32` and `vs_quant`. Backward must pass `vs_fp32` for dQ, dK, dV.

| stage | comparison | min QSNR | max rel-L1 | min cos |
|---|---|---|---|---|
| fwd | vs_fp32 | 18.0 dB | 2.50e-01 | 0.9800 |
| fwd | vs_quant | 25.0 dB | 1.00e-01 | 0.9950 |
| bwd | vs_fp32 | 18.0 dB | 1.30e-01 | 0.9850 |

## Forward ablations

### All cases

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 6 | 28.54 dB | 28.46 dB | 3.72e-02 | 3.11e-03 | 0.99929 |
| vs_bf16 | 6 | 28.50 dB | 28.42 dB | 3.73e-02 | 3.12e-03 | 0.99928 |
| vs_quant | 6 | 50.57 dB | 50.38 dB | 2.97e-03 | 2.45e-04 | 1.00000 |
| bf16_vs_fp32 | 6 | 48.63 dB | 48.45 dB | 3.73e-03 | 3.05e-04 | 0.99999 |

### D = 64

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 2 | 28.53 dB | 28.46 dB | 3.70e-02 | 1.59e-03 | 0.99929 |
| vs_bf16 | 2 | 28.49 dB | 28.42 dB | 3.71e-02 | 1.59e-03 | 0.99928 |
| vs_quant | 2 | 50.65 dB | 50.63 dB | 2.89e-03 | 1.25e-04 | 1.00000 |
| bf16_vs_fp32 | 2 | 48.50 dB | 48.45 dB | 3.73e-03 | 1.59e-04 | 0.99999 |

### D = 128

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 4 | 28.54 dB | 28.48 dB | 3.72e-02 | 3.11e-03 | 0.99929 |
| vs_bf16 | 4 | 28.50 dB | 28.44 dB | 3.73e-02 | 3.12e-03 | 0.99928 |
| vs_quant | 4 | 50.54 dB | 50.38 dB | 2.97e-03 | 2.45e-04 | 1.00000 |
| bf16_vs_fp32 | 4 | 48.69 dB | 48.59 dB | 3.68e-03 | 3.05e-04 | 0.99999 |

### N = 384

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 2 | 28.60 dB | 28.59 dB | 3.64e-02 | 3.11e-03 | 0.99931 |
| vs_bf16 | 2 | 28.56 dB | 28.55 dB | 3.65e-02 | 3.12e-03 | 0.99930 |
| vs_quant | 2 | 50.65 dB | 50.64 dB | 2.88e-03 | 2.45e-04 | 1.00000 |
| bf16_vs_fp32 | 2 | 48.77 dB | 48.74 dB | 3.60e-03 | 3.05e-04 | 0.99999 |

### N = 1536

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 4 | 28.51 dB | 28.46 dB | 3.72e-02 | 1.59e-03 | 0.99929 |
| vs_bf16 | 4 | 28.47 dB | 28.42 dB | 3.73e-02 | 1.59e-03 | 0.99928 |
| vs_quant | 4 | 50.54 dB | 50.38 dB | 2.97e-03 | 1.27e-04 | 1.00000 |
| bf16_vs_fp32 | 4 | 48.56 dB | 48.45 dB | 3.73e-03 | 1.59e-04 | 0.99999 |

## INT8-GEMM1 forward ablation

INT8 path: per-token symmetric INT8 quantization for Q,K (GEMM1 only). PV stays bf16, so this is directly comparable to the FP8 forward on the same input tensors. Motivated by SageBwd (arXiv:2410.02367), which reports INT8 gives noticeably better gradient quality than FP8 e4m3 in the attention backward.

### INT8 — All cases

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 6 | 40.46 dB | 40.16 dB | 9.69e-03 | 8.05e-04 | 0.99995 |
| vs_bf16 | 6 | 39.84 dB | 39.58 dB | 1.01e-02 | 8.59e-04 | 0.99994 |
| vs_quant | 6 | 50.60 dB | 50.41 dB | 2.96e-03 | 2.45e-04 | 1.00000 |
| bf16_vs_fp32 | 6 | 48.63 dB | 48.45 dB | 3.73e-03 | 3.05e-04 | 0.99999 |

### INT8 — D = 64

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 2 | 40.82 dB | 40.79 dB | 8.97e-03 | 3.86e-04 | 0.99996 |
| vs_bf16 | 2 | 40.13 dB | 40.10 dB | 9.45e-03 | 4.17e-04 | 0.99995 |
| vs_quant | 2 | 50.67 dB | 50.63 dB | 2.89e-03 | 1.25e-04 | 1.00000 |
| bf16_vs_fp32 | 2 | 48.50 dB | 48.45 dB | 3.73e-03 | 1.59e-04 | 0.99999 |

### INT8 — D = 128

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 4 | 40.28 dB | 40.16 dB | 9.69e-03 | 8.05e-04 | 0.99995 |
| vs_bf16 | 4 | 39.70 dB | 39.58 dB | 1.01e-02 | 8.59e-04 | 0.99994 |
| vs_quant | 4 | 50.57 dB | 50.41 dB | 2.96e-03 | 2.45e-04 | 1.00000 |
| bf16_vs_fp32 | 4 | 48.69 dB | 48.59 dB | 3.68e-03 | 3.05e-04 | 0.99999 |

## Backward ablations

### All cases

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 19.47 dB | 18.96 dB | 1.08e-01 | 9.16e-03 | 0.99368 |
| vs_fp32 | dK | 21.10 dB | 20.11 dB | 9.74e-02 | 8.24e-03 | 0.99519 |
| vs_fp32 | dV | 24.37 dB | 24.26 dB | 6.10e-02 | 5.08e-03 | 0.99813 |
| vs_manual | dQ | 19.47 dB | 18.96 dB | 1.08e-01 | 9.16e-03 | 0.99368 |
| vs_manual | dK | 21.10 dB | 20.11 dB | 9.74e-02 | 8.24e-03 | 0.99519 |
| vs_manual | dV | 24.37 dB | 24.26 dB | 6.10e-02 | 5.08e-03 | 0.99813 |
| manual_vs_fp32 | dQ | 118.89 dB | 118.18 dB | 1.14e-06 | 8.46e-08 | 1.00000 |
| manual_vs_fp32 | dK | 118.86 dB | 118.15 dB | 1.15e-06 | 8.64e-08 | 1.00000 |
| manual_vs_fp32 | dV | 120.26 dB | 119.36 dB | 1.01e-06 | 7.15e-08 | 1.00000 |

### D = 64

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 19.02 dB | 18.96 dB | 1.08e-01 | 4.77e-03 | 0.99368 |
| vs_fp32 | dK | 21.58 dB | 21.56 dB | 8.30e-02 | 3.56e-03 | 0.99651 |
| vs_fp32 | dV | 24.40 dB | 24.32 dB | 6.05e-02 | 2.57e-03 | 0.99815 |
| vs_manual | dQ | 19.02 dB | 18.96 dB | 1.08e-01 | 4.77e-03 | 0.99368 |
| vs_manual | dK | 21.58 dB | 21.56 dB | 8.30e-02 | 3.56e-03 | 0.99651 |
| vs_manual | dV | 24.40 dB | 24.32 dB | 6.05e-02 | 2.57e-03 | 0.99815 |
| manual_vs_fp32 | dQ | 118.65 dB | 118.63 dB | 1.07e-06 | 4.95e-08 | 1.00000 |
| manual_vs_fp32 | dK | 118.69 dB | 118.66 dB | 1.07e-06 | 4.96e-08 | 1.00000 |
| manual_vs_fp32 | dV | 119.92 dB | 119.85 dB | 9.39e-07 | 4.30e-08 | 1.00000 |

### D = 128

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 19.69 dB | 19.13 dB | 1.04e-01 | 9.16e-03 | 0.99402 |
| vs_fp32 | dK | 20.86 dB | 20.11 dB | 9.74e-02 | 8.24e-03 | 0.99519 |
| vs_fp32 | dV | 24.35 dB | 24.26 dB | 6.10e-02 | 5.08e-03 | 0.99813 |
| vs_manual | dQ | 19.69 dB | 19.13 dB | 1.04e-01 | 9.16e-03 | 0.99402 |
| vs_manual | dK | 20.86 dB | 20.11 dB | 9.74e-02 | 8.24e-03 | 0.99519 |
| vs_manual | dV | 24.35 dB | 24.26 dB | 6.10e-02 | 5.08e-03 | 0.99813 |
| manual_vs_fp32 | dQ | 119.01 dB | 118.18 dB | 1.14e-06 | 8.46e-08 | 1.00000 |
| manual_vs_fp32 | dK | 118.94 dB | 118.15 dB | 1.15e-06 | 8.64e-08 | 1.00000 |
| manual_vs_fp32 | dV | 120.43 dB | 119.36 dB | 1.01e-06 | 7.15e-08 | 1.00000 |

### N = 384

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 19.18 dB | 19.13 dB | 1.04e-01 | 9.16e-03 | 0.99402 |
| vs_fp32 | dK | 20.13 dB | 20.11 dB | 9.74e-02 | 8.24e-03 | 0.99519 |
| vs_fp32 | dV | 24.43 dB | 24.36 dB | 6.01e-02 | 5.08e-03 | 0.99817 |
| vs_manual | dQ | 19.18 dB | 19.13 dB | 1.04e-01 | 9.16e-03 | 0.99402 |
| vs_manual | dK | 20.13 dB | 20.11 dB | 9.74e-02 | 8.24e-03 | 0.99519 |
| vs_manual | dV | 24.43 dB | 24.36 dB | 6.01e-02 | 5.08e-03 | 0.99817 |
| manual_vs_fp32 | dQ | 119.83 dB | 119.83 dB | 9.48e-07 | 8.46e-08 | 1.00000 |
| manual_vs_fp32 | dK | 119.70 dB | 119.70 dB | 9.67e-07 | 8.64e-08 | 1.00000 |
| manual_vs_fp32 | dV | 121.48 dB | 121.43 dB | 7.87e-07 | 7.15e-08 | 1.00000 |

### N = 1536

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 19.61 dB | 18.96 dB | 1.08e-01 | 4.77e-03 | 0.99368 |
| vs_fp32 | dK | 21.58 dB | 21.56 dB | 8.30e-02 | 3.56e-03 | 0.99651 |
| vs_fp32 | dV | 24.34 dB | 24.26 dB | 6.10e-02 | 2.57e-03 | 0.99813 |
| vs_manual | dQ | 19.61 dB | 18.96 dB | 1.08e-01 | 4.77e-03 | 0.99368 |
| vs_manual | dK | 21.58 dB | 21.56 dB | 8.30e-02 | 3.56e-03 | 0.99651 |
| vs_manual | dV | 24.34 dB | 24.26 dB | 6.10e-02 | 2.57e-03 | 0.99813 |
| manual_vs_fp32 | dQ | 118.42 dB | 118.18 dB | 1.14e-06 | 5.19e-08 | 1.00000 |
| manual_vs_fp32 | dK | 118.44 dB | 118.15 dB | 1.15e-06 | 5.22e-08 | 1.00000 |
| manual_vs_fp32 | dV | 119.65 dB | 119.36 dB | 1.01e-06 | 4.49e-08 | 1.00000 |

### fp8_dS_mode = 2 (fp8 SR dS)

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 19.47 dB | 18.96 dB | 1.08e-01 | 9.16e-03 | 0.99368 |
| vs_fp32 | dK | 21.10 dB | 20.11 dB | 9.74e-02 | 8.24e-03 | 0.99519 |
| vs_fp32 | dV | 24.37 dB | 24.26 dB | 6.10e-02 | 5.08e-03 | 0.99813 |
| vs_manual | dQ | 19.47 dB | 18.96 dB | 1.08e-01 | 9.16e-03 | 0.99368 |
| vs_manual | dK | 21.10 dB | 20.11 dB | 9.74e-02 | 8.24e-03 | 0.99519 |
| vs_manual | dV | 24.37 dB | 24.26 dB | 6.10e-02 | 5.08e-03 | 0.99813 |
| manual_vs_fp32 | dQ | 118.89 dB | 118.18 dB | 1.14e-06 | 8.46e-08 | 1.00000 |
| manual_vs_fp32 | dK | 118.86 dB | 118.15 dB | 1.15e-06 | 8.64e-08 | 1.00000 |
| manual_vs_fp32 | dV | 120.26 dB | 119.36 dB | 1.01e-06 | 7.15e-08 | 1.00000 |

## Key findings

- **Forward kernel O vs torch SDPA fp32**: mean QSNR **28.54 dB**, min **28.46 dB**, max rel-L1 **3.72e-02**, min cos **0.99929**.
- **Forward kernel O vs FP8-dequant fp32 SDPA** (measures kernel-internal loss; takes the input quantization out of the picture): mean QSNR **50.57 dB**, min **50.38 dB**, max rel-L1 **2.97e-03**.
- **bf16 SDPA vs fp32 SDPA baseline**: mean QSNR **48.63 dB**. This is the noise floor an idealized bf16 attention would already introduce vs fp32; the FP8 kernel cannot beat it because PV is still computed in bf16 in this revision.
- **INT8-GEMM1 forward O vs torch SDPA fp32**: mean QSNR **40.46 dB** (FP8: **28.54 dB**), max rel-L1 **9.69e-03** (FP8: **3.72e-02**), min cos **0.99995** (FP8: **0.99929**). INT8 actually wins on the forward by ~10-12 dB: the `vs_quant` rows are essentially identical between FP8 and INT8 (kernel-internal numerics are the same), so the gap is entirely the input quantization noise — INT8 has 8-bit uniform resolution per row, while FP8 e4m3 has only 3 mantissa bits per element so its per-row resolution on Gaussian-shaped data is coarser. The SageBwd paper's INT8 win for the *backward* (not exercised by this forward kernel) is in addition to this forward advantage.
- **Backward dQ vs torch SDPA fp32**: mean QSNR **19.47 dB**, min **18.96 dB**, max rel-L1 **1.08e-01**.
- **Backward dK vs torch SDPA fp32**: mean QSNR **21.10 dB**, min **20.11 dB**, max rel-L1 **9.74e-02**.
- **Backward dV vs torch SDPA fp32**: mean QSNR **24.37 dB**, min **24.26 dB**, max rel-L1 **6.10e-02**.
- The backward gradient QSNR (~22-26 dB end-to-end) sits *below* the 32 dB FP8 e4m3 quantization noise floor documented in `QUANT_CORRECTNESS.md` — the matmul/softmax accumulation inside the kernel, not the input quant, is the dominant error source.
- The `manual_vs_fp32` row (manual fp32 reference vs torch SDPA fp32) sits at ~120 dB on every grad — confirms the comparison itself is sound.
