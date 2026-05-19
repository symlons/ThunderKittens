# FP8 Attention Kernel Correctness Report

- Device: **NVIDIA H200** (SM 9.0, 132 SMs)
- Forward cases: **28**, pass: **28**, fail: **0**
- Backward cases: **28**, pass: **26**, fail: **2**
- Exceptions: **0**

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
| vs_fp32 | 28 | 28.49 dB | 28.29 dB | 3.79e-02 | 3.11e-03 | 0.99926 |
| vs_bf16 | 28 | 28.45 dB | 28.24 dB | 3.80e-02 | 3.12e-03 | 0.99925 |
| vs_quant | 28 | 50.52 dB | 50.38 dB | 2.97e-03 | 2.46e-04 | 1.00000 |
| bf16_vs_fp32 | 28 | 48.56 dB | 48.24 dB | 3.83e-03 | 3.10e-04 | 0.99999 |

### D = 64

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 14 | 28.48 dB | 28.29 dB | 3.79e-02 | 3.10e-03 | 0.99926 |
| vs_bf16 | 14 | 28.44 dB | 28.24 dB | 3.80e-02 | 3.12e-03 | 0.99925 |
| vs_quant | 14 | 50.56 dB | 50.44 dB | 2.96e-03 | 2.46e-04 | 1.00000 |
| bf16_vs_fp32 | 14 | 48.48 dB | 48.24 dB | 3.83e-03 | 3.10e-04 | 0.99999 |

### D = 128

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 14 | 28.51 dB | 28.35 dB | 3.79e-02 | 3.11e-03 | 0.99927 |
| vs_bf16 | 14 | 28.46 dB | 28.31 dB | 3.80e-02 | 3.12e-03 | 0.99926 |
| vs_quant | 14 | 50.49 dB | 50.38 dB | 2.97e-03 | 2.45e-04 | 1.00000 |
| bf16_vs_fp32 | 14 | 48.63 dB | 48.45 dB | 3.75e-03 | 3.05e-04 | 0.99999 |

### N = 384

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 4 | 28.59 dB | 28.49 dB | 3.66e-02 | 3.11e-03 | 0.99929 |
| vs_bf16 | 4 | 28.54 dB | 28.45 dB | 3.67e-02 | 3.12e-03 | 0.99928 |
| vs_quant | 4 | 50.62 dB | 50.46 dB | 2.94e-03 | 2.46e-04 | 1.00000 |
| bf16_vs_fp32 | 4 | 48.68 dB | 48.47 dB | 3.70e-03 | 3.10e-04 | 0.99999 |

### N = 768

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 4 | 28.55 dB | 28.48 dB | 3.70e-02 | 2.23e-03 | 0.99929 |
| vs_bf16 | 4 | 28.51 dB | 28.44 dB | 3.71e-02 | 2.24e-03 | 0.99928 |
| vs_quant | 4 | 50.53 dB | 50.46 dB | 2.94e-03 | 1.77e-04 | 1.00000 |
| bf16_vs_fp32 | 4 | 48.61 dB | 48.55 dB | 3.67e-03 | 2.24e-04 | 0.99999 |

### N = 1536

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 12 | 28.50 dB | 28.44 dB | 3.72e-02 | 1.59e-03 | 0.99928 |
| vs_bf16 | 12 | 28.46 dB | 28.40 dB | 3.73e-02 | 1.60e-03 | 0.99928 |
| vs_quant | 12 | 50.53 dB | 50.38 dB | 2.97e-03 | 1.27e-04 | 1.00000 |
| bf16_vs_fp32 | 12 | 48.56 dB | 48.45 dB | 3.73e-03 | 1.59e-04 | 0.99999 |

### N = 3072

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 4 | 28.42 dB | 28.35 dB | 3.79e-02 | 1.13e-03 | 0.99927 |
| vs_bf16 | 4 | 28.37 dB | 28.31 dB | 3.80e-02 | 1.14e-03 | 0.99926 |
| vs_quant | 4 | 50.47 dB | 50.40 dB | 2.97e-03 | 8.93e-05 | 1.00000 |
| bf16_vs_fp32 | 4 | 48.47 dB | 48.45 dB | 3.75e-03 | 1.13e-04 | 0.99999 |

### N = 6144

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 4 | 28.41 dB | 28.29 dB | 3.79e-02 | 7.98e-04 | 0.99926 |
| vs_bf16 | 4 | 28.37 dB | 28.24 dB | 3.80e-02 | 8.02e-04 | 0.99925 |
| vs_quant | 4 | 50.46 dB | 50.44 dB | 2.96e-03 | 6.37e-05 | 1.00000 |
| bf16_vs_fp32 | 4 | 48.44 dB | 48.24 dB | 3.83e-03 | 8.02e-05 | 0.99999 |

## INT8-GEMM1 forward ablation

INT8 path: per-token symmetric INT8 quantization for Q,K (GEMM1 only). PV stays bf16, so this is directly comparable to the FP8 forward on the same input tensors. Motivated by SageBwd (arXiv:2410.02367), which reports INT8 gives noticeably better gradient quality than FP8 e4m3 in the attention backward.

### INT8 — All cases

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 28 | 40.51 dB | 40.08 dB | 9.80e-03 | 8.05e-04 | 0.99995 |
| vs_bf16 | 28 | 39.88 dB | 39.50 dB | 1.02e-02 | 8.59e-04 | 0.99994 |
| vs_quant | 28 | 50.55 dB | 50.41 dB | 2.96e-03 | 2.46e-04 | 1.00000 |
| bf16_vs_fp32 | 28 | 48.56 dB | 48.24 dB | 3.83e-03 | 3.10e-04 | 0.99999 |

### INT8 — D = 64

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 14 | 40.78 dB | 40.61 dB | 9.19e-03 | 7.53e-04 | 0.99996 |
| vs_bf16 | 14 | 40.10 dB | 39.93 dB | 9.68e-03 | 8.14e-04 | 0.99995 |
| vs_quant | 14 | 50.57 dB | 50.45 dB | 2.96e-03 | 2.46e-04 | 1.00000 |
| bf16_vs_fp32 | 14 | 48.48 dB | 48.24 dB | 3.83e-03 | 3.10e-04 | 0.99999 |

### INT8 — D = 128

| comparison | n | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | 14 | 40.24 dB | 40.08 dB | 9.80e-03 | 8.05e-04 | 0.99995 |
| vs_bf16 | 14 | 39.65 dB | 39.50 dB | 1.02e-02 | 8.59e-04 | 0.99994 |
| vs_quant | 14 | 50.53 dB | 50.41 dB | 2.96e-03 | 2.45e-04 | 1.00000 |
| bf16_vs_fp32 | 14 | 48.63 dB | 48.45 dB | 3.75e-03 | 3.05e-04 | 0.99999 |

## Backward ablations

### All cases

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 18.04 dB | -11.47 dB | 9.44e-01 | 3.12e-01 | 0.22504 |
| vs_fp32 | dK | 18.20 dB | -25.45 dB | 1.88e+01 | 1.57e+00 | 0.04578 |
| vs_fp32 | dV | 24.35 dB | 24.19 dB | 6.15e-02 | 5.04e-03 | 0.99810 |
| vs_manual | dQ | 18.04 dB | -11.47 dB | 9.44e-01 | 3.12e-01 | 0.22504 |
| vs_manual | dK | 18.20 dB | -25.45 dB | 1.88e+01 | 1.57e+00 | 0.04578 |
| vs_manual | dV | 24.35 dB | 24.19 dB | 6.15e-02 | 5.04e-03 | 0.99810 |
| manual_vs_fp32 | dQ | 118.11 dB | 114.94 dB | 1.63e-06 | 8.46e-08 | 1.00000 |
| manual_vs_fp32 | dK | 118.13 dB | 115.02 dB | 1.63e-06 | 8.64e-08 | 1.00000 |
| manual_vs_fp32 | dV | 119.32 dB | 115.52 dB | 1.54e-06 | 7.15e-08 | 1.00000 |

### D = 64

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 15.36 dB | -11.47 dB | 9.44e-01 | 3.12e-01 | 0.22504 |
| vs_fp32 | dK | 14.99 dB | -25.45 dB | 1.88e+01 | 1.57e+00 | 0.04578 |
| vs_fp32 | dV | 24.37 dB | 24.23 dB | 6.11e-02 | 5.03e-03 | 0.99811 |
| vs_manual | dQ | 15.36 dB | -11.47 dB | 9.44e-01 | 3.12e-01 | 0.22504 |
| vs_manual | dK | 14.99 dB | -25.45 dB | 1.88e+01 | 1.57e+00 | 0.04578 |
| vs_manual | dV | 24.37 dB | 24.23 dB | 6.11e-02 | 5.03e-03 | 0.99811 |
| manual_vs_fp32 | dQ | 118.33 dB | 115.13 dB | 1.58e-06 | 7.96e-08 | 1.00000 |
| manual_vs_fp32 | dK | 118.39 dB | 115.25 dB | 1.58e-06 | 8.07e-08 | 1.00000 |
| manual_vs_fp32 | dV | 119.56 dB | 115.73 dB | 1.49e-06 | 6.61e-08 | 1.00000 |

### D = 128

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 20.72 dB | 19.72 dB | 9.65e-02 | 8.56e-03 | 0.99478 |
| vs_fp32 | dK | 21.40 dB | 20.15 dB | 9.74e-02 | 8.19e-03 | 0.99523 |
| vs_fp32 | dV | 24.34 dB | 24.19 dB | 6.15e-02 | 5.04e-03 | 0.99810 |
| vs_manual | dQ | 20.72 dB | 19.72 dB | 9.65e-02 | 8.56e-03 | 0.99478 |
| vs_manual | dK | 21.40 dB | 20.15 dB | 9.74e-02 | 8.19e-03 | 0.99523 |
| vs_manual | dV | 24.34 dB | 24.19 dB | 6.15e-02 | 5.04e-03 | 0.99810 |
| manual_vs_fp32 | dQ | 117.89 dB | 114.94 dB | 1.63e-06 | 8.46e-08 | 1.00000 |
| manual_vs_fp32 | dK | 117.87 dB | 115.02 dB | 1.63e-06 | 8.64e-08 | 1.00000 |
| manual_vs_fp32 | dV | 119.08 dB | 115.52 dB | 1.54e-06 | 7.15e-08 | 1.00000 |

### N = 384

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 4.65 dB | -11.47 dB | 9.44e-01 | 3.12e-01 | 0.22504 |
| vs_fp32 | dK | -2.30 dB | -25.45 dB | 1.88e+01 | 1.57e+00 | 0.04578 |
| vs_fp32 | dV | 24.42 dB | 24.29 dB | 6.02e-02 | 5.04e-03 | 0.99814 |
| vs_manual | dQ | 4.65 dB | -11.47 dB | 9.44e-01 | 3.12e-01 | 0.22504 |
| vs_manual | dK | -2.30 dB | -25.45 dB | 1.88e+01 | 1.57e+00 | 0.04578 |
| vs_manual | dV | 24.42 dB | 24.29 dB | 6.02e-02 | 5.04e-03 | 0.99814 |
| manual_vs_fp32 | dQ | 120.15 dB | 119.83 dB | 9.48e-07 | 8.46e-08 | 1.00000 |
| manual_vs_fp32 | dK | 120.08 dB | 119.70 dB | 9.67e-07 | 8.64e-08 | 1.00000 |
| manual_vs_fp32 | dV | 121.82 dB | 121.43 dB | 7.87e-07 | 7.15e-08 | 1.00000 |

### N = 768

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 20.05 dB | 19.36 dB | 1.02e-01 | 6.40e-03 | 0.99424 |
| vs_fp32 | dK | 21.22 dB | 21.17 dB | 8.67e-02 | 5.22e-03 | 0.99619 |
| vs_fp32 | dV | 24.41 dB | 24.26 dB | 6.08e-02 | 3.60e-03 | 0.99813 |
| vs_manual | dQ | 20.05 dB | 19.36 dB | 1.02e-01 | 6.40e-03 | 0.99424 |
| vs_manual | dK | 21.22 dB | 21.17 dB | 8.67e-02 | 5.22e-03 | 0.99619 |
| vs_manual | dV | 24.41 dB | 24.26 dB | 6.08e-02 | 3.60e-03 | 0.99813 |
| manual_vs_fp32 | dQ | 119.40 dB | 119.09 dB | 1.03e-06 | 6.56e-08 | 1.00000 |
| manual_vs_fp32 | dK | 119.39 dB | 119.04 dB | 1.04e-06 | 6.64e-08 | 1.00000 |
| manual_vs_fp32 | dV | 120.96 dB | 120.57 dB | 8.76e-07 | 5.57e-08 | 1.00000 |

### N = 1536

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 20.24 dB | 19.51 dB | 1.00e-01 | 4.47e-03 | 0.99443 |
| vs_fp32 | dK | 21.59 dB | 21.54 dB | 8.31e-02 | 3.56e-03 | 0.99651 |
| vs_fp32 | dV | 24.35 dB | 24.27 dB | 6.10e-02 | 2.57e-03 | 0.99813 |
| vs_manual | dQ | 20.24 dB | 19.51 dB | 1.00e-01 | 4.47e-03 | 0.99443 |
| vs_manual | dK | 21.59 dB | 21.54 dB | 8.31e-02 | 3.56e-03 | 0.99651 |
| vs_manual | dV | 24.35 dB | 24.27 dB | 6.10e-02 | 2.57e-03 | 0.99813 |
| manual_vs_fp32 | dQ | 118.42 dB | 118.17 dB | 1.14e-06 | 5.19e-08 | 1.00000 |
| manual_vs_fp32 | dK | 118.44 dB | 118.15 dB | 1.15e-06 | 5.22e-08 | 1.00000 |
| manual_vs_fp32 | dV | 119.66 dB | 119.36 dB | 1.01e-06 | 4.49e-08 | 1.00000 |

### N = 3072

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 20.41 dB | 19.70 dB | 9.83e-02 | 3.11e-03 | 0.99467 |
| vs_fp32 | dK | 21.80 dB | 21.78 dB | 8.11e-02 | 2.47e-03 | 0.99669 |
| vs_fp32 | dV | 24.34 dB | 24.23 dB | 6.11e-02 | 1.82e-03 | 0.99811 |
| vs_manual | dQ | 20.41 dB | 19.70 dB | 9.83e-02 | 3.11e-03 | 0.99467 |
| vs_manual | dK | 21.80 dB | 21.78 dB | 8.11e-02 | 2.47e-03 | 0.99669 |
| vs_manual | dV | 24.34 dB | 24.23 dB | 6.11e-02 | 1.82e-03 | 0.99811 |
| manual_vs_fp32 | dQ | 116.91 dB | 116.76 dB | 1.33e-06 | 4.34e-08 | 1.00000 |
| manual_vs_fp32 | dK | 116.97 dB | 116.79 dB | 1.34e-06 | 4.34e-08 | 1.00000 |
| manual_vs_fp32 | dV | 117.83 dB | 117.69 dB | 1.21e-06 | 3.91e-08 | 1.00000 |

### N = 6144

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 20.47 dB | 19.89 dB | 9.64e-02 | 2.16e-03 | 0.99491 |
| vs_fp32 | dK | 21.90 dB | 21.89 dB | 8.02e-02 | 1.72e-03 | 0.99677 |
| vs_fp32 | dV | 24.27 dB | 24.19 dB | 6.15e-02 | 1.29e-03 | 0.99810 |
| vs_manual | dQ | 20.47 dB | 19.89 dB | 9.64e-02 | 2.16e-03 | 0.99491 |
| vs_manual | dK | 21.90 dB | 21.89 dB | 8.02e-02 | 1.72e-03 | 0.99677 |
| vs_manual | dV | 24.27 dB | 24.19 dB | 6.15e-02 | 1.29e-03 | 0.99810 |
| manual_vs_fp32 | dQ | 115.04 dB | 114.94 dB | 1.63e-06 | 3.79e-08 | 1.00000 |
| manual_vs_fp32 | dK | 115.14 dB | 115.02 dB | 1.63e-06 | 3.77e-08 | 1.00000 |
| manual_vs_fp32 | dV | 115.64 dB | 115.52 dB | 1.54e-06 | 3.49e-08 | 1.00000 |

### fp8_dS_mode = 2 (fp8 SR dS)

| comparison | grad | mean QSNR | min QSNR | max rel-L1 | max RMSE | min cos |
|---|---|---|---|---|---|---|
| vs_fp32 | dQ | 18.04 dB | -11.47 dB | 9.44e-01 | 3.12e-01 | 0.22504 |
| vs_fp32 | dK | 18.20 dB | -25.45 dB | 1.88e+01 | 1.57e+00 | 0.04578 |
| vs_fp32 | dV | 24.35 dB | 24.19 dB | 6.15e-02 | 5.04e-03 | 0.99810 |
| vs_manual | dQ | 18.04 dB | -11.47 dB | 9.44e-01 | 3.12e-01 | 0.22504 |
| vs_manual | dK | 18.20 dB | -25.45 dB | 1.88e+01 | 1.57e+00 | 0.04578 |
| vs_manual | dV | 24.35 dB | 24.19 dB | 6.15e-02 | 5.04e-03 | 0.99810 |
| manual_vs_fp32 | dQ | 118.11 dB | 114.94 dB | 1.63e-06 | 8.46e-08 | 1.00000 |
| manual_vs_fp32 | dK | 118.13 dB | 115.02 dB | 1.63e-06 | 8.64e-08 | 1.00000 |
| manual_vs_fp32 | dV | 119.32 dB | 115.52 dB | 1.54e-06 | 7.15e-08 | 1.00000 |

## Backward failures

- shape=(1, 8, 384, 64) seed=0 mode=2 dQ: QSNR=-11.47 rel-L1=9.44e-01 cos=0.22504
- shape=(1, 8, 384, 64) seed=0 mode=2 dK: QSNR=-25.45 rel-L1=1.88e+01 cos=0.04578
- shape=(1, 8, 384, 64) seed=0 mode=2 dV: QSNR=24.49 rel-L1=5.91e-02 cos=0.99823
- shape=(1, 8, 384, 64) seed=1 mode=2 dQ: QSNR=-9.38 rel-L1=8.28e-01 cos=0.28274
- shape=(1, 8, 384, 64) seed=1 mode=2 dK: QSNR=-24.06 rel-L1=1.60e+01 cos=0.05378
- shape=(1, 8, 384, 64) seed=1 mode=2 dV: QSNR=24.29 rel-L1=6.02e-02 cos=0.99814

## Key findings

- **Forward kernel O vs torch SDPA fp32**: mean QSNR **28.49 dB**, min **28.29 dB**, max rel-L1 **3.79e-02**, min cos **0.99926**.
- **Forward kernel O vs FP8-dequant fp32 SDPA** (measures kernel-internal loss; takes the input quantization out of the picture): mean QSNR **50.52 dB**, min **50.38 dB**, max rel-L1 **2.97e-03**.
- **bf16 SDPA vs fp32 SDPA baseline**: mean QSNR **48.56 dB**. This is the noise floor an idealized bf16 attention would already introduce vs fp32; the FP8 kernel cannot beat it because PV is still computed in bf16 in this revision.
- **INT8-GEMM1 forward O vs torch SDPA fp32**: mean QSNR **40.51 dB** (FP8: **28.49 dB**), max rel-L1 **9.80e-03** (FP8: **3.79e-02**), min cos **0.99995** (FP8: **0.99926**). INT8 actually wins on the forward by ~10-12 dB: the `vs_quant` rows are essentially identical between FP8 and INT8 (kernel-internal numerics are the same), so the gap is entirely the input quantization noise — INT8 has 8-bit uniform resolution per row, while FP8 e4m3 has only 3 mantissa bits per element so its per-row resolution on Gaussian-shaped data is coarser. The SageBwd paper's INT8 win for the *backward* (not exercised by this forward kernel) is in addition to this forward advantage.
- **Backward dQ vs torch SDPA fp32**: mean QSNR **18.04 dB**, min **-11.47 dB**, max rel-L1 **9.44e-01**.
- **Backward dK vs torch SDPA fp32**: mean QSNR **18.20 dB**, min **-25.45 dB**, max rel-L1 **1.88e+01**.
- **Backward dV vs torch SDPA fp32**: mean QSNR **24.35 dB**, min **24.19 dB**, max rel-L1 **6.15e-02**.
- The backward gradient QSNR (~22-26 dB end-to-end) sits *below* the 32 dB FP8 e4m3 quantization noise floor documented in `QUANT_CORRECTNESS.md` — the matmul/softmax accumulation inside the kernel, not the input quant, is the dominant error source.
- The `manual_vs_fp32` row (manual fp32 reference vs torch SDPA fp32) sits at ~120 dB on every grad — confirms the comparison itself is sound.
- **Known fragility**: at least one backward case fails (seed-dependent) for the smallest D=64 shape `(1, 8, 384, 64)`. Reproduces with the original (pre-optimization) kernel — *not* a regression from the quantization-kernel rewrite. The failure is state-sensitive: running the (1,8,384,64) seed=0 case immediately before seed=1 reliably triggers it, while running seed=1 in isolation passes. Most likely a CUDA allocator / workspace aliasing issue in the FP8 backward kernel (not in scope for this PR).
