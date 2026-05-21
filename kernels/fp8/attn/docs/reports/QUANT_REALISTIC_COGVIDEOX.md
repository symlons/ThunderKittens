# FP8 vs INT8: heavy-tail and real-model quant noise

- Device: **NVIDIA H200** (SM 9.0, 132 SMs)
- Per-token symmetric, scale = `max(amax / max_int, 1e-12)`
- Shape used for synthetic: B=1, H=8, N=1536, D=128 (12288 rows of D=128)

## Synthetic distributions

Mean QSNR / rel-L1 / RMSE / cos over the listed seeds. `amax/σ` is the row-aggregate dynamic-range ratio that governs the FP8-vs-INT8 trade-off: small → INT8 wins, large → FP8 wins.

| distribution | FP8 QSNR | INT8 QSNR | Δ (INT8 − FP8) | winner | FP8 / INT8 rel-L1 | FP8 / INT8 RMSE | FP8 / INT8 cos | amax/σ (mean, p95) |
|---|---|---|---|---|---|---|---|---|
| gaussian | 31.80 dB | 43.79 dB | +11.99 dB | **INT8** | 2.19e-02 / 6.93e-03 | 2.57e-02 / 6.47e-03 | 0.99967 / 0.99998 | 2.83 / 3.48 |
| laplace | 32.11 dB | 40.98 dB | +8.88 dB | **INT8** | 2.16e-02 / 1.06e-02 | 3.51e-02 / 1.26e-02 | 0.99969 / 0.99996 | 3.84 / 5.16 |
| student_t_df3 | 33.02 dB | 37.27 dB | +4.25 dB | **INT8** | 2.11e-02 / 1.54e-02 | 3.84e-02 / 2.36e-02 | 0.99975 / 0.99991 | 4.96 / 8.02 |
| student_t_df5 | 32.19 dB | 40.41 dB | +8.22 dB | **INT8** | 2.16e-02 / 1.05e-02 | 3.17e-02 / 1.23e-02 | 0.99970 / 0.99995 | 3.93 / 5.82 |
| log_normal | 33.71 dB | 35.93 dB | +2.22 dB | **INT8** | 2.07e-02 / 2.03e-02 | 4.46e-02 / 3.46e-02 | 0.99979 / 0.99987 | 6.03 / 9.03 |
| channel_outlier_x8 | 33.37 dB | 36.46 dB | +3.09 dB | **INT8** | 2.11e-02 / 1.58e-02 | 2.62e-02 / 1.84e-02 | 0.99977 / 0.99989 | 5.27 / 9.23 |
| channel_outlier_x32 | 41.15 dB | 32.33 dB | -8.81 dB | **FP8** | 1.80e-02 / 5.06e-02 | 2.63e-02 / 7.27e-02 | 0.99996 / 0.99971 | 8.82 / 11.14 |
| mix_outlier_5pct_x10 | 33.83 dB | 35.84 dB | +2.01 dB | **INT8** | 2.00e-02 / 2.76e-02 | 4.95e-02 / 3.93e-02 | 0.99979 / 0.99987 | 6.90 / 9.08 |
| mix_outlier_1pct_x50 | 38.28 dB | 32.90 dB | -5.38 dB | **FP8** | 1.70e-02 / 6.14e-02 | 6.19e-02 / 1.15e-01 | 0.99993 / 0.99974 | 7.68 / 11.20 |

## Real CogVideoX transformer (MMDiT) Q/K/V activations

Model: `THUDM/CogVideoX-2b`. Captured by forward-hooking every `Attention.to_q/to_k/to_v` projection during one forward pass with random latents and a random text embedding, then reshaped to (B, heads, N, head_dim). Activations are upcast to fp32 before quantization.

| layer | role | self/cross | shape | FP8 QSNR | INT8 QSNR | Δ | winner | FP8 / INT8 rel-L1 | FP8 / INT8 RMSE | FP8 / INT8 cos | amax/σ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| transformer_blocks.0.attn1 | q | self_attn | (1, 30, 556, 64) | 32.17 dB | 43.45 dB | +11.27 dB | **INT8** | 2.12e-02 / 7.36e-03 | 4.29e-02 / 1.17e-02 | 0.99970 / 0.99998 | 2.94 |
| transformer_blocks.0.attn1 | k | self_attn | (1, 30, 556, 64) | 32.11 dB | 43.89 dB | +11.78 dB | **INT8** | 2.13e-02 / 6.90e-03 | 5.73e-02 / 1.47e-02 | 0.99969 / 0.99998 | 2.76 |
| transformer_blocks.0.attn1 | v | self_attn | (1, 30, 556, 64) | 32.04 dB | 44.40 dB | +12.36 dB | **INT8** | 2.14e-02 / 6.43e-03 | 3.35e-02 / 8.07e-03 | 0.99969 / 0.99998 | 2.65 |
| transformer_blocks.5.attn1 | q | self_attn | (1, 30, 556, 64) | 32.18 dB | 43.46 dB | +11.28 dB | **INT8** | 2.12e-02 / 7.18e-03 | 3.15e-02 / 8.59e-03 | 0.99970 / 0.99998 | 2.88 |
| transformer_blocks.5.attn1 | k | self_attn | (1, 30, 556, 64) | 32.05 dB | 44.38 dB | +12.33 dB | **INT8** | 2.14e-02 / 6.42e-03 | 3.32e-02 / 8.04e-03 | 0.99969 / 0.99998 | 2.62 |
| transformer_blocks.5.attn1 | v | self_attn | (1, 30, 556, 64) | 32.00 dB | 44.61 dB | +12.61 dB | **INT8** | 2.15e-02 / 6.25e-03 | 2.47e-02 / 5.79e-03 | 0.99968 / 0.99998 | 2.59 |
| transformer_blocks.10.attn1 | q | self_attn | (1, 30, 556, 64) | 32.33 dB | 42.70 dB | +10.37 dB | **INT8** | 2.11e-02 / 7.88e-03 | 4.92e-02 / 1.49e-02 | 0.99971 / 0.99997 | 3.09 |
| transformer_blocks.10.attn1 | k | self_attn | (1, 30, 556, 64) | 32.04 dB | 44.37 dB | +12.33 dB | **INT8** | 2.14e-02 / 6.41e-03 | 5.89e-02 / 1.42e-02 | 0.99969 / 0.99998 | 2.62 |
| transformer_blocks.10.attn1 | v | self_attn | (1, 30, 556, 64) | 32.02 dB | 44.52 dB | +12.50 dB | **INT8** | 2.14e-02 / 6.32e-03 | 2.79e-02 / 6.61e-03 | 0.99969 / 0.99998 | 2.61 |
| transformer_blocks.15.attn1 | q | self_attn | (1, 30, 556, 64) | 32.30 dB | 42.74 dB | +10.43 dB | **INT8** | 2.11e-02 / 7.77e-03 | 8.37e-02 / 2.52e-02 | 0.99971 / 0.99997 | 3.07 |
| transformer_blocks.15.attn1 | k | self_attn | (1, 30, 556, 64) | 32.00 dB | 44.68 dB | +12.67 dB | **INT8** | 2.15e-02 / 6.13e-03 | 1.08e-01 / 2.52e-02 | 0.99968 / 0.99998 | 2.54 |
| transformer_blocks.15.attn1 | v | self_attn | (1, 30, 556, 64) | 32.02 dB | 44.55 dB | +12.53 dB | **INT8** | 2.14e-02 / 6.30e-03 | 5.24e-02 / 1.24e-02 | 0.99969 / 0.99998 | 2.61 |
| transformer_blocks.20.attn1 | q | self_attn | (1, 30, 556, 64) | 32.34 dB | 42.67 dB | +10.33 dB | **INT8** | 2.11e-02 / 7.98e-03 | 9.79e-02 / 2.98e-02 | 0.99971 / 0.99997 | 3.16 |
| transformer_blocks.20.attn1 | k | self_attn | (1, 30, 556, 64) | 32.02 dB | 44.60 dB | +12.58 dB | **INT8** | 2.14e-02 / 6.19e-03 | 1.23e-01 / 2.88e-02 | 0.99969 / 0.99998 | 2.55 |
| transformer_blocks.20.attn1 | v | self_attn | (1, 30, 556, 64) | 32.03 dB | 44.58 dB | +12.55 dB | **INT8** | 2.14e-02 / 6.29e-03 | 5.84e-02 / 1.38e-02 | 0.99969 / 0.99998 | 2.61 |
| transformer_blocks.25.attn1 | q | self_attn | (1, 30, 556, 64) | 32.01 dB | 44.93 dB | +12.93 dB | **INT8** | 2.15e-02 / 5.82e-03 | 1.48e-01 / 3.35e-02 | 0.99969 / 0.99998 | 2.51 |
| transformer_blocks.25.attn1 | k | self_attn | (1, 30, 556, 64) | 32.10 dB | 44.27 dB | +12.17 dB | **INT8** | 2.13e-02 / 6.52e-03 | 2.57e-01 / 6.33e-02 | 0.99969 / 0.99998 | 2.74 |
| transformer_blocks.25.attn1 | v | self_attn | (1, 30, 556, 64) | 32.05 dB | 44.42 dB | +12.36 dB | **INT8** | 2.14e-02 / 6.41e-03 | 5.94e-02 / 1.43e-02 | 0.99969 / 0.99998 | 2.63 |

### Aggregate (mean over captured layers)

| role | self/cross | n | mean FP8 QSNR | mean INT8 QSNR | Δ | winner |
|---|---|---|---|---|---|---|
| k | self_attn | 6 | 32.05 dB | 44.37 dB | +12.31 dB | **INT8** |
| q | self_attn | 6 | 32.22 dB | 43.33 dB | +11.10 dB | **INT8** |
| v | self_attn | 6 | 32.03 dB | 44.51 dB | +12.49 dB | **INT8** |

## Key findings

- **FP8 wins** on heavy-tail / outlier-driven distributions: `channel_outlier_x32`, `mix_outlier_1pct_x50`.
- **INT8 wins** on bounded-tail distributions: `gaussian`, `laplace`, `student_t_df3`, `student_t_df5`, `log_normal`, `channel_outlier_x8`, `mix_outlier_5pct_x10`.
- The crossover is well predicted by per-row `amax/σ`: FP8's constant relative precision pays off once that ratio is large enough that uniform INT8 has to spend its budget on the outliers.
