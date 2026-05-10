# SageAttention2 Smoothing Demo

PyTorch simulation for studying low-bit attention variants inspired by
SageAttention2. Used for accuracy research, not kernel benchmarking.

It compares Q/K dtypes (INT4, INT8, FP8 E4M3, FP8 E5M2), three smoothing
techniques (SageAttention2-style block-mean subtraction, channel-mean for V,
and SmoothQuant), four Q/K quantization granularities (per_tensor, per_block,
per_token, per_thread), and runs both forward and backward ablations.

## Quick Start

```bash
# Synthetic Q,K,V
python3 smooth.py --qk all --smooth all

# Real CogVideoX-2b Q,K,V (after first running the capture, see below)
python3 smooth.py --source cogvideox --n 1024
```

## Sweeps

```bash
# Per-tensor / per-block / per-token / per-thread granularity sweep
python3 smooth.py --source cogvideox --n 1024 --granularity-sweep

# SmoothQuant alpha sweep (defaults: 0.0..1.0 step 0.1, on INT4)
python3 smooth.py --source cogvideox --n 1024 --smoothquant-sweep \
    --smoothquant-qk int4

# Backward pass ablation
python3 smooth.py --source cogvideox --n 1024 --bwd
```

## Visualizations

```bash
# Per-tensor + smoothing plots, plus dtype x smoothing heatmap
python3 smooth.py --source cogvideox --n 1024 \
    --plots --ablation-plots \
    --plot-qk int4 --plot-dir smooth_plots_cogvideox
```

Plots produced (`--plot-dir`):
- `tensor_heatmaps.png` — |Q|, |K|, |V| before/after smoothing
- `channel_means.png` — channel-mean removal
- `quantized_value_hist_<dtype>.png` — quantized value distribution
- `reconstruction_qsnr_<dtype>.png` — per-channel QSNR before/after smoothing
- `reconstruction_qsnr_density_<dtype>.png` — QSNR density
- `granularity_qsnr_<dtype>.png` — per-channel QSNR across granularities
- `granularity_sweep.png` — bar chart of output QSNR across granularities × dtypes
- `smoothquant_sweep_<dtype>.png` — SmoothQuant α vs out_l1 / QSNR
- `backward_qsnr_heatmap.png` — dQ, dK, dV QSNR heatmap
- `ablation_heatmap.png` — dtype × smoothing out_l1 / QSNR heatmap

## Capturing Real Q/K/V from CogVideoX-2b

The capture script subclasses `CogVideoXAttnProcessor2_0`, snapshots
post-RoPE Q/K/V on a single transformer layer, runs one diffusion step,
and saves the tensors to `captures/cogvideox.pt`.

```bash
python3 capture_cogvideox.py --layer 0 --num-inference-steps 1 \
    --num-frames 9 --max-tokens 4096
```

The saved bundle is `{Q, K, V}` with shape `(B, H, N, D)` =
`(2, 30, 4096, 64)` for CogVideoX-2b. `make_inputs(source="cogvideox",
head_index=...)` picks one head and returns `(N, D)` tensors.

Use `--head-index` to switch heads:
```bash
python3 smooth.py --source cogvideox --head-index 7 --n 1024
```

## Interactive Menu

```bash
python3 smooth.py --interactive
```

The menu supports filtering by Q/K dtype, filtering by smoothing
combination, changing sort metrics, grouping results, and saving plots.

## Key Findings (CogVideoX-2b layer 0, head 0, N=1024)

| Recipe                               | out QSNR (dB) |
|---|---|
| INT4 per_thread, no smoothing        | 17.4 |
| INT4 per_thread, Q+K mean-sub        | 20.3 |
| INT4 per_token, Q+K mean-sub         | 22.3 |
| INT4 per_tensor, Q+K mean-sub        | 15.2 |
| INT4 SmoothQuant α=0.7               | 22.3 |
| INT8 per_thread, Q+K mean-sub        | 34.6 |
| FP8 E4M3 per_thread, Q+K mean-sub    | 33.6 |
| FP8 E5M2 per_thread, Q+K mean-sub    | 30.5 |

Backward pass (FP8 dO/dS, per-row):

| Recipe                          | dQ QSNR | dK QSNR | dV QSNR |
|---|---|---|---|
| INT4 Q+K+V smoothing            | 14.1 | 11.6 | 16.0 |
| INT8 Q+K+V smoothing            | 29.7 | 28.7 | 25.9 |
| FP8 E4M3 Q+K+V smoothing        | 28.1 | 28.5 | 25.6 |

## Files

- `smooth.py` — CLI and interactive entry point
- `smooth_core.py` — quantization, smoothing, attention forward/backward, ablation rows
- `smooth_report.py` — terminal tables, filtering, sorting, grouping
- `smooth_viz.py` — tensor visualization helpers and ablation plots
- `capture_cogvideox.py` — capture Q/K/V from CogVideoX-2b
