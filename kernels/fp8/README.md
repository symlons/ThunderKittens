# SageAttention2 Smoothing Demo

This directory contains a PyTorch simulation for studying low-bit attention
variants inspired by SageAttention2. It is intended for accuracy experiments and
visual inspection, not for kernel benchmarking.

## Run All Ablations

```bash
python3 smooth.py --qk all --smooth all
```

## Generate Tensor Visualizations

```bash
python3 smooth.py --qk all --smooth all --plots --plot-qk int4 --plot-dir smooth_plots
```

The plots include tensor magnitude heatmaps, channel-mean artifacts,
quantized-value histograms for the selected Q/K dtype, and reconstruction QSNR
by channel and density before and after smoothing.

## Interactive Menu

```bash
python3 smooth.py --interactive
```

The menu supports filtering by Q/K dtype, filtering by smoothing combination,
changing sort metrics, grouping results, and saving plots.

## Files

- `smooth.py`: CLI and interactive entry point.
- `smooth_core.py`: quantization, smoothing, attention simulation, and ablation
  data generation.
- `smooth_report.py`: terminal tables, filtering, sorting, and grouping.
- `smooth_viz.py`: tensor visualization helpers.

Smooth K/V
