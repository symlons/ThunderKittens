# DiT torch.compile Capacity Findings

Current target: regular PyTorch/torch.compile DiT-L training setup without TK/custom kernels, using the `timm` attention backend from `dit3d_e2e_bench.py`.

## Script

Use:

```bash
uvx modal run kernels/gemm/bf16_h100_custom/modal_adaln.py --gpu H100 --mode torch_compile_capacity_1024
uvx modal run kernels/gemm/bf16_h100_custom/modal_adaln.py --gpu H100 --mode torch_compile_capacity_4096
```

The standalone script is `torch_compile_dit_capacity.py`. It builds a fresh model per batch size, wraps it with `torch.compile`, runs a realistic training step, records peak allocated/reserved CUDA memory, and stops after the first OOM when requested.

## Modal Setup

- Modal Volume: `tk_kernels`
- Mount path: `/data`
- Cached artifacts path: `/data/adaln`
- The Modal runner builds CPU-side extension artifacts first and reuses cached `.so` files from the volume on GPU runs.
- The torch.compile capacity path disables TK/custom kernels in the model (`fused=False`, `fused_residual=False`, `tk_mlp=False`, fused input/output projections disabled). The cached artifacts may still be loaded by the generic runner, but they are not used by this baseline model.

## H100 80GB Results

GPU reported by Modal: NVIDIA H100 80GB HBM3, total memory 79.18 GiB.

### Sequence Length 1024

Spatial shape: `(8, 8, 16)`.

| Batch | Result | Peak Allocated | Peak Reserved | Notes |
| ---: | :--- | ---: | ---: | :--- |
| 64 | PASS | 55.62 GiB | 56.39 GiB | revised high-batch sweep |
| 80 | PASS | 69.29 GiB | 70.03 GiB | highest passing batch so far |
| 96 | OOM | 78.02 GiB | 78.15 GiB | failed allocating another 768 MiB |

Current H100 capacity for this setup at sequence length 1024: **batch 80**.

Earlier low-batch sanity checks also passed at B=1, 2, 4, 8, 16, 32, and 64. Those are no longer useful for capacity search on 80GB+ GPUs except as smoke tests.

### Sequence Length 4096

Spatial shape: `(16, 16, 16)`.

| Batch | Result | Peak Allocated | Peak Reserved |
| ---: | :--- | ---: | ---: |
| 8 | PASS | 28.29 GiB | 29.10 GiB |
| 12 | PASS | 41.98 GiB | 42.73 GiB |
| 16 | PASS | 55.56 GiB | 56.35 GiB |
| 20 | PASS | 69.22 GiB | 69.99 GiB |
| 24 | OOM | 77.99 GiB | 78.15 GiB |

Current H100 capacity for this setup at sequence length 4096: **batch 20**.

## Practical Notes

- For H100/H200 capacity sweeps, avoid tiny batches by default. Start near expected memory pressure and bracket the OOM boundary.
- Use `--second-step` only when steady-state post-compile memory matters; it roughly doubles runtime and is not needed for a first capacity boundary.
- Full torch.compile train-step compilation can take minutes for a new shape. Peak memory results are still the key signal for capacity.
