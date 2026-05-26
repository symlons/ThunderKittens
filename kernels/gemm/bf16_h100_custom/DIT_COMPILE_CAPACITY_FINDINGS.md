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

## Speedup Measurement Scope

Measure custom-kernel speedups at three levels. Treat the highest level as the headline result, and use lower levels to explain where the time moved.

1. **E2E full training step**

   This is the primary number. Use the same realistic full-DiT training path used for capacity search: full model, forward, backward, representative batch/sequence shape, and `torch.compile` as the baseline. Compare:

   - regular eager PyTorch
   - regular `torch.compile` without custom/TK kernels
   - custom/TK-enabled model variants

   Report wall time per train step, peak allocated/reserved memory, batch size, token count, GPU, attention backend, and whether the first compile step is excluded. Use the maximum realistic batch for each sequence length rather than small synthetic batches.

2. **DiTBlock-level speedups**

   Measure the block in the same shape regime as E2E, using this logical scope:

   ```python
   shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
   x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
   x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
   ```

   Report speedups for:

   - pre-attention path: `norm1 + AdaLN modulation + qkv/input projection` where applicable
   - post-attention path: attention output projection plus gated residual where applicable
   - MLP path: `norm2 + AdaLN modulation + fc1 + GELU + fc2 + gated residual`

   Attention itself is not the current optimization target, so isolate it or keep the backend fixed when reporting block-level results. The block-level result should explain how much of the E2E gain is available before/after attention and inside the MLP.

3. **Individual fused kernels**

   Use individual kernel profiling to attribute wins and losses, not as the headline. Compare custom kernels against eager and `torch.compile` for:

   - `pre_qkv_ln_adaln_linear`
   - `mlp_fc1_ln_adaln_linear_gelu`
   - `post_linear_gated_residual`
   - `full_mlp_branch`

   Report forward+backward train-step time for each case, correctness deltas, and speedup versus `torch.compile`. Prefer capacity-relevant `B*T` sizes over small convenience shapes.

## Individual Kernel Profiling Shapes

Use explicit `BxT` shape pairs for custom-vs-compile kernel profiling so each sequence length is tested near the realistic full-DiT capacity regime instead of as a Cartesian product of small batches and token counts.

H100 80GB shape pairs:

- T64: B1024 and B1280
- T128: B512 and B640
- T1024: B64 and B80
- T4096: B16 and B20
- T16384: B4 and B5 for the long-context mode

H200 140GB shape pairs should start at roughly double the H100 batch sizes:

- T64: B2048 and B2560
- T128: B1024 and B1280
- T1024: B128 and B160
- T4096: B32 and B40
- T16384: B8 and B10 for the long-context mode

Use the `_h200` profiling modes for these shapes:

```bash
uvx modal run kernels/gemm/bf16_h100_custom/modal_adaln.py --gpu H200 --mode custom_vs_compile_h200
uvx modal run kernels/gemm/bf16_h100_custom/modal_adaln.py --gpu H200 --mode custom_vs_compile_large_batch_h200
uvx modal run kernels/gemm/bf16_h100_custom/modal_adaln.py --gpu H200 --mode custom_vs_compile_long_h200
```

The T64/T128 pairs keep the same rough total-token pressure as the measured T1024/T4096 capacity boundary. Smaller batch sizes at those short sequence lengths are not useful for judging realistic high-memory H100/H200 training regimes.
