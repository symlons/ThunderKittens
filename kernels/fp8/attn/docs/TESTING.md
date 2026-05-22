# Testing index — FP8 attention kernels

All entrypoints live in this directory. Build the extension first:

```bash
make BUILD_MODE=torch KERNEL=fp8
```

Tests fall into three layers: **correctness sweeps** (markdown report + exit
code), **inline harnesses** (printed metrics, fail loudly on regression),
and **profilers** (covered in `BENCHMARKING.md`). All correctness paths
report **QSNR / cosine / RMSE / rel-L1** through the same
`fp8_suite.metrics.tensor_metrics` helper.

## Correctness sweeps (write a markdown report, exit non-zero on fail)

| script | what it does | output | typical runtime |
|---|---|---|---|
| `correctness_quant.py` | 8046-case sweep of FP8 per-token, FP8 per-channel, and INT8 per-token quant kernels vs PyTorch reference; FP8 + INT8 noise-floor comparison vs original fp32 | [QUANT_CORRECTNESS.md](reports/QUANT_CORRECTNESS.md) | ~20 s |
| `correctness_attn.py` | 28-case shape sweep × 2 seeds of FP8 forward, **INT8-GEMM1 forward** (kernel-level ablation), and FP8 backward; vs fp32 SDPA, bf16 SDPA, fp8-dequant SDPA, and manual fp32 ref | [ATTN_CORRECTNESS.md](reports/ATTN_CORRECTNESS.md) | ~3 min |
| `correctness_realistic.py` | FP8 vs INT8 noise on heavy-tail / outlier synthetic distributions + real Q/K/V activations captured from a pretrained diffusion model (SD or CogVideoX) | [QUANT_REALISTIC.md](reports/QUANT_REALISTIC.md), [QUANT_REALISTIC_COGVIDEOX.md](reports/QUANT_REALISTIC_COGVIDEOX.md) | ~1 min synthetic, +download for the model |

How to run:

```bash
# full quant sweep (8046 cases)
python3 correctness_quant.py

# full forward + INT8 + backward sweep
python3 correctness_attn.py
# quick subset (3 shapes × 2 seeds)
python3 correctness_attn.py --quick
# skip INT8 ablation
python3 correctness_attn.py --no-int8
# pick which dS rounding modes to sweep
python3 correctness_attn.py --fp8-dS-modes 0 1 2

# synthetic-only realistic-inputs sweep
python3 correctness_realistic.py --unet none

# add real Stable-Diffusion UNet activations (~330 MB cache)
python3 correctness_realistic.py --unet segmind/tiny-sd

# add real CogVideoX-2b transformer activations (~3.4 GB cache)
HF_HOME=/scratch python3 correctness_realistic.py --unet THUDM/CogVideoX-2b \
    --out docs/reports/QUANT_REALISTIC_COGVIDEOX.md
```


## Quantization Kernel Coverage

Implemented CUDA quantization APIs and their correctness coverage:

| API | dtype | scale granularity | output scale shape | tested by | profiled by |
|---|---|---|---|---|---|
| `fp8_quantize_per_token` | FP8 e4m3 | per token/row over `D` | `(B,H,N)` | `correctness_quant.py`, `test_fp8_bwd_kernel.py` | `profile_quant.py` via `_out` |
| `fp8_quantize_per_token_out` | FP8 e4m3 | per token/row over `D` | `(B,H,N)` | allocation-free wrapper used by profilers | `profile_quant.py` |
| `fp8_quantize_per_channel` | FP8 e4m3 | per channel over `N` | `(B,H,D)` | `correctness_quant.py`, `test_fp8_bwd_kernel.py` | `profile_quant.py` via `_out` |
| `fp8_quantize_per_channel_out` | FP8 e4m3 | per channel over `N` | `(B,H,D)` | allocation-free wrapper used by profilers | `profile_quant.py` |
| `int8_quantize_per_token` | INT8 symmetric | per token/row over `D` | `(B,H,N)` | `correctness_quant.py`, INT8 forward ablation | `profile_quant.py` via `_out` |
| `int8_quantize_per_token_out` | INT8 symmetric | per token/row over `D` | `(B,H,N)` | allocation-free wrapper used by profilers | `profile_quant.py` |

All CUDA quantization kernels currently require fp32 input shaped `(B,H,N,D)`
and support `D=64` or `D=128`. The correctness sweep compares CUDA outputs
against PyTorch references from `fp8_suite.quant`, checks scales, dequantized
values, quantized-code byte agreement, and rerun determinism.

Quantization-only commands:

```bash
# quick correctness subset for FP8 token, FP8 channel, and INT8 token
python3 correctness_quant.py --quick

# full correctness sweep and report
python3 correctness_quant.py

# final-reporting quantization profiling, including INT8 token
python3 profile_quant.py --B 4 8 16 --H 16 --N 2048 4096 8192 16384 --D 128

# quick profiling smoke run, not for final reported numbers
python3 profile_quant.py --B 4 --H 16 --N 4096 --D 128 --quick-profile
```

## Inline harnesses (print metrics, raise on regression)

These call the kernels directly with assertion-style bounds. Useful in CI
and during iteration. All four wrappers at the directory root are thin
compatibility shims into `fp8_suite/`:

| entrypoint | wraps | covers |
|---|---|---|
| `test_fp8.py`            | `fp8_suite.test_smoke`            | one-shot fwd correctness + dump of metrics |
| `test_fp8_extensive.py`  | `fp8_suite.test_forward`          | forward sweep over `fp8_suite.cases.forward_cases` |
| `test_fp8_bwd.py`        | `fp8_suite.test_backward_ref`     | backward sweep against the manual fp32 reference recipe |
| `test_fp8_bwd_kernel.py` | `fp8_suite.test_backward_kernel`  | backward sweep with the CUDA kernel; also validates FP8 and INT8 quant kernels against the PyTorch reference |

How to run:

```bash
python3 test_fp8.py                  # forward smoke
python3 test_fp8_extensive.py        # forward sweep
python3 test_fp8_bwd.py              # backward (reference recipe)
python3 test_fp8_bwd_kernel.py       # backward (CUDA kernel)
```

Each accepts `-h` for shape / seed overrides and (in the bwd kernel test)
flags to control quantization-mode ablations.

## Profilers

See `BENCHMARKING.md` for `profile_fp8.py`, `profile_quant.py`,
`profile_long_context.py`, `hbm_bandwidth`, and the `profile_fp8_runs.sh`
sweep driver. `profile_long_context.py` also contains the unified long-context,
backward dS-mode, and SDPA-backward-peak profiling modes.

## Shared building blocks (reused by every test above)

| module | purpose |
|---|---|
| `fp8_suite/metrics.py`         | `tensor_metrics` (QSNR/cos/RMSE/rel-L1), `fmt_forward`, `fmt_grad`, `check_grad_metrics` |
| `fp8_suite/quant.py`           | PyTorch reference quantizers — FP8 per-token, FP8 per-channel, **INT8 per-token** |
| `fp8_suite/kernel_api.py`      | thin wrappers around the C++ extension (`fp8_forward`, `int8_forward`, `cuda_quantize_per_token`, …) |
| `fp8_suite/recipe.py`          | `prepare_forward_inputs(..., quant_dtype="fp8"\|"int8")`, `prepare_backward_inputs(...)` |
| `fp8_suite/references.py`      | fp32 SDPA reference, FP8-dequant SDPA reference, manual fp32 backward |
| `fp8_suite/cases.py`           | shape/seed enumerators shared by the inline harnesses |
| `fp8_suite/attn_correctness.py`| `forward_metrics`, `forward_metrics_int8`, `backward_metrics` used by `correctness_attn.py` |
| `fp8_suite/quant_correctness.py`| `quant_kernel_metrics` and ablation case generator used by `correctness_quant.py` |
| `fp8_suite/realistic_inputs.py`| synthetic heavy-tail distributions + diffusers-UNet / CogVideoX Q/K/V capture used by `correctness_realistic.py` |
