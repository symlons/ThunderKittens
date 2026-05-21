# FP8 Attention Kernels

This directory contains the FP8 attention CUDA/PyTorch extension, correctness harnesses, and profiling tools.

## Documentation

- [docs/TESTING.md](docs/TESTING.md): build steps, correctness sweeps, smoke tests, and shared test modules.
- [docs/BENCHMARKING.md](docs/BENCHMARKING.md): profiling commands, benchmarking protocol, unified long-context/backward profiling modes, and performance interpretation.
- [docs/reports/](docs/reports/): generated correctness and quantization reports.

## Common Commands

```bash
make BUILD_MODE=torch KERNEL=fp8
python3 correctness_attn.py --quick
python3 profile_fp8.py --B 1 --H 8 --N 1536 --D 128
```
