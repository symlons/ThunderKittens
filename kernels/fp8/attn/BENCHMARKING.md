# FP8 Attention Benchmarking

This directory uses `profile_fp8.py` for reproducible kernel profiling. The
script reports:

- quantization kernels as effective bandwidth in GB/s
- FP8 attention forward/backward as TFLOP/s
- PyTorch SDPA fp32 and bf16 baselines
- FP8 speedups against those baselines

## Protocol

The default protocol follows the local benchmarking convention:

- uniform `[-1, 1]` float32 inputs
- automatic input groups sized to exceed `3x` L2 cache when needed
- 500 warmup launches
- 100 measured launches
- two CUDA events around the full measured launch loop
- no synchronization between measured launches
- short cooldown between benchmarked kernels

The defaults can be overridden:

```bash
python3 profile_fp8.py \
  --B 1 --H 8 --N 1536 --D 128 --seed 0 \
  --bench-warmup 500 --bench-iters 100
```

Use `--bench-groups` to force a specific number of input groups, and
`--no-sdpa` to skip PyTorch baselines.

## Reproducing The Current Sweep

```bash
bash profile_fp8_runs.sh
```

The script runs:

```text
B=1 H=8  N=1536 D=128
B=1 H=8  N=3072 D=128
B=2 H=16 N=3072 D=128
```

## Reference Results

Measured on `NVIDIA H100 PCIe`, seed `0`, with the default protocol.

### B=1 H=8 N=1536 D=128

```text
benchmark protocol: uniform[-1,1], groups=7, warmup=500, iters=100, cooldown=0.20s
quant Q token                 0.0140 ms     564.8 GB/s
quant K token                 0.0159 ms     498.2 GB/s
quant V channel               0.0547 ms     143.7 GB/s
fp8 attention fwd             0.0861 ms   112.21 TFLOP/s
fp8 attention bwd             1.0668 ms    22.65 TFLOP/s

torch sdpa float32 fwd        0.3419 ms    28.26 TFLOP/s
torch sdpa float32 bwd        1.3859 ms    17.43 TFLOP/s
torch sdpa bfloat16 fwd       0.0329 ms   293.62 TFLOP/s
torch sdpa bfloat16 bwd       0.4234 ms    57.06 TFLOP/s
```

### B=1 H=8 N=3072 D=128

```text
benchmark protocol: uniform[-1,1], groups=4, warmup=500, iters=100, cooldown=0.20s
quant Q token                 0.0242 ms     653.0 GB/s
quant K token                 0.0241 ms     655.9 GB/s
quant V channel               0.1089 ms     144.4 GB/s
fp8 attention fwd             0.2428 ms   159.20 TFLOP/s
fp8 attention bwd             3.9268 ms    24.61 TFLOP/s

torch sdpa float32 fwd        1.3127 ms    29.45 TFLOP/s
torch sdpa float32 bwd        5.3426 ms    18.09 TFLOP/s
torch sdpa bfloat16 fwd       0.1078 ms   358.54 TFLOP/s
torch sdpa bfloat16 bwd       0.4325 ms   223.44 TFLOP/s
```

### B=2 H=16 N=3072 D=128

```text
benchmark protocol: uniform[-1,1], groups=1, warmup=500, iters=100, cooldown=0.20s
quant Q token                 0.0857 ms     738.5 GB/s
quant K token                 0.0856 ms     739.8 GB/s
quant V channel               0.4014 ms     156.8 GB/s
fp8 attention fwd             0.6513 ms   237.41 TFLOP/s
fp8 attention bwd            13.6146 ms    28.39 TFLOP/s

torch sdpa float32 fwd        4.5538 ms    33.95 TFLOP/s
torch sdpa float32 bwd       18.5930 ms    20.79 TFLOP/s
torch sdpa bfloat16 fwd       0.4024 ms   384.21 TFLOP/s
torch sdpa bfloat16 bwd       1.6542 ms   233.68 TFLOP/s
```

## Reading The Results

The FP8 forward kernel improves as the workload grows:

```text
B=1 H=8  N=1536 D=128: 112.21 TFLOP/s
B=1 H=8  N=3072 D=128: 159.20 TFLOP/s
B=2 H=16 N=3072 D=128: 237.41 TFLOP/s
```

It is faster than fp32 SDPA for these shapes. PyTorch bf16 SDPA is still
faster, especially for backward.

