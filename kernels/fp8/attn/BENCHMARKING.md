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

The FP8 PyTorch extension wrappers must not synchronize internally. Earlier
measurements included `cudaStreamSynchronize(stream)` inside the FP8 forward
and backward wrappers, which made the measured FP8 launch loop synchronize on
every iteration while SDPA ran back-to-back between CUDA events. Those internal
synchronizations have been removed.

### B=1 H=8 N=1536 D=128

```text
benchmark protocol: uniform[-1,1], groups=7, warmup=500, iters=100, cooldown=0.20s
quant Q token                 0.0140 ms     564.8 GB/s
quant K token                 0.0141 ms     562.0 GB/s
quant V channel               0.0546 ms     144.2 GB/s
fp8 attention fwd             0.0656 ms   147.26 TFLOP/s
fp8 attention bwd             1.0184 ms    23.72 TFLOP/s

torch sdpa float32 fwd        0.3428 ms    28.19 TFLOP/s
torch sdpa float32 bwd        1.3995 ms    17.26 TFLOP/s
torch sdpa bfloat16 fwd       0.0372 ms   260.02 TFLOP/s
torch sdpa bfloat16 bwd       0.2723 ms    88.73 TFLOP/s
```

### B=1 H=8 N=3072 D=128

```text
benchmark protocol: uniform[-1,1], groups=4, warmup=500, iters=100, cooldown=0.20s
quant Q token                 0.0245 ms     646.9 GB/s
quant K token                 0.0243 ms     650.3 GB/s
quant V channel               0.1071 ms     147.0 GB/s
fp8 attention fwd             0.2326 ms   166.21 TFLOP/s
fp8 attention bwd             3.8826 ms    24.89 TFLOP/s

torch sdpa float32 fwd        1.3145 ms    29.41 TFLOP/s
torch sdpa float32 bwd        5.3453 ms    18.08 TFLOP/s
torch sdpa bfloat16 fwd       0.1075 ms   359.58 TFLOP/s
torch sdpa bfloat16 bwd       0.4365 ms   221.38 TFLOP/s
```

### B=2 H=16 N=3072 D=128

```text
benchmark protocol: uniform[-1,1], groups=1, warmup=500, iters=100, cooldown=0.20s
quant Q token                 0.0857 ms     738.4 GB/s
quant K token                 0.0856 ms     739.6 GB/s
quant V channel               0.4011 ms     156.9 GB/s
fp8 attention fwd             0.6070 ms   254.71 TFLOP/s
fp8 attention bwd            13.5789 ms    28.47 TFLOP/s

torch sdpa float32 fwd        4.5531 ms    33.96 TFLOP/s
torch sdpa float32 bwd       18.4964 ms    20.90 TFLOP/s
torch sdpa bfloat16 fwd       0.4034 ms   383.31 TFLOP/s
torch sdpa bfloat16 bwd       1.6827 ms   229.72 TFLOP/s
```

## Reading The Results

The FP8 forward kernel improves as the workload grows:

```text
B=1 H=8  N=1536 D=128: 147.26 TFLOP/s
B=1 H=8  N=3072 D=128: 166.21 TFLOP/s
B=2 H=16 N=3072 D=128: 254.71 TFLOP/s
```

It is faster than fp32 SDPA for these shapes. PyTorch bf16 SDPA is still
faster, especially for backward.
