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

PyTorch SDPA backward is reported as `bwd-only`: the forward graph is built
before timing, then the measured function runs `torch.autograd.grad` against the
retained output. This keeps the SDPA backward number comparable to
`fp8_attention_backward`, which also receives saved forward outputs.

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
fp8 attention fwd             0.0671 ms   144.02 TFLOP/s
fp8 attention bwd             1.0116 ms    23.88 TFLOP/s

torch sdpa float32 fwd        0.3416 ms    28.29 TFLOP/s
torch sdpa float32 bwd-only   1.0599 ms    22.79 TFLOP/s
torch sdpa bfloat16 fwd       0.0295 ms   327.35 TFLOP/s
torch sdpa bfloat16 bwd-only  0.1642 ms   147.16 TFLOP/s
```

### B=1 H=8 N=3072 D=128

```text
benchmark protocol: uniform[-1,1], groups=4, warmup=500, iters=100, cooldown=0.20s
quant Q token                 0.0244 ms     647.4 GB/s
quant K token                 0.0243 ms     651.4 GB/s
quant V channel               0.1076 ms     146.3 GB/s
fp8 attention fwd             0.2334 ms   165.64 TFLOP/s
fp8 attention bwd             3.8822 ms    24.89 TFLOP/s

torch sdpa float32 fwd        1.3131 ms    29.44 TFLOP/s
torch sdpa float32 bwd-only   4.0305 ms    23.98 TFLOP/s
torch sdpa bfloat16 fwd       0.1076 ms   359.10 TFLOP/s
torch sdpa bfloat16 bwd-only  0.3278 ms   294.80 TFLOP/s
```

### B=2 H=16 N=3072 D=128

```text
benchmark protocol: uniform[-1,1], groups=1, warmup=500, iters=100, cooldown=0.20s
quant Q token                 0.0859 ms     737.4 GB/s
quant K token                 0.0857 ms     738.5 GB/s
quant V channel               0.4012 ms     156.9 GB/s
fp8 attention fwd             0.6048 ms   255.67 TFLOP/s
fp8 attention bwd            13.5716 ms    28.48 TFLOP/s

torch sdpa float32 fwd        4.5608 ms    33.90 TFLOP/s
torch sdpa float32 bwd-only  13.9994 ms    27.61 TFLOP/s
torch sdpa bfloat16 fwd       0.4033 ms   383.38 TFLOP/s
torch sdpa bfloat16 bwd-only  1.1819 ms   327.06 TFLOP/s
```

## Reading The Results

The FP8 forward kernel improves as the workload grows:

```text
B=1 H=8  N=1536 D=128: 144.02 TFLOP/s
B=1 H=8  N=3072 D=128: 165.64 TFLOP/s
B=2 H=16 N=3072 D=128: 255.67 TFLOP/s
```

It is faster than fp32 SDPA for these shapes. PyTorch bf16 SDPA is still
faster, especially for backward.

## Current Slowdown Diagnosis

The current forward kernel is not fully FP8. It computes `QK^T` with FP8 WGMMA,
then casts the softmax tile to bf16 and computes `PV` with bf16 WGMMA. That
means only about half of the attention matmul work can benefit from FP8 Tensor
Cores, while the kernel still pays extra scale-load and scale-multiply overhead.

The backward kernel has a larger structural gap. It uses FP8 WGMMA for several
matmuls, but `dQ` still uses a bf16 shadow `K` path because the current TK FP8
primitive/layout only exposes the needed `AB^T` form. The D=128 backward
instantiations also spill registers heavily at compile time, which is consistent
with the measured low backward throughput.

Short clean check on H100 PCIe, seed `0`, `B=2 H=16 N=3072 D=128`,
default protocol:

```text
fp8 attention fwd             0.6048 ms   255.67 TFLOP/s
fp8 attention bwd            13.5716 ms    28.48 TFLOP/s
torch sdpa bfloat16 fwd       0.4033 ms   383.38 TFLOP/s
torch sdpa bfloat16 bwd-only  1.1819 ms   327.06 TFLOP/s
```

So the current implementation is not slow because of a small or unsaturated
shape alone. The next meaningful performance work is to make `PV` low-bit in
forward, remove the bf16 `dQ` fallback in backward, and reduce D=128 register
pressure before expecting speedups over PyTorch bf16 SDPA.
