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

## Quantization Kernel Profiling

`profile_quant.py` sweeps the standalone FP8 quantization kernels
(`fp8_quantize_per_token_out`, `fp8_quantize_per_channel_out`) across batch
and sequence length, reporting per-launch ms, GB/s (logical traffic =
read 4 B/elem fp32 + write 1 B/elem fp8 + scale writes), and elements/s.

```bash
python3 profile_quant.py --B 4 8 16 --H 16 \
  --N 2048 4096 8192 16384 --D 128 \
  --bench-iters 50 --bench-warmup 30
```

### Optimized kernel numbers (NVIDIA H200, HBM peak ≈ 4.48 TB/s)

```text
B=4   H=16  N=8192   D=128 | per-token:   0.089 ms  3533 GB/s | per-channel:  0.213 ms  1470 GB/s
B=8   H=16  N=8192   D=128 | per-token:   0.174 ms  3611 GB/s | per-channel:  0.400 ms  1561 GB/s
B=16  H=16  N=16384  D=128 | per-token:   0.686 ms  3668 GB/s | per-channel:  1.530 ms  1634 GB/s
```

- **Per-token** reaches ~82% of measured HBM peak (single read+write pass).
- **Per-channel** reaches ~36% as reported, but the kernel does two HBM
  passes (amax + quantize); the true HBM traffic per element is ~9 B
  (4 read in pass 1 + 4 read in pass 2 + 1 write), so the realized
  bandwidth is ≈ 1634 × 9/5 ≈ 2940 GB/s, i.e. ~66% of HBM peak. The
  second read of `x` is fundamentally required: the FP8 code for any
  element depends on `amax(x[:, d])` over all N rows, and N×D×4 B per
  slab (8 MB at N=16384, D=128) does not fit in shared memory or
  registers, so pass 2 must re-read from HBM.

For reference, the same kernels before optimization measured
≈ 955 GB/s (per-token) and ≈ 180 GB/s (per-channel) at the same
shapes — a 3.8× and 9.3× speedup respectively. The optimizations
were:

- **Per-token**: one warp per row (4 warps/block), `float4`/`float2`
  vectorized loads, warp-shuffle amax reduction (no shared memory),
  packed `fp8e4m3_4`/`fp8e4m3_2` 32/16-bit stores, IEEE-rounded
  division via `__fdiv_rn` to remain bit-exact under `--use_fast_math`.
- **Per-channel**: thread `d` owns column `d` within a `(B,H)` slab
  (fully coalesced reads/writes, previously stride-D scalar reads);
  N split into 256-row chunks reduced across blocks via float-bit
  `atomicMax` on the (non-negative) amax; a second kernel reads the
  final amax, computes the scale, and quantizes. The split-N grid
  keeps SMs saturated even when `B*H` is small.

## Raw HBM Bandwidth Reference

`hbm_bandwidth.cu` is a self-contained single-GPU HBM bandwidth
profiler used to put the FP8 kernel numbers in context. It reports
`cudaMemcpy D2D`, byte/uint4/PTX kernel copies, read-only,
write-only, and `cudaMemset`, each as GB/s and % of theoretical
peak (queried via `cudaDeviceGetAttribute`).

```bash
nvcc -O3 -arch=sm_90a -std=c++17 hbm_bandwidth.cu -o hbm_bandwidth
./hbm_bandwidth [device_id] [size_MB] [num_iters]
```

Representative result on `NVIDIA H200`:

```text
Device 0: NVIDIA H200 (SM 9.0, 132 SMs)
HBM peak (from device props): 4483.67 GB/s
  cudaMemcpy D2D                  3932.9 GB/s  (87.7% of peak)
  kernel copy (uint4)             3743.4 GB/s  (83.5% of peak)
  kernel read-only                4009.7 GB/s  (89.4% of peak)
  kernel write-only               4024.8 GB/s  (89.8% of peak)
  cudaMemset                      4196.3 GB/s  (93.6% of peak)
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
