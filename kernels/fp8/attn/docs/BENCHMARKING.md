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

Any benchmark intended for final reported numbers should use this protocol. Lower
warmup/iteration counts, skipped baselines, or timing-only descale shortcuts are
acceptable for smoke tests and rough iteration, but those runs should be labeled
quick or timing-only and should not be used as final public numbers.

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

## Unified Profiling Entrypoints

`profile_fp8.py` is the short/moderate-context default. `profile_long_context.py`
now owns the longer and more specialized profiling flows that used to live in
separate scratch scripts:

```bash
# Final-reporting long-context sweep. Uses 500 warmup / 100 measured launches.
python3 profile_long_context.py --mode long

# Quick long-context smoke/timing run, not for final reported numbers.
python3 profile_long_context.py --mode long --quick-profile

# Broad 1K-to-300K sequence sweep. This includes short, moderate, long,
# and near-300K shapes with B/H varied to keep the largest cases on 80 GB H100.
python3 profile_long_context.py --mode seq-sweep --quick-profile --skip-sdpa-bwd

# Final-reporting backward dS-mode sweep on a small subset.
# Modes: 0=bf16/off, 1=FP8 RTNE, 2=FP8 SR.
python3 profile_long_context.py --mode bwd-sweep --quick --bwd-modes 0 1 2 \
  --bwd-sdp-descale-mode estimate

# Long-N timing-only backward sweep. The constant descale avoids the O(N^2)
# Python dS range estimate; do not use it for correctness-quality or final numbers.
python3 profile_long_context.py --mode bwd-sweep \
  --bwd-sdp-descale-mode constant --skip-sdpa-bwd --quick-profile

# Final-reporting bf16 SDPA backward-only baseline peak sweep.
python3 profile_long_context.py --mode sdpa-bwd-peak

# Custom final-reporting shape subset for either long or bwd-sweep modes.
python3 profile_long_context.py --mode bwd-sweep \
  --shapes "1,8,1536,128;2,16,3072,128" --bwd-modes 2 \
  --bwd-sdp-descale-mode estimate
```

The old one-off scripts `profile_bwd_sweep.py`, `profile_bwd_extensive.py`,
`profile_oom_squeezed.py`, and `profile_sdpa_bwd_peak.py` have been folded into
these modes. The unified modes default to the final-reporting launch counts; pass
`--quick-profile` or smaller `--bench-warmup` / `--bench-iters` values only for quick local checks.

## Reproducing The Current Sweep

```bash
bash profile_fp8_runs.sh
```

The script runs the short / moderate context sweep through `profile_fp8.py`:

```text
B=1 H=8  N=1536 D=128
B=1 H=8  N=3072 D=128
B=2 H=16 N=3072 D=128
```

…followed by the long-context sweep through `profile_long_context.py`
(N ∈ {57600, 71808, 144000, 162048, 323712}, B ∈ {1, 2, 4, 8}).

For the current broad forward sweep from roughly 1K through 300K tokens, use:

```bash
python3 profile_long_context.py --mode seq-sweep
```

For Modal H100 timing smoke runs, use:

```bash
uvx modal run kernels/fp8/attn/modal_fp8_attn_h100.py --seq-sweep --continue-on-error
```

`profile_long_context.py` runs FP8 fwd-only above `--bwd-threshold` (default
32000) because the backward recipe builds an O(N²) reference attention
matrix that OOMs at long N. In fwd-only mode it now also **skips the bf16
SDPA backward**, which previously dominated the wall-clock time of the
sweep (700 ms – 1.1 s per launch × 35 launches × ~15 shapes ≈ 8+ minutes
spent on a baseline we can't even compare against). The FP8-vs-bf16 fwd
comparison is unchanged.

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

## Long-Context Sweep Summary (H200)

`profile_long_context.py` over N ∈ {57600, 71808, 144000, 162048, 323712}
and B ∈ {1, 2, 4, 8} on `NVIDIA H200`, seed `0`, using the long-context
measurement setting noted at the time (warmup=20, iters=15). Treat those as
historical sweep results; final new reports should use the protocol above
(warmup=500, iters=100). The script now queries HBM peak from the device at
runtime — on the measured machine that is **4814 GB/s** (H200, 5.0 TB/s
nominal). FP8 dense SOL on H200 is ~1979 TFLOP/s and BF16 dense is
~989 TFLOP/s.

### Best FP8 attention forward shapes (measured)

| Shape (B,H,N)   | FP8 fwd TFLOP/s | bf16 SDPA fwd TFLOP/s | Speedup |
| --------------- | --------------: | --------------------: | ------: |
| (8, 1,  71808)  |         **413** |                   354 |   1.17× |
| (4, 2,  71808)  |             412 |                   356 |   1.16× |
| (4, 4,  57600)  |             413 |                   353 |   1.17× |
| (8, 2,  57600)  |             412 |                   353 |   1.17× |
| (2, 4,  57600)  |             408 |                   345 | **1.18×** |
| (2, 1, 162048)  |             408 |                   344 | **1.19×** |
| (2, 2, 144000)  |             406 |                   348 |   1.17× |
| (4, 1, 144000)  |             406 |                   347 |   1.17× |
| (4, 1, 162048)  |             403 |                   352 |   1.15× |
| (8, 1, 144000)  |             403 |                   353 |   1.14× |
| (2, 2,  71808)  |             401 |                   346 |   1.16× |
| (1, 2, 144000)  |             401 |                   343 |   1.17× |
| (1, 2,  71808)  |             397 |                   343 |   1.16× |
| (1, 1, 323712)  |             397 |                   345 |   1.15× |
| (1, 4,  57600)  |             390 |                   357 |   1.09× |
| (2, 1, 323712)  |             390 |                   351 |   1.11× |
| (1, 1, 162048)  |             389 |                   341 |   1.14× |
| (4, 1, 323712)  |             387 |                   354 |   1.09× |
| (8, 1, 162048)  |             402 |                   355 |   1.13× |

Summary across the long-context sweep:

- **Peak FP8 fwd throughput: ~413 TFLOP/s sustained** (≈ 21 % of H200 FP8 SOL).
- Best absolute fwd throughput sits at B·H ≥ 8 in the 57k–72k context band.
- Larger N (162k–323k) drops modestly to ~387–408 TFLOP/s; the kernel
  stays >1.1× faster than bf16 SDPA fwd everywhere measured.
- Best relative speedup over bf16 SDPA fwd:
  - (2, 1, 162048) → **1.19×**
  - (2, 4, 57600)  → **1.18×**
  - 8 shapes tie at **1.17×**

### FP8 backward (measured at moderate context)

FP8 backward currently peaks at **35.22 TFLOP/s** at (B=2, H=16, N=3072,
D=128) — the same shape where bf16 SDPA bwd-only does **493.76 TFLOP/s**:
bf16 SDPA bwd is ~14× faster than the FP8 bwd today on this hardware.
The backward path still falls back to bf16 for `dQ` and spills registers
at D=128; see "Current Slowdown Diagnosis" below.

### Quantization kernels in the long-context sweep

- **Per-token Q/K** (single-pass HBM): peaks at **3900 GB/s ≈ 81 % of
  HBM SOL** (B=8, H=1, N=162048).
- **Per-channel V** (two-pass HBM): peaks at **1558 GB/s** (B=8, H=2,
  N=57600). With 9 B/elem true traffic this corresponds to ~58 % of HBM
  SOL.
- Per-channel V is the only meaningful HBM gap left; per-token quant is
  already HBM-bound.

### Summary table — best absolute & speedup across fwd + bwd (H200)

| Direction | Best abs. TFLOP/s     | Shape (B,H,N,D)    | Best speedup vs bf16 SDPA | Shape (B,H,N,D)        |
| --------- | --------------------: | ------------------ | ------------------------: | ---------------------- |
| FP8 fwd   |               **413** | (8, 1, 71808, 128) |                  **1.19×** | (2, 1, 162048, 128)    |
| FP8 bwd   |             **35.22** | (2, 16, 3072, 128) |                  **0.07×** | (2, 16, 3072, 128)     |

(FP8 bwd "speedup" is below 1 because the kernel is currently slower than
bf16 SDPA bwd on every measured shape.)

## Quantization Kernel Profiling

`profile_quant.py` sweeps the standalone allocation-free quantization kernels
(`fp8_quantize_per_token_out`, `fp8_quantize_per_channel_out`, and
`int8_quantize_per_token_out`) across batch and sequence length, reporting
per-launch ms, GB/s (logical traffic = read 4 B/elem fp32 + write 1 B/elem
quantized code + scale writes), and elements/s.

```bash
python3 profile_quant.py --B 4 8 16 --H 16 \
  --N 2048 4096 8192 16384 --D 128

# Quick quantization smoke/timing run, not for final reported numbers.
python3 profile_quant.py --B 4 --H 16 --N 4096 --D 128 --quick-profile
```

The profiler uses the final-reporting benchmark recipe by default. Pass
`--quick-profile` only for local smoke/timing runs.

### Optimized kernel numbers (NVIDIA H200, HBM peak ≈ 4.48 TB/s)

```text
B=4   H=16  N=8192   D=128 | per-token:   0.089 ms  3533 GB/s | per-channel:  0.213 ms  1470 GB/s
B=8   H=16  N=8192   D=128 | per-token:   0.174 ms  3611 GB/s | per-channel:  0.400 ms  1561 GB/s
B=16  H=16  N=16384  D=128 | per-token:   0.686 ms  3668 GB/s | per-channel:  1.530 ms  1634 GB/s
```

**Strongest shapes (best absolute throughput):**

FP8 per-token quantization (effective HBM bandwidth, single-pass):

- (B=16, H=16, N=16384, D=128) → **3668 GB/s** (≈ 82 % of H200 HBM SOL)
- (B=8,  H=16, N=8192,  D=128) → **3611 GB/s**
- (B=4,  H=16, N=8192,  D=128) → **3533 GB/s**

FP8 per-channel quantization (two HBM passes; see note below):

- (B=16, H=16, N=16384, D=128) → **1634 GB/s** (≈ 2940 GB/s of true HBM
  traffic, ~66 % of H200 HBM SOL)
- (B=8,  H=16, N=8192,  D=128) → **1561 GB/s**
- (B=4,  H=16, N=8192,  D=128) → **1470 GB/s**

These numbers are H200-only (HBM peak 4.48 TB/s). On an H100 PCIe
(HBM peak ~2.0 TB/s) the per-token kernel is HBM-bound and tops out at
~1.6 TB/s — `profile_long_context.py` reports its SOL against that H100
PCIe peak.

- **FP8 per-token** reaches ~82% of measured HBM peak (single read+write pass).
- **INT8 per-token** uses the same one-warp-per-row structure and is reported
  beside FP8 per-token by `profile_quant.py`.
- **FP8 per-channel** reaches ~36% as reported, but the kernel does two HBM
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

…and saturates at long context around B·H ≥ 8, N ∈ [57k, 72k]:

```text
B=2 H=4  N=57600  D=128: 302 TFLOP/s   (1.16× bf16 SDPA fwd)
B=4 H=4  N=57600  D=128: 301 TFLOP/s   (1.22× bf16 SDPA fwd)
B=4 H=2  N=71808  D=128: 301 TFLOP/s   (1.20× bf16 SDPA fwd)
```

It is faster than fp32 SDPA for every measured shape, and **faster than
bf16 SDPA in forward by up to 1.22×** at long context. bf16 SDPA is still
faster in backward (the FP8 backward path is currently the bottleneck —
see below).

Key pattern across the whole sweep:

- Quantization kernels scale with B·H·N and become strongly HBM-bound on
  large workloads — per-token saturates at ~82 % of HBM SOL.
- The FP8 attention kernel itself peaks around medium-long contexts
  (~50k–70k tokens) before utilization slowly degrades at ultra-long
  context (162k–323k).

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

The actionable backward bottlenecks are now tracked here rather than in a
separate stale notes file:

- `dQ = dS @ K` is the critical path and currently falls back to bf16 WGMMA.
  Reformulating as `dQ^T = K^T @ dS^T` would make the matmul fit the FP8 `AB^T`
  primitive and remove the bf16 shadow-K load.
- Backward runs at one CTA per SM because shared memory is essentially full.
  Removing bf16 shadow buffers may make two CTAs per SM possible and improve
  latency hiding.
- The loop quantizes intermediate `dS`, `dV`, and `dK` tiles; the scalar div,
  clamp, and stochastic-rounding work is serialized with the matmul pipeline.
- Consumer work is imbalanced: the consumer that owns `dQ` has more work and
  can hold back the other consumer.

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
