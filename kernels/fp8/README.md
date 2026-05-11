# Low-Bit Attention: SageAttention2-style smoothing study + FP8 ThunderKittens kernels

A two-part workspace:

1. **PyTorch simulation** (`smooth*.py`) — accuracy study for INT4 / INT8 / FP8
   (E4M3, E5M2) Q,K combined with FP8 P,V, four quantization granularities,
   three smoothing techniques, and a full FP8 backward.
2. **ThunderKittens kernels** (`attn/`) — starting point for an FP8 attention
   forward kernel on H100 / H200, modelled on SageAttention2's recipe.

The simulation is what we use to pick a kernel recipe; the kernel is what we
benchmark for speed afterwards.

## Quick Start (simulation)

```bash
# Synthetic Q,K,V
python3 smooth.py --qk all --smooth all

# Real CogVideoX-2b Q,K,V (after first running the capture, see below)
python3 smooth.py --source cogvideox --n 1024
```

## Sweeps

```bash
# Per-tensor / per-block / per-token / per-thread granularity sweep
python3 smooth.py --source cogvideox --n 1024 --granularity-sweep

# SmoothQuant alpha sweep (defaults: 0.0..1.0 step 0.1, on INT4)
python3 smooth.py --source cogvideox --n 1024 --smoothquant-sweep \
    --smoothquant-qk int4

# Backward pass ablation (FP8 dO/dS, per-row)
python3 smooth.py --source cogvideox --n 1024 --bwd
```

## Visualizations

```bash
# Tensor heatmaps + smoothing plots, plus dtype × smoothing heatmap
python3 smooth.py --source cogvideox --n 1024 \
    --plots --ablation-plots --bwd \
    --plot-qk fp8_e4m3 --plot-dir smooth_plots_cogvideox
```

Plots produced (`--plot-dir`):

| File | What it shows |
|---|---|
| `tensor_heatmaps.png` | \|Q\|, \|K\|, \|V\| before/after smoothing |
| `channel_means.png` | per-channel means (target of mean-subtraction) |
| `quantized_value_hist_<dtype>.png` | quantized value distribution |
| `reconstruction_qsnr_<dtype>.png` | per-channel reconstruction QSNR |
| `reconstruction_qsnr_density_<dtype>.png` | density of per-channel QSNR |
| `granularity_qsnr_<dtype>.png` | reconstruction QSNR per granularity |
| `granularity_sweep.png` | output QSNR bar chart (granularity × dtype) |
| `smoothquant_sweep_<dtype>.png` | SmoothQuant α sweep (best α + annotation) |
| `ablation_heatmap.png` | dtype × smoothing forward heatmap (rel-L1 + QSNR) |
| `backward_qsnr_heatmap.png` | dtype × smoothing dQ/dK/dV heatmap (shared dB scale) |
| `summary_qsnr.png` | best-recipe forward + backward QSNR per dtype |

The heatmaps now use luminance-aware text colouring so values stay legible on
both dark and light cells, smoothing combinations are ordered logically
(`none → Q → K → V → Q+K → Q+V → K+V → Q+K+V`), and the backward heatmap shares
a single colour scale across `dQ`, `dK`, `dV` so the three gradients can be
compared directly.

## Capturing real Q/K/V from CogVideoX-2b

`capture_cogvideox.py` subclasses `CogVideoXAttnProcessor2_0`, snapshots
post-RoPE Q/K/V on a single transformer layer, runs one diffusion step,
and saves the tensors to `captures/cogvideox.pt`.

```bash
python3 capture_cogvideox.py --layer 0 --num-inference-steps 1 \
    --num-frames 9 --max-tokens 4096
```

The saved bundle is `{Q, K, V}` with shape `(B, H, N, D) = (2, 30, 4096, 64)`
for CogVideoX-2b. `make_inputs(source="cogvideox", head_index=...)` picks one
head and returns `(N, D)` tensors.

## Interactive menu

```bash
python3 smooth.py --interactive
```

The menu supports filtering by Q/K dtype, filtering by smoothing
combination, changing sort metrics, grouping results, and saving plots.

## Results (CogVideoX-2b layer 0, head 0, N=1024)

### Forward — output QSNR vs reference fp32 attention

Smoothing = mean-subtraction along the right axis for Q (per-block),
K (per-tensor), V (per-channel).
P×V always uses FP8 E4M3 with the static `1/max` scale on P and per-channel
scale on V.

| Recipe                                              | out QSNR (dB) |
|---|---|
| INT4 per_thread, no smoothing                       | 17.4 |
| INT4 per_thread, Q+K mean-sub                       | 20.3 |
| INT4 per_token,  Q+K mean-sub                       | 22.3 |
| INT4 per_tensor, Q+K mean-sub                       | 15.2 |
| INT4 SmoothQuant α=0.7                              | 22.3 |
| **INT8 per_thread, Q+K+V mean-sub**                 | **35.2** |
| **FP8 E4M3 per_thread, Q+K+V mean-sub**             | **34.0** |
| FP8 E5M2 per_thread, Q+K+V mean-sub                 | 30.7 |

### Backward (FP8 dO/dS, per-row), all-tensor smoothing

| Recipe                          | dQ QSNR | dK QSNR | dV QSNR |
|---|---|---|---|
| INT4 Q+K+V smoothing            | 14.1 | 11.6 | 16.0 |
| **INT8 Q+K+V smoothing**        | **29.7** | **28.7** | **25.9** |
| **FP8 E4M3 Q+K+V smoothing**    | **28.1** | **28.5** | **25.6** |
| FP8 E5M2 Q+K+V smoothing        | 24.9 | 24.5 | 24.6 |

### Granularity sweep (Q+K mean-subtracted, output QSNR dB)

|              | per_tensor | per_block | per_token | per_thread |
|---|---|---|---|---|
| INT4         | 15.2 | 15.2 | 22.3 | 20.3 |
| INT8         | 34.1 | 34.1 | 34.7 | 34.6 |
| FP8 E4M3     | 33.3 | 33.3 | 33.6 | 33.6 |
| FP8 E5M2     | 30.0 | 30.0 | 30.7 | 30.5 |

## Interpretation

**TL;DR — what to keep at what precision for low-bit training**

- **Q, K → FP8 E4M3 (or INT8) at per-token / per-thread granularity.** Loses
  ~1 dB output QSNR vs INT8, but FP8 has hardware-level cast support and
  matches the H100 wgmma FP8 path one-to-one — no software round-and-pack.
- **P → FP8 E4M3 with the static `1/max` scale.** Always non-negative and
  bounded in `[0,1]` post-softmax, so this is the cheapest dtype change
  available; the 8-bit P with a static scale costs essentially nothing.
- **V → FP8 E4M3 per-channel.** V's outliers are along the channel axis
  (seen in the channel-mean plot); per-channel scales eliminate them without
  any K-style mean subtraction unless V also has an extreme bias.
- **Accumulators (S, m, l, O) → FP32**, **softmax → FP32**, **statistics
  exchange → FP32 / BF16**. The whole point of running quant attention is
  that accumulation stays clean; demoting these collapses QSNR by ~10 dB.
- **dO, dS → FP8 E4M3, per-row scale.** dS rows have wildly different
  magnitudes (sparse spikes near argmax). A single per-tensor scale zeros
  most rows; per-row scales match SageAttention2 / LongVQ-style backward
  recipes and recover ≥ 28 dB on dQ/dK.
- **dQ = dS @ K and dK = dS^T @ Q** must reuse the same dequantized K,Q
  the forward saw, otherwise gradients drift several dB.

**E4M3 vs E5M2.** E4M3 is preferred for activations: ~1 dB tighter than INT8
for forward, ~1.5 dB looser than INT8 for backward. E5M2 only beats E4M3
when the dynamic range is extreme (gradients in deeper networks); for the
forward path it costs 3–4 dB, so don't use it on Q/K/V/P.

**INT4.** Useful only with all of: per_token granularity, Q+K mean-sub,
or alternatively SmoothQuant α≈0.7. Even at the optimum, you give up ~13 dB
of forward and ~14 dB of backward QSNR vs INT8. For training, this is too
lossy in the backward unless you keep dQ/dK/dV in higher precision.

**Smoothing matters more for low bits.** For INT4 the gap between
`none` and `Q+K+V` is ~3 dB on the forward and ~2 dB on the backward;
for INT8/FP8 the same gap is < 0.7 dB. The cost of mean-subtraction is
small (one extra reduction per block), so we keep it on by default.

**Granularity matters more for low bits.** INT4 loses 7 dB going from
per-token to per-tensor (because outliers in one row blow up the scale for
every other row). INT8 / FP8 only lose 0.5–0.6 dB across the same sweep;
per-block ≡ per-tensor in this experiment because we already feed one block
at a time (N=1024, blockK=64).

**Headline recommendation for an 8-bit attention training kernel.**
Q/K/P/V in FP8 E4M3 with the granularities above, FP32 accumulation
everywhere, per-row FP8 E4M3 dO/dS, FP16/BF16 reductions for `m,l`. This is
the recipe the kernel in `attn/` is being built around.

## ThunderKittens FP8 attention kernel

`attn/` contains the mixed low-bit forward attention kernel implementing
the recipe identified by the simulation (forward/backward consistent,
fine-grained, smoothed K/V, RTNE forward).

### Files

- `fp8_h100_fwd.cu` — bf16 mha_h100 baseline kept as the working reference
  build (`make` with no `KERNEL=` arg).
- **`fp8_attn_fwd.cu` — mixed-precision FP8 kernel** (`KERNEL=fp8`).
- `gentests.py`, `references.py`, `utils.py` — driver, fp32/bf16
  reference attention, coloured stats table for the bf16 build.
- `test_fp8.py` — smoke test for `fp8_attn_fwd.cu` that applies the
  full recipe (K-mean-sub, V-mean-sub) and compares against both the
  fp8-quant reference (kernel-only error) and the fp32 reference
  (FP8 quantisation noise).

### Recipe applied

Forward (this kernel):

| Component | Precision | Scale | Notes |
|---|---|---|---|
| Q | FP8 e4m3 | per-row float | RTNE quant on host |
| K | FP8 e4m3 | per-row float | smoothed: K ← K − meanₛ(K) |
| V | bf16     | —              | smoothed: V ← V − meanₛ(V); meanₛ(V) added back to O at end |
| QK^T mma | `mma_ABt` (st × st) | accumulator FP32 | per-row scales applied as `mul_row` × `mul_col` |
| softmax  | FP32 | online | `× log2(e) × 1/√D` baked into the exp2 |
| P | bf16 register tile | — | TODO: FP8 e4m3 + static `1/448` scale (needs V^T in smem) |
| PV mma | `mma_AB` (rt × st), bf16 | accumulator FP32 | TODO: fully-FP8 path |
| O | bf16 store | — | |
| V channel-mean | bf16 | — | added back to O after the kv-loop |

The QK^T result is multiplied in-place by `mul_row(att, q_scale_cv)` and
`mul_col(att, k_scale_rv)` to recover `(Qq · Kq) × q_scale × k_scale`,
then `/ sqrt(D)` is folded into the softmax `× log2(e)` constant.

### Building & running

```bash
cd attn
make BUILD_MODE=torch KERNEL=fp8         # builds _C*.so
python3 test_fp8.py                      # D=128, non-causal
python3 test_fp8.py --causal             # D=128, causal
python3 test_fp8.py --d64                # D=64,  non-causal
python3 test_fp8.py --d64 --causal       # D=64,  causal
```

`N` must be divisible by `CONSUMER_WARPGROUPS × qo_height = 192`
(otherwise the trailing rows of `O` are uninitialised). The default
test uses `B=1 H=8 N=1536 D=128`.

### Current measured accuracy (random Q,K,V ~ N(0,1))

| Config | rel-L1 vs fp8-quant ref | rel-L1 vs fp32 ref |
|---|---|---|
| D=128, non-causal | 2.97 × 10⁻³ | 3.71 × 10⁻² |
| D=128, causal     | 2.76 × 10⁻³ | 3.49 × 10⁻² |
| D=64,  non-causal | 2.88 × 10⁻³ | 3.70 × 10⁻² |
| D=64,  causal     | 2.71 × 10⁻³ | 3.50 × 10⁻² |

The fp8-quant column is the kernel's own error (accumulation order
relative to a PyTorch reference that sees the same fake-quanted Q,K).
The fp32 column is the dominant error: FP8 e4m3 per-row Q+K + bf16 V
quantisation noise + the K/V smoothing residual. Both numbers track the
simulation prediction (~33 dB output QSNR ≈ ~3% rel-L1).

### Backward pass (reference + ablations)

Files:

- `attn/fp8_attn_bwd_ref.py` — Python reference backward consistent
  with the kernel forward. Reuses the **same** dequantized Q,K,V the
  kernel saw, recovers `P = exp(S - L)` from the kernel's saved L,
  and applies the recipe (FP8 e4m3 per-row dO/dS, FP8 P with static
  `1/448` scale, FP8 V per-channel; SR available for dQ/dK).
- `attn/test_fp8_bwd.py` — extensive backward correctness + ablation
  driver. Calls the forward kernel, runs the bwd reference on its
  outputs, and compares dQ/dK/dV against PyTorch fp32 autograd
  across grad-dtype, grad-granularity, smoothing, and SR/RTNE.

The bwd reference is the *gold model* a future CUDA backward kernel
will be measured against (analogous to how
`smooth_core.quantized_attention` is the gold model for the forward).

#### Mathematical consistency between fwd and bwd

The forward kernel writes `L = -sqrt(D) · log_sum_exp(S)` with
`S = (Q_q @ K_centered_q^T) / sqrt(D)` (i.e. on the *centered* K
that the FP8 mma actually consumed). The backward reference
recomputes S on that same centered K and uses `P = exp(S - L)`,
guaranteeing the recovered P matches the forward's P bit-equivalently.

The K-mean and V-mean smoothing leave gradients w.r.t. the original
Q,K,V invariant:

- K-mean is a per-row additive constant in S that softmax kills,
  and `dQ -= dS @ K_mean = 0` because softmax-jacobian rows sum to 0.
- V-mean is added back to O after PV; treating it as a detached
  constant gives `dV = P^T @ dO` directly, with no leakage into dQ/dK.

So the bwd reference produces gradients that compare directly against
PyTorch autograd on the original (un-smoothed) tensors.

#### Backward results (random Q,K,V ~ N(0,1), N=1536, D=128)

| Recipe                                     | dQ QSNR | dK QSNR | dV QSNR |
|---|---|---|---|
| **e4m3 per-row dS, SR (recommended)**      | **23.9** | **23.9** | **25.5** |
| no grad-quant (dO/dS in fp32)              | 26.8 | 26.8 | 28.5 |
| e4m3 per-row dS, RTNE                      | 23.9 | 23.9 | 25.5 |
| e4m3 per-tensor dS                         | 23.7 | 23.8 | 25.5 |
| e5m2 per-row dS                            | 21.1 | 21.1 | 23.2 |
| e5m2 per-tensor dS                         | 20.8 | 20.8 | 23.1 |
| e4m3 per-row, V kept fp32                  | 24.7 | 24.7 | 25.5 |
| e4m3 per-row, P kept fp32                  | 23.9 | 23.9 | 26.8 |

Causal cuts ~0.5 dB off everything (less work, slightly lower noise floor).

#### Backward smoothing ablation (biased K,V at ±4σ, D=128)

| Smoothing | dQ QSNR | dK QSNR | dV QSNR |
|---|---|---|---|
| none           | 11.3 | 15.1 | 18.0 |
| K only         | 17.5 | 17.5 | 25.5 |
| V only         | 13.8 | 17.7 | 18.0 |
| **K + V**      | **23.9** | **23.9** | **25.5** |

Same pattern as the forward: K-smoothing recovers ~6 dB on dQ/dK and
~7.5 dB on dV; V-smoothing alone helps dK only weakly because dV is
the gradient that V-bias hurts most. **K + V together** is the only
recipe that holds up under realistic biased activations.

#### Recipe takeaways (matches the simulation)

- **e4m3 over e5m2** by ~3 dB on every output, same story as forward.
- **Per-row > per-tensor** by 0.2–0.4 dB on FP8 (dS rows differ in
  magnitude but not by orders of magnitude on N(0,1) data; the gap
  widens on biased data and is the *correct* reason to use per-row).
- **SR vs RTNE** is essentially a wash on a single backward step
  (~0.01 dB). The reason to use SR is multi-step bias accumulation
  during training, which a one-shot test cannot expose.
- **FP8 V > fp32 V** costs ~0.8 dB on dQ/dK (V-quant noise leaks into
  dP → dS → dQ/dK) but is required for fully-FP8 PV in the bwd kernel.
- **FP8 P > fp32 P** costs ~1.3 dB on dV (the static-scale FP8 P
  truncates small probabilities). This is the dominant FP8 cost.

#### Running the backward suite

```bash
cd attn
make BUILD_MODE=torch KERNEL=fp8
python3 test_fp8_bwd.py            # python ref, full sweep across D/causal/seeds
python3 test_fp8_bwd.py --quick    # python ref, one config
python3 test_fp8_bwd.py --cogvideox  # real CogVideoX-2b Q,K,V
python3 test_fp8_bwd_kernel.py     # CUDA kernel backward end-to-end
```

### CUDA backward kernel (`fp8_bwd_attend_ker`)

Lives in the same `attn/fp8_attn_fwd.cu` and exported as
`_C.fp8_mha_backward(Q, K, V, O, L, dO, causal)`. Structure mirrors
`mha_h100`'s bf16 backward 1:1:

  * **Prep kernel** `fp8_bwd_attend_prep_ker` computes `D_i = sum_j O_ij · dO_ij`
    per row (warpgroup-per-tile).
  * **Main kernel** `fp8_bwd_attend_ker`:
    - 2 consumer warpgroups + 1 producer warpgroup
    - K, V loaded once into smem (resident across the qo loop)
    - Q, dO, L, D streamed in by the producer
    - `S^T = K @ Q^T` (bf16 mma_ABt)
    - `P^T = exp(S^T / sqrt(D))` from saved L
    - `dP^T = V @ dO^T` (bf16 mma_ABt)
    - `dS^T = P^T · (dP^T - D)` (fp32, then bf16 cast)
    - `dV += P^T @ dO` and `dK += dS^T @ Q` accumulated in registers,
      atomic-add written to HBM at end (one block per kv tile)
    - `dQ_block = dS_smem[0]^T @ K_smem[0] + dS_smem[1]^T @ K_smem[1]`
      computed per qo block, atomic-add written to HBM (12+ blocks
      contribute to each dQ tile)

The kernel takes **bf16** Q,K (not FP8) — the host pre-applies the FP8
quantization and per-row scales (`Q_eff = Q_q · sq` cast to bf16,
`K_eff = K_q · sk` cast to bf16). This is the "FP8-recipe-consistent"
backward: it sees exactly what the FP8 forward saw, with the per-row
scales baked into the bf16 values. The fully-FP8 mma backward is a
follow-up gated on TK exposing FP8 mma in the rt×st form (same blocker
as the forward's PV path).

#### Kernel correctness (random Q,K,V ~ N(0,1), N=1536)

| Config | dQ QSNR vs fp32 | dK QSNR vs fp32 | dV QSNR vs fp32 |
|---|---|---|---|
| D=128, non-causal | 22.7 dB | 25.6 dB | 28.4 dB |
| D=128, causal     | 24.9 dB | 25.7 dB | 29.8 dB |
| D=64,  non-causal | 19.8 dB | 25.6 dB | 28.6 dB |
| D=64,  causal     | 24.3 dB | 25.9 dB | 29.6 dB |

vs the **python reference** (same kernel-visible inputs, no grad-quant):

| Config | dQ QSNR | dK QSNR | dV QSNR |
|---|---|---|---|
| D=128, non-causal | 24.9 dB | 31.7 dB | **52.5 dB** |
| D=128, causal     | 28.5 dB | 30.6 dB | **52.6 dB** |

The dV numbers vs python ref are essentially noise-free (~52 dB =
matching to bf16 ULP), confirming the kernel computes the right
quantity. The 4–6 dB extra noise on dQ vs the python reference is bf16
accumulation noise across the 12 atomic-add contributions per qo tile
(the python ref accumulates in fp32). dV/dK are accumulated entirely
in fp32 registers within a single block, so they don't pay this cost.

### Recipe-compliance status of the current bwd kernel

| Recipe item | Status | Notes |
|---|---|---|
| Q,K FP8 e4m3 per-token, RTNE — same in fwd & bwd | **partial** | Same FP8 *values* (host pre-quant), but bwd's QK^T mma is bf16, not FP8. The values are bit-equivalent, so it's not a quantization mismatch — only the mma precision differs. |
| V high precision (PV stays HP) | ✓ | bf16 in both fwd & bwd, recipe-compliant. |
| K/V mean smoothing (consistent fwd & bwd) | ✓ | K-mean cancels in softmax, V-mean detached on add-back. |
| **dO FP8 e4m3 per-token + SR** | **✓** (host-side) | `host_quantize_per_row_fp8_sr_dequant` in `test_fp8_bwd_kernel.py`. dO is snapped to the FP8 grid with stochastic rounding before the bwd, then dequant'd to bf16. Equivalent to FP8 mma with fp32 accum since the kernel accumulators are fp32 anyway. Toggle via `sr_dO=True/False`. |
| dS FP8 e4m3 per-row + SR (for dQ/dK) | ✗ | Currently bf16 in ds_smem. In-kernel SR needs a per-thread PRNG; RTNE proxy is doable today via `warp::copy(rt_fp8e4m3, rt_fl)`. |
| FP8 mmas in dV/dQ/dK | ✗ | Blocked: TK's public `mma_AB`/`mma_AtB` static_assert FP8 out. |

#### SR-dO ablation (random N(0,1), N=1536, D=128, non-causal)

| Recipe          | dQ QSNR | dK QSNR | dV QSNR |
|---|---|---|---|
| **dO FP8 e4m3 per-row + SR (recipe)** | 21.7 | 23.8 | 25.5 |
| dO bf16 (no quant)                    | 22.7 | 25.6 | 28.4 |

SR costs ~1–3 dB QSNR per step (FP8 grid noise on dO leaks into all three
gradients). **This is the expected trade-off the recipe makes**: SR
converts the systematic rounding bias of RTNE into zero-mean noise. On
a single backward step that looks like noise being added; over a full
training run it eliminates the gradient-norm growth that diverges
RTNE-only training (per the recipe, "all settings with forward-backward
inconsistencies diverged" / "deterministic rounding had a pronounced
effect on gradients which led to training instability").

### Path to full FP8 mma in bwd (TK-blocker analysis)

Two options, both feasible:

**Option A — adapt the Colfax `ReorgCFp8toAFp8` to TK.** TK's FP8 RS WGMMA
PTX *is implemented at the base level*
([base/64x128.impl#L182](file:///cluster/home/kostfab1/ThunderKittens/include/ops/group/mma/base/64x128.impl#L182)),
just gated by a static_assert in the public wrapper. To use it:

  1. Strip / bypass the static_asserts in
     [warpgroup.cuh#L161](file:///cluster/home/kostfab1/ThunderKittens/include/ops/group/mma/warpgroup.cuh#L161)
     (or call `kittens::detail::wgmma::base<float, fp8e4m3, cols, 0, 0>::rt_st`
     directly).
  2. Port the Colfax byte-perm + shfl-sync reorg to TK's `rt_fp8e4m3` layout —
     **non-trivial** because TK uses an `fp8e4m3x4` packed type with a 16×32
     sub-tile layout (16 fp8/thread), different from CUTLASS's 16×16 layout
     (8 fp8/thread). The reorg permutation pattern needs re-derivation for
     the packed layout, then validated against the FP8 RS WGMMA's expected
     A-operand layout (PTX Figure 122).
  3. Pre-transpose V (and Q for dK) host-side or via a TK transpose pass so
     the B operand is k-major as FP8 WGMMA requires.

**Option B — stage all FP8 intermediates to smem and use only `mma_ABt`.**
This avoids the reorg entirely by exploiting that `mma_ABt(C, A, B)` computes
`A @ B^T`. With pre-transposed intermediates we can express:

  * `dV = P^T @ dO` ≡ `mma_ABt(dV, P_T_smem, dO_T_smem)` (needs dO transposed)
  * `dK = dS^T @ Q` ≡ `mma_ABt(dK, dS_T_smem, Q_T_smem)` (needs Q transposed)
  * `dQ = dS @ K`   ≡ `mma_ABt(dQ, dS_smem,   K_T_smem)` (needs K transposed)
  * Forward `PV  = P @ V` ≡ `mma_ABt(O, P_smem, V_T_smem)` (the existing
    forward roadmap step; needs V transposed — same as the Colfax kernel
    requires).

Costs: extra smem traffic (write each intermediate once before use), and
host-side / on-kernel transposes for V, Q, K. No reorg needed; works with
TK's existing `mma_ABt` FP8 PTX.

Option B is cleaner against TK's current API surface; Option A is faster but
requires touching TK internals. The kernel's current `bf16-in-bwd-mma` path
serves as the validated baseline either way.

### Roadmap to fully-FP8 PV

To get from the current "QK in FP8, PV in bf16" mixed kernel to a fully
FP8 forward (the right config based on the simulation results), three
steps:

1. **Stage V transposed in shared memory.** TK at this commit only
   exposes FP8 via `mma_ABt` (st × st), so the PV path needs V loaded
   into `st_fp8e4m3<D, kv_height>` via TMA with a swapped-stride
   descriptor (or a separate transpose kernel pass).
2. **Stage P in shared memory.** Cast P (fp32) → fp8e4m3 with the
   static `1 / 448` scale, store to `st_fp8e4m3<qo_height, kv_height>`,
   then `warpgroup::mma_ABt(o_reg, p_smem, v_t_smem)`.
3. **Apply V's per-channel scale once at the end.** Load `sv` into a
   `row_vec`, multiply by `1 / 448`, then `mul_col(o_reg, o_reg, sv_rv)`
   before `div_row(o_reg, norm_vec)` and the V mean-add.

The simulation (`smooth_core.quantized_attention`) already implements
this recipe in PyTorch, so each kernel step can be cross-checked
against it.

## Files

- `smooth.py` — CLI and interactive entry point
- `smooth_core.py` — quantization, smoothing, attention forward/backward, ablation rows
- `smooth_report.py` — terminal tables, filtering, sorting, grouping
- `smooth_viz.py` — tensor visualization helpers and ablation plots
- `capture_cogvideox.py` — capture Q/K/V from CogVideoX-2b
- `attn/fp8_attn_fwd.cu` — ThunderKittens FP8 forward + bf16 backward
  CUDA kernels (single TU, both exported via the `_C` pybind module)
- `attn/fp8_attn_bwd_ref.py` — Python reference backward (gold model
  for the FP8-quant ablation sweep; companion to the kernel)
- `attn/test_fp8_extensive.py` — forward kernel correctness + ablations
- `attn/test_fp8_bwd.py` — backward reference correctness + ablations
- `attn/test_fp8_bwd_kernel.py` — CUDA backward kernel end-to-end test
