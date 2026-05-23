// Mixed low-bit forward attention for ThunderKittens on H100 / H200.
//
// Implements the SVQ-style mixed low-bit recipe (per the study notes):
//
//   * Forward & backward must use the same dtype/granularity (this file = fwd).
//   * Smaller granularity beats per-tensor — we use per-token (per-row) scales
//     for Q and K.
//   * Smooth K (and V) by subtracting the channel-wise mean before quant.
//     For K the constant logit shift Q · mean_K cancels in softmax.
//     For V we do PV = P (V - mean_V) + mean_V at the end (sum_j P_ij = 1).
//   * Treat PV as the "dangerous" path: we keep V (and the PV matmul) in
//     bf16 in this revision because TK's FP8 path at this commit only
//     exposes `mma_ABt` (st × st), not `mma_AB` (rt × st) needed for the
//     fused softmax→PV step. The next iteration will add an FP8 V^T smem
//     load and a static-scaled FP8 P.
//   * Forward uses RTNE rounding (default cast behaviour).
//
//   Q, K          : fp8 e4m3
//   per-row Q     : float scale (one per token)
//   per-row K     : float scale (one per token)
//   K mean        : per-channel float, subtracted before quant of K (host),
//                   no logit-shift correction needed (cancels in softmax).
//   QK^T mma      : fp8 e4m3 (mma_ABt) → fp32 accumulator
//   softmax       : fp32 (online; LOG2E*1/sqrt(D) baked into the exponent)
//   V             : bf16, channel-mean-subtracted (host).
//   V channel-mean: bf16, added back to O after the kv-loop.
//   P             : fp32 → bf16 register tile
//   PV mma        : bf16 → fp32 accumulator
//   O             : bf16 store
//
// The kernel is structurally a copy of mha_h100 (warpgroup-specialised
// producer/consumer with TMA-pipelined K,V loads).

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <cooperative_groups.h>
#include <iostream>

constexpr int CONSUMER_WARPGROUPS = 3;
constexpr int PRODUCER_WARPGROUPS = 1;
constexpr int NUM_WARPGROUPS      = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
constexpr int NUM_WORKERS         = NUM_WARPGROUPS * kittens::WARPGROUP_WARPS;

using namespace kittens;
namespace cg = cooperative_groups;

// log2(e), used to convert exp -> exp2 for the softmax fast-path.
constexpr float LOG2E = 1.44269504089f;
constexpr float FP8_E4M3_MAX_F = 448.0f;

// -----------------------------------------------------------------------------
// Standalone FP8 quantization kernels used by the Python test harness.
// -----------------------------------------------------------------------------

// ---- Per-token (per-row) quantization ----------------------------------------
//
// One warp processes one row of width D. 4 warps per block -> 4 rows per
// block. Loads are vectorized (float4 for D=128, float2 for D=64) and the
// amax reduction uses a warp shuffle (no shared memory). FP8 outputs are
// packed into 32-bit (D=128) or 16-bit (D=64) stores via fp8e4m3_4 /
// fp8e4m3_2 so the writes are coalesced and minimal.

template <int D>
__global__ void fp8_quantize_per_token_ker(const float *__restrict__ x,
                                           fp8e4m3 *__restrict__ xq,
                                           float *__restrict__ descale,
                                           int rows) {
    static_assert(D == 64 || D == 128, "Only D=64 or D=128 supported");
    constexpr int WARPS_PER_BLOCK = 4;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row  = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= rows) return;

    const float *xr = x  + row * D;
    fp8e4m3    *xqr = xq + row * D;

    if constexpr (D == 128) {
        const float4 *xr_v = reinterpret_cast<const float4*>(xr);
        float4 v = xr_v[lane];
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                           fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));

        // IEEE-rounded division to match the PyTorch reference exactly
        // (the build uses --use_fast_math, which turns `/` into recip*mul).
        float s = fmaxf(__fdiv_rn(amax, FP8_E4M3_MAX_F), 1.0e-12f);
        if (lane == 0) descale[row] = s;

        float4 y;
        y.x = fminf(fmaxf(__fdiv_rn(v.x, s), -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        y.y = fminf(fmaxf(__fdiv_rn(v.y, s), -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        y.z = fminf(fmaxf(__fdiv_rn(v.z, s), -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        y.w = fminf(fmaxf(__fdiv_rn(v.w, s), -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        fp8e4m3_4 packed(y);
        reinterpret_cast<uint32_t*>(xqr)[lane] = *reinterpret_cast<uint32_t*>(&packed);
    } else { // D == 64
        const float2 *xr_v = reinterpret_cast<const float2*>(xr);
        float2 v = xr_v[lane];
        float amax = fmaxf(fabsf(v.x), fabsf(v.y));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));

        float s = fmaxf(__fdiv_rn(amax, FP8_E4M3_MAX_F), 1.0e-12f);
        if (lane == 0) descale[row] = s;

        float2 y;
        y.x = fminf(fmaxf(__fdiv_rn(v.x, s), -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        y.y = fminf(fmaxf(__fdiv_rn(v.y, s), -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        fp8e4m3_2 packed(y);
        reinterpret_cast<uint16_t*>(xqr)[lane] = *reinterpret_cast<uint16_t*>(&packed);
    }
}

// ---- Per-channel (per-column) quantization ----------------------------------
//
// Original kernel mapped one block to one channel and read along N with a D
// stride - uncoalesced. The new layout maps thread d to column d within a
// (B,H) slab so consecutive threads access consecutive memory positions. To
// fill the grid even for small (B*H), N is split into CHUNK_ROWS tiles and
// the amax is reduced across blocks via an atomicMax on the float
// bit-pattern (amax is non-negative so the bit order is monotonic). A second
// pass kernel reads the final amax, computes the scale, and quantizes.

constexpr int FP8_PCH_CHUNK_ROWS = 256;

template <int D>
__global__ void fp8_per_channel_amax_ker(const float *__restrict__ x,
                                         float *__restrict__ amax_buf,
                                         int N) {
    int bh    = blockIdx.x;
    int chunk = blockIdx.y;
    int d     = threadIdx.x;
    int n0    = chunk * FP8_PCH_CHUNK_ROWS;
    int n1    = min(n0 + FP8_PCH_CHUNK_ROWS, N);
    int base  = bh * N * D;

    float amax = 0.0f;
    #pragma unroll 4
    for (int n = n0; n < n1; ++n)
        amax = fmaxf(amax, fabsf(x[base + n * D + d]));

    if (amax > 0.0f) {
        atomicMax(reinterpret_cast<int*>(&amax_buf[bh * D + d]),
                  __float_as_int(amax));
    }
}

template <int D>
__global__ void fp8_per_channel_quant_ker(const float *__restrict__ x,
                                          fp8e4m3 *__restrict__ xq,
                                          const float *__restrict__ amax_buf,
                                          float *__restrict__ descale,
                                          int N) {
    int bh    = blockIdx.x;
    int chunk = blockIdx.y;
    int d     = threadIdx.x;
    int n0    = chunk * FP8_PCH_CHUNK_ROWS;
    int n1    = min(n0 + FP8_PCH_CHUNK_ROWS, N);
    int base  = bh * N * D;

    float a = amax_buf[bh * D + d];
    float s = fmaxf(__fdiv_rn(a, FP8_E4M3_MAX_F), 1.0e-12f);
    if (chunk == 0) descale[bh * D + d] = s;

    #pragma unroll 4
    for (int n = n0; n < n1; ++n) {
        float y = fminf(fmaxf(__fdiv_rn(x[base + n * D + d], s),
                              -FP8_E4M3_MAX_F),
                        FP8_E4M3_MAX_F);
        xq[base + n * D + d] = fp8e4m3(y);
    }
}

static inline void launch_fp8_quantize_per_token(const float *x,
                                                 fp8e4m3 *xq,
                                                 float *s,
                                                 int rows,
                                                 int D,
                                                 cudaStream_t stream) {
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (D == 128) {
        fp8_quantize_per_token_ker<128>
            <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(x, xq, s, rows);
    } else {
        fp8_quantize_per_token_ker<64>
            <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(x, xq, s, rows);
    }
}

static inline void launch_fp8_quantize_per_channel(const float *x,
                                                   fp8e4m3 *xq,
                                                   float *descale,
                                                   int B, int H, int N, int D,
                                                   cudaStream_t stream) {
    int bh = B * H;
    int chunks = (N + FP8_PCH_CHUNK_ROWS - 1) / FP8_PCH_CHUNK_ROWS;
    dim3 grid((unsigned)bh, (unsigned)chunks);

    // Scratch for atomic-reduced amax. Zero-init because we use atomicMax on
    // the int bit-pattern of non-negative floats (0.0f -> 0).
    auto opts = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    at::Tensor amax_buf = at::zeros({bh * D}, opts);
    float *amax_ptr = amax_buf.data_ptr<float>();

    if (D == 128) {
        fp8_per_channel_amax_ker<128>
            <<<grid, 128, 0, stream>>>(x, amax_ptr, N);
        fp8_per_channel_quant_ker<128>
            <<<grid, 128, 0, stream>>>(x, xq, amax_ptr, descale, N);
    } else {
        fp8_per_channel_amax_ker<64>
            <<<grid, 64,  0, stream>>>(x, amax_ptr, N);
        fp8_per_channel_quant_ker<64>
            <<<grid, 64,  0, stream>>>(x, xq, amax_ptr, descale, N);
    }
}

std::vector<at::Tensor> fp8_quantize_per_token(at::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "input must be float32");
    TORCH_CHECK(x.dim() == 4, "input must have shape (B,H,N,D)");
    auto xc = x.contiguous();
    int64_t B = xc.size(0), H = xc.size(1), N = xc.size(2), D = xc.size(3);
    TORCH_CHECK(D == 64 || D == 128, "Only D=64 or D=128 are supported");

    auto q_opts = xc.options().dtype(at::ScalarType::Float8_e4m3fn);
    auto s_opts = xc.options().dtype(at::kFloat);
    at::Tensor xq = at::empty_like(xc, q_opts);
    at::Tensor s = at::empty({B, H, N}, s_opts);

    int rows = (int)(B * H * N);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fp8_quantize_per_token(
        xc.data_ptr<float>(),
        reinterpret_cast<fp8e4m3*>(xq.data_ptr()),
        s.data_ptr<float>(),
        rows, (int)D, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {xq, s};
}

void fp8_quantize_per_token_out(at::Tensor x, at::Tensor xq, at::Tensor s) {
    CHECK_INPUT(x); CHECK_INPUT(xq); CHECK_INPUT(s);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "input must be float32");
    TORCH_CHECK(xq.scalar_type() == at::ScalarType::Float8_e4m3fn, "xq must be FP8 e4m3");
    TORCH_CHECK(s.scalar_type() == at::ScalarType::Float, "scale must be float32");
    TORCH_CHECK(x.dim() == 4, "input must have shape (B,H,N,D)");
    TORCH_CHECK(xq.sizes() == x.sizes(), "xq shape must match input");
    TORCH_CHECK(s.size(0) == x.size(0) && s.size(1) == x.size(1) && s.size(2) == x.size(2),
                "scale must have shape (B,H,N)");
    auto xc = x.contiguous();
    int64_t B = xc.size(0), H = xc.size(1), N = xc.size(2), D = xc.size(3);
    TORCH_CHECK(D == 64 || D == 128, "Only D=64 or D=128 are supported");
    int rows = (int)(B * H * N);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fp8_quantize_per_token(
        xc.data_ptr<float>(),
        reinterpret_cast<fp8e4m3*>(xq.data_ptr()),
        s.data_ptr<float>(),
        rows, (int)D, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

std::vector<at::Tensor> fp8_quantize_per_channel(at::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "input must be float32");
    TORCH_CHECK(x.dim() == 4, "input must have shape (B,H,N,D)");
    auto xc = x.contiguous();
    int64_t B = xc.size(0), H = xc.size(1), N = xc.size(2), D = xc.size(3);
    TORCH_CHECK(D == 64 || D == 128, "Only D=64 or D=128 are supported");

    auto q_opts = xc.options().dtype(at::ScalarType::Float8_e4m3fn);
    auto s_opts = xc.options().dtype(at::kFloat);
    at::Tensor xq = at::empty_like(xc, q_opts);
    at::Tensor s = at::empty({B, H, D}, s_opts);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fp8_quantize_per_channel(
        xc.data_ptr<float>(),
        reinterpret_cast<fp8e4m3*>(xq.data_ptr()),
        s.data_ptr<float>(),
        (int)B, (int)H, (int)N, (int)D, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {xq, s};
}

void fp8_quantize_per_channel_out(at::Tensor x, at::Tensor xq, at::Tensor s) {
    CHECK_INPUT(x); CHECK_INPUT(xq); CHECK_INPUT(s);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "input must be float32");
    TORCH_CHECK(xq.scalar_type() == at::ScalarType::Float8_e4m3fn, "xq must be FP8 e4m3");
    TORCH_CHECK(s.scalar_type() == at::ScalarType::Float, "scale must be float32");
    TORCH_CHECK(x.dim() == 4, "input must have shape (B,H,N,D)");
    TORCH_CHECK(xq.sizes() == x.sizes(), "xq shape must match input");
    TORCH_CHECK(s.size(0) == x.size(0) && s.size(1) == x.size(1) && s.size(2) == x.size(3),
                "scale must have shape (B,H,D)");
    auto xc = x.contiguous();
    int64_t B = xc.size(0), H = xc.size(1), N = xc.size(2), D = xc.size(3);
    TORCH_CHECK(D == 64 || D == 128, "Only D=64 or D=128 are supported");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fp8_quantize_per_channel(
        xc.data_ptr<float>(),
        reinterpret_cast<fp8e4m3*>(xq.data_ptr()),
        s.data_ptr<float>(),
        (int)B, (int)H, (int)N, (int)D, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Tile dimensions
// -----------------------------------------------------------------------------

template<int D> struct fp8_attn_tile_dims {};
template<> struct fp8_attn_tile_dims<128> {
    constexpr static int tile_width = 128;
    constexpr static int qo_height  = 64;          // 4 * 16 rows
    constexpr static int kv_height  = 128;         // 8 * 16 rows
    constexpr static int stages     = 3;
};
template<> struct fp8_attn_tile_dims<64> {
    constexpr static int tile_width = 64;
    constexpr static int qo_height  = 64;
    constexpr static int kv_height  = 128;
    constexpr static int stages     = 3;
};

// -----------------------------------------------------------------------------
// Globals: tensors + scales
// -----------------------------------------------------------------------------

template<int D> struct fp8_attn_globals {
    using K = fp8_attn_tile_dims<D>;

    using q_tile = st_fp8e4m3<K::qo_height, K::tile_width>;
    using k_tile = st_fp8e4m3<K::kv_height, K::tile_width>;
    using v_tile = st_bf      <K::kv_height, K::tile_width>;
    using o_tile = st_bf      <K::qo_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;

    // Scale + mean layouts.
    //   Q, K row-scales:  [B, H,    1, N]   loaded as col_vec / row_vec along
    //                                       the cols (last) axis of the gl.
    //   V channel-mean:   [B, H_kv, 1, D]   added back to O at the end.
    using scale_qk_layout = gl<float, -1, -1, 1, -1>;
    using vmean_layout    = gl<bf16,  -1, -1, 1, -1>;

    using q_gl = gl<fp8e4m3, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<fp8e4m3, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,    -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float,   -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,    -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    scale_qk_layout sq;   // [B, H,    1, N]   Q row-scales
    scale_qk_layout sk;   // [B, H_kv, 1, N]   K row-scales
    vmean_layout    vm;   // [B, H_kv, 1, D]   V channel-mean (bf16)

    const int N;
    const int hr;
};

// -----------------------------------------------------------------------------
// Forward kernel
// -----------------------------------------------------------------------------

// Diffusion attention is bidirectional; no causal path is needed.
template<int D>
__global__ __launch_bounds__(NUM_WORKERS * kittens::WARP_THREADS, 1)
void fp8_attn_fwd_ker(const __grid_constant__ fp8_attn_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    using K = fp8_attn_tile_dims<D>;
    using q_tile    = st_fp8e4m3<K::qo_height, K::tile_width>;
    using k_tile    = st_fp8e4m3<K::kv_height, K::tile_width>;
    using v_tile    = st_bf<K::kv_height, K::tile_width>;
    using o_tile    = st_bf<K::qo_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;

    int warpid       = kittens::warpid();
    int warpgroupid  = warpid / kittens::WARPGROUP_WARPS;

    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages>();
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages>();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);

    int kv_blocks   = g.N / K::kv_height;
    int q_blocks    = g.N / (K::qo_height * CONSUMER_WARPGROUPS);
    int qo_heads    = gridDim.y;
    int batch_idx   = blockIdx.z;
    int qo_head_idx = blockIdx.y;
    int q_block     = blockIdx.x;
    int size_one_kv_head = g.N * K::tile_width * (int)(sizeof(fp8e4m3) + sizeof(bf16));
    constexpr int size_l2 = 50 * 1024 * 1024;

    // Keep blocks using different Q heads but the same resident K/V footprint
    // close in launch order. For very long contexts the K/V head is larger
    // than L2, so the original x/y/z mapping is cheaper and equivalent.
    if (size_one_kv_head > 0 && size_l2 >= size_one_kv_head) {
        int logical_bid = blockIdx.x + (blockIdx.y + blockIdx.z * qo_heads) * q_blocks;
        int num_hb = qo_heads * gridDim.z;
        int swizzle = 1 << (31 - __clz(size_l2 / size_one_kv_head));
        swizzle = swizzle > num_hb ? num_hb : swizzle;

        int l2_major = q_blocks * swizzle;
        int hb_group = logical_bid / l2_major;
        int l2_mod   = logical_bid - hb_group * l2_major;
        q_block      = l2_mod / swizzle;
        int hb_resid = l2_mod - q_block * swizzle;
        int hb_idx   = hb_group * swizzle + hb_resid;
        batch_idx    = hb_idx / qo_heads;
        qo_head_idx  = hb_idx - batch_idx * qo_heads;
    }
    int kv_head_idx = qo_head_idx / g.hr;
    int seq_idx     = q_block * CONSUMER_WARPGROUPS;

    __shared__ kittens::semaphore qsmem_sem,
                                  k_sem[K::stages], v_sem[K::stages],
                                  done_sem[K::stages];

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile>();
        g.k.template prefetch_tma<k_tile>();
        g.v.template prefetch_tma<v_tile>();
        g.o.template prefetch_tma<o_tile>();

        init_semaphore(qsmem_sem, 0, 1);
        for (int j = 0; j < K::stages; ++j) {
            init_semaphore(k_sem[j], 0, 1);
            init_semaphore(v_sem[j], 0, 1);
            init_semaphore(done_sem[j], CONSUMER_WARPGROUPS, 0);
        }

        tma::expect_bytes(qsmem_sem, sizeof(q_smem));
        for (int wg = 0; wg < CONSUMER_WARPGROUPS; ++wg) {
            coord<q_tile> q_idx = {batch_idx, qo_head_idx, seq_idx + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_idx, qsmem_sem);
        }

        for (int j = 0; j < K::stages - 1; ++j) {
            coord<k_tile> kv_idx = {batch_idx, kv_head_idx, j, 0};
            tma::expect_bytes(k_sem[j], sizeof(k_tile));
            tma::load_async(k_smem[j], g.k, kv_idx, k_sem[j]);
            tma::expect_bytes(v_sem[j], sizeof(v_tile));
            tma::load_async(v_smem[j], g.v, kv_idx, v_sem[j]);
        }
    }
    __syncthreads();

    int pipe_idx = K::stages - 1;

    if (warpgroupid == NUM_WARPGROUPS - 1) {
        // ---- Producer warpgroup: stream K and V tiles into smem ------------
        warpgroup::decrease_registers<32>();

        int kv_iters = kv_blocks - 2;

        if (warpid == NUM_WORKERS - 4) {
            for (int kv_idx = pipe_idx - 1; kv_idx <= kv_iters; ++kv_idx) {
                coord<k_tile> kv_tile_idx = {batch_idx, kv_head_idx, kv_idx + 1, 0};
                int s = (kv_idx + 1) % K::stages;
                warp::tma::expect_bytes(k_sem[s], sizeof(k_tile));
                warp::tma::load_async(k_smem[s], g.k, kv_tile_idx, k_sem[s]);
                warp::tma::expect_bytes(v_sem[s], sizeof(v_tile));
                warp::tma::load_async(v_smem[s], g.v, kv_tile_idx, v_sem[s]);

                wait(done_sem[kv_idx % K::stages], (kv_idx / K::stages) % 2);
            }
        }
        return;
    }

    // ---- Consumer warpgroup: compute online softmax + PV ---------------------
    warpgroup::increase_registers<160>();

    rt_fl<16, K::kv_height>  att_block;             // QK^T accumulator (fp32)
    rt_bf<16, K::kv_height>  p_bf;                  // P in bf16 for PV (kept bf16 per SageBwd)
    rt_fl<16, K::tile_width> o_reg;                 // O accumulator (fp32)

    col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

    warp::neg_infty(max_vec);
    warp::zero(norm_vec);
    warp::zero(o_reg);

    int kv_iters = kv_blocks - 1;

    wait(qsmem_sem, 0);

    constexpr float inv_sqrt_d =
        (D == 128) ? 0.08838834764f /* 1/sqrt(128) */ : 0.125f /* 1/sqrt(64) */;

    for (int kv_idx = 0; kv_idx <= kv_iters; ++kv_idx) {
        // ---- QK^T : fp8 wgmma into fp32 ------------------------------------
        wait(k_sem[kv_idx % K::stages], (kv_idx / K::stages) % 2);
        warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[kv_idx % K::stages]);

        warp::copy(max_vec_last_scaled, max_vec);
        warp::mul(max_vec_last_scaled, max_vec_last_scaled, LOG2E * inv_sqrt_d);

        // These scale loads are independent of the QK WGMMA result, so issue
        // them before waiting and keep only this iteration's vectors live.
        col_vec<rt_fl<16, K::kv_height>> q_scale_cv;
        row_vec<rt_fl<16, K::kv_height>> k_scale_rv;
        warpgroup::load(q_scale_cv, g.sq,
            {batch_idx, qo_head_idx, 0, seq_idx + warpgroupid});
        warp::load(k_scale_rv, g.sk,
            {batch_idx, kv_head_idx, 0, kv_idx});

        warpgroup::mma_async_wait();

        // ---- Apply per-row Q,K scales --------------------------------------
        // att_block[i, j] = (Qq[i] · Kq[j]) * sq[i] * sk[j]
        //
        // sq, sk are [B, H, 1, N]. Vector loads index the gl's last (cols)
        // axis: coord.c selects the c-th vec_length-element chunk along N,
        // i.e. seq_idx + warpgroupid for q (qo_height = 64 entries) and
        // kv_idx for k (kv_height = 128 entries).
        warp::mul_row(att_block, att_block, q_scale_cv);
        warp::mul_col(att_block, att_block, k_scale_rv);

        // ---- Online softmax (fp32) -----------------------------------------
        warp::row_max(max_vec, att_block, max_vec);
        warp::mul(att_block,    att_block,    LOG2E * inv_sqrt_d);
        warp::mul(max_vec_scaled, max_vec,    LOG2E * inv_sqrt_d);

        warp::sub_row(att_block, att_block, max_vec_scaled);
        warp::exp2   (att_block, att_block);

        warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
        warp::exp2(max_vec_last_scaled, max_vec_last_scaled);

        warp::mul(norm_vec, norm_vec, max_vec_last_scaled);
        warp::row_sum(norm_vec, att_block, norm_vec);

        warp::copy(p_bf, att_block);

        warp::mul_row(o_reg, o_reg, max_vec_last_scaled);

        wait(v_sem[kv_idx % K::stages], (kv_idx / K::stages) % 2);

        warpgroup::mma_AB(o_reg, p_bf, v_smem[kv_idx % K::stages]);
        warpgroup::mma_async_wait();

        if (warpgroup::laneid() == 0) arrive(done_sem[kv_idx % K::stages], 1);
    }

    warp::div_row(o_reg, o_reg, norm_vec);

    // ---- Add V's channel mean back ------------------------------------------
    // PV = P (V - mean_V) + mean_V  (since sum_j P_ij = 1).
    // vm is [B, H_kv, 1, D]; load D consecutive entries for this kv head into
    // a row_vec of length tile_width = D, then accumulate into o_reg as a per-
    // column add.
    {
        row_vec<rt_fl<16, K::tile_width>> v_mean_rv;
        warp::load(v_mean_rv, g.vm, {batch_idx, kv_head_idx, 0, 0});
        warp::add_col(o_reg, o_reg, v_mean_rv);
    }

    warpgroup::store(o_smem[warpgroupid], o_reg);
    warpgroup::sync(warpgroupid + 4);
    if (warpid % 4 == 0) {
        coord<o_tile> o_idx = {batch_idx, qo_head_idx, seq_idx + warpgroupid, 0};
        warp::tma::store_async(g.o, o_smem[warpgroupid], o_idx);
    }

    // ---- Save L = log-sum-exp for the bwd ------------------------------------
    warp::mul(max_vec_scaled, max_vec_scaled, 0.69314718056f /* ln(2) */);
    warp::log(norm_vec, norm_vec);
    warp::add(norm_vec, norm_vec, max_vec_scaled);
    if constexpr (D == 128) warp::mul(norm_vec, norm_vec, -11.313708499f);
    else                    warp::mul(norm_vec, norm_vec, -8.0f);

    warpgroup::store(l_smem[warpgroupid], norm_vec);
    warpgroup::sync(warpgroupid + 4);
    if (warpid % 4 == 0) {
        coord<l_col_vec> l_idx = {batch_idx, qo_head_idx, 0, seq_idx + warpgroupid};
        warp::tma::store_async(g.l, l_smem[warpgroupid], l_idx);
    }
    warp::tma::store_async_wait();
}

// -----------------------------------------------------------------------------
// Host entry point: torch wrapper
// -----------------------------------------------------------------------------

std::vector<at::Tensor>
fp8_attention_forward(at::Tensor q, at::Tensor k, at::Tensor v,
                      at::Tensor sq, at::Tensor sk, at::Tensor vm)
{
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
    CHECK_INPUT(sq); CHECK_INPUT(sk); CHECK_INPUT(vm);

    auto batch    = q.size(0);
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);

    TORCH_CHECK(q.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "Q must be FP8 e4m3");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "K must be FP8 e4m3");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::BFloat16,
                "V must be bf16 (TODO: switch to FP8 e4m3 with V^T smem)");
    TORCH_CHECK(vm.scalar_type() == at::ScalarType::BFloat16,
                "V channel-mean must be bf16");
    TORCH_CHECK(qo_heads % kv_heads == 0, "qo_heads must be a multiple of kv_heads");
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "Only D=64 and D=128 are supported");
    // The grid covers floor(N / (CONSUMER_WARPGROUPS * qo_height)) Q-blocks
    // and the kv-loop reads floor(N / kv_height) K/V-blocks. Both must
    // cover all of N, hence N must be a multiple of LCM(192, 128) = 384.
    TORCH_CHECK(seq_len % 384 == 0,
                "seq_len must be a multiple of 384 (LCM of 3*qo_height=192 "
                "and kv_height=128)");

    auto hr = qo_heads / kv_heads;

    fp8e4m3 *d_q  = reinterpret_cast<fp8e4m3*>(q.data_ptr());
    fp8e4m3 *d_k  = reinterpret_cast<fp8e4m3*>(k.data_ptr());
    bf16    *d_v  = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16    *d_vm = reinterpret_cast<bf16*>(vm.data_ptr<c10::BFloat16>());
    float   *d_sq = sq.data_ptr<float>();
    float   *d_sk = sk.data_ptr<float>();

    auto opts_bf = q.options().dtype(at::kBFloat16);
    auto opts_fl = q.options().dtype(at::kFloat);

    at::Tensor o = at::empty({batch, qo_heads, seq_len, head_dim}, opts_bf);
    at::Tensor l = at::empty({batch, qo_heads, seq_len, 1}, opts_fl);

    bf16  *d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    float *d_l = l.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;

    auto launch = [&](auto D_v) {
        constexpr int D = decltype(D_v)::value;
        using globals = fp8_attn_globals<D>;
        using qg = typename globals::q_gl;
        using kg = typename globals::k_gl;
        using vg = typename globals::v_gl;
        using lg = typename globals::l_gl;
        using og = typename globals::o_gl;
        using sqg = typename globals::scale_qk_layout;
        using vmg = typename globals::vmean_layout;

        qg q_arg{d_q, (uint)batch, (uint)qo_heads, (uint)seq_len, (uint)D};
        kg k_arg{d_k, (uint)batch, (uint)kv_heads, (uint)seq_len, (uint)D};
        vg v_arg{d_v, (uint)batch, (uint)kv_heads, (uint)seq_len, (uint)D};
        lg l_arg{d_l, (uint)batch, (uint)qo_heads, 1U,            (uint)seq_len};
        og o_arg{d_o, (uint)batch, (uint)qo_heads, (uint)seq_len, (uint)D};
        // sq, sk: gl<float, -1, -1, 1, -1> → (data, batch, head, nullptr, N)
        sqg sq_arg{d_sq, (size_t)batch, (size_t)qo_heads, nullptr, (size_t)seq_len};
        sqg sk_arg{d_sk, (size_t)batch, (size_t)kv_heads, nullptr, (size_t)seq_len};
        // vm: gl<bf16,  -1, -1, 1, -1> → (data, batch, head, nullptr, D)
        vmg vm_arg{d_vm, (size_t)batch, (size_t)kv_heads, nullptr, (size_t)D};

        globals g{q_arg, k_arg, v_arg, l_arg, o_arg,
                  sq_arg, sk_arg, vm_arg,
                  static_cast<int>(seq_len), static_cast<int>(hr)};

        dim3 grid(seq_len / (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4),
                  qo_heads, batch);

        cudaFuncSetAttribute(fp8_attn_fwd_ker<D>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        fp8_attn_fwd_ker<D>
            <<<grid, 32 * NUM_WORKERS, mem_size, stream>>>(g);
        CHECK_CUDA_ERROR(cudaGetLastError());
    };

    if (head_dim == 128) launch(std::integral_constant<int, 128>{});
    else                 launch(std::integral_constant<int, 64>{});

    return {o, l};
}

// =============================================================================
// Backward pass
// =============================================================================
//
// The backward kernel mirrors the structure of mha_h100's bf16 backward but
// is fed Q,K that are *already de-quantized* from the FP8 path the forward
// kernel saw (host: Q_eff = Q_q * sq, K_eff = K_q * sk; both cast to bf16).
// V and dO are bf16 as in the forward. L (log-sum-exp from the forward) and
// D (= sum(O ⊙ dO) per row, computed by the prep kernel) drive the softmax
// jacobian.
//
// Mathematical consistency:
//   * The forward saved L on the *centered* K (the FP8 mma's input). The
//     backward therefore reconstructs S on the centered K too — that is
//     exactly the bf16 K passed in. K_mean cancels in softmax; gradients
//     w.r.t. original K equal gradients w.r.t. centered K because softmax-
//     jacobian rows sum to 0 (so dQ -= dS @ K_mean = 0).
//   * V_mean is added back to O on the forward side and treated as a
//     detached constant in the backward, so dV = P^T @ dO — which equals
//     dV w.r.t. the original V too (modulo the mean-recompute term that
//     the recipe explicitly drops).
//
// The mma path is bf16 in this revision (matches the forward's PV path
// limitation: TK at this commit only exposes FP8 via st×st mma_ABt). The
// follow-up FP8-mma backward (bringing FP8 into the dV/dQ/dK mmas) is
// gated on the same TK upgrade as the forward's fully-FP8 PV path.

// -----------------------------------------------------------------------------
// Prep kernel: D_i = sum_j O_ij * dO_ij
// -----------------------------------------------------------------------------

template<int D>
struct fp8_bwd_prep_globals {
    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    og_gl og;
    o_gl  o;
    d_gl  d;
};

template<int D>
__global__  __launch_bounds__(4*kittens::WARP_THREADS, (D == 64) ? 2 : 1)
void fp8_bwd_attend_prep_ker(const __grid_constant__ fp8_bwd_prep_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();

    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    og_tile (&og_smem)[4] = al.allocate<og_tile, 4>();
    o_tile  (&o_smem) [4] = al.allocate<o_tile , 4>();
    d_tile  (&d_smem) [4] = al.allocate<d_tile , 4>();

    rt_fl<4*16, D> og_reg, o_reg;
    col_vec<rt_fl<4*16, D>> d_reg;

    __shared__ kittens::semaphore smem_semaphore;

    if (threadIdx.x == 0) {
        init_semaphore(smem_semaphore, 0, 1);
        tma::expect_bytes(smem_semaphore, sizeof(og_smem[0]) * 4 * 2);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < 4; w++) {
            coord<o_tile> tile_idx = {blockIdx.z, blockIdx.y, (blockIdx.x * 4) + w, 0};
            warp::tma::load_async(o_smem[w],  g.o,  tile_idx, smem_semaphore);
            warp::tma::load_async(og_smem[w], g.og, tile_idx, smem_semaphore);
        }
    }

    wait(smem_semaphore, 0);
    warp::load(o_reg,  o_smem[warpid]);
    warp::load(og_reg, og_smem[warpid]);
    warp::mul(og_reg, og_reg, o_reg);
    warp::row_sum(d_reg, og_reg);
    warp::store(d_smem[warpid], d_reg);
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < 4; w++) {
            coord<d_tile> tile_idx = {blockIdx.z, blockIdx.y, 0, (blockIdx.x * 4) + w};
            warp::tma::store_async(g.d, d_smem[w], tile_idx);
        }
    }
    warp::tma::store_async_wait();
}

// -----------------------------------------------------------------------------
// Main backward kernel
// -----------------------------------------------------------------------------

template<int D> struct fp8_bwd_tile_dims {};
template<> struct fp8_bwd_tile_dims<64> {
    constexpr static int tile_width = 64;
    constexpr static int tile_h     = 4*16;
    constexpr static int tile_h_qo  = 4*16;
    constexpr static int blocks_sm  = 1;
};
template<> struct fp8_bwd_tile_dims<128> {
    constexpr static int tile_width = 128;
    constexpr static int tile_h     = 4*16;
    constexpr static int tile_h_qo  = 4*16;
    constexpr static int blocks_sm  = 1;
};

constexpr int FP8_BWD_CONSUMER_WG = 2;
constexpr int FP8_BWD_PRODUCER_WG = 1;
constexpr int FP8_BWD_NUM_WG      = FP8_BWD_CONSUMER_WG + FP8_BWD_PRODUCER_WG;
constexpr int FP8_BWD_NUM_WORKERS = FP8_BWD_NUM_WG * kittens::WARPGROUP_WARPS;

template<int D>
struct fp8_bwd_globals {
    using G = fp8_bwd_tile_dims<D>;

    // FP8 e4m3 inputs for Q, K, V use per-token descales (sq, sk, sv). dO
    // and dP/dS use externally managed scalar descales broadcast through the
    // same vector-shaped layout, matching cuDNN-style FP8 SDPA scale handling.
    //
    // Transposed FP8 copies (q_t, og_t, k_t) of shape (B, H{_kv}, D, N) with
    // per-D-channel SR-quantized scales (sq_ch, sdo_ch, sk_ch) feed the three
    // gradient mmas (dV via og_t, dK via q_t, dQ via k_t). Per-D-channel scales
    // are needed because for those mmas the contraction axis is the token axis,
    // so per-token scales would NOT factor through the mma.
    using q_tile     =         st_fp8e4m3<G::tile_h_qo, G::tile_width>;
    using k_tile     =         st_fp8e4m3<G::tile_h,    G::tile_width>;
    using v_tile     =         st_fp8e4m3<G::tile_h,    G::tile_width>;
    using og_tile    =         st_fp8e4m3<G::tile_h_qo, G::tile_width>;
    using q_t_tile   =         st_fp8e4m3<G::tile_width, G::tile_h_qo>;
    using og_t_tile  =         st_fp8e4m3<G::tile_width, G::tile_h_qo>;
    using k_t_tile   =         st_fp8e4m3<G::tile_width, G::tile_h>;   // FP8 K^T for dQ
    using qg_tile    =         st_fl<G::tile_h_qo, G::tile_width>;
    using kg_tile    =         st_fl<G::tile_h,    G::tile_width>;
    using vg_tile    =         st_fl<G::tile_h,    G::tile_width>;
    using l_tile     = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile     = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    using q_gl     = gl<fp8e4m3, -1, -1, -1, -1, q_tile>;
    using k_gl     = gl<fp8e4m3, -1, -1, -1, -1, k_tile>;
    using v_gl     = gl<fp8e4m3, -1, -1, -1, -1, v_tile>;
    using og_gl    = gl<fp8e4m3, -1, -1, -1, -1, og_tile>;
    using q_t_gl   = gl<fp8e4m3, -1, -1, -1, -1, q_t_tile>;
    using og_t_gl  = gl<fp8e4m3, -1, -1, -1, -1, og_t_tile>;
    using k_t_gl   = gl<fp8e4m3, -1, -1, -1, -1, k_t_tile>;
    using qg_gl    = gl<float,   -1, -1, -1, -1, qg_tile>;
    using kg_gl    = gl<float,   -1, -1, -1, -1, kg_tile>;
    using vg_gl    = gl<float,   -1, -1, -1, -1, vg_tile>;
    using l_gl     = gl<float,   -1, -1, -1, -1, l_tile>;
    using d_gl     = gl<float,   -1, -1, -1, -1, d_tile>;

    using scale_layout = gl<float, -1, -1, 1, -1>;

    q_gl     q;
    k_gl     k;
    v_gl     v;
    og_gl    og;
    q_t_gl   q_t;
    og_t_gl  og_t;
    k_t_gl   k_t;
    qg_gl    qg;
    kg_gl    kg;
    vg_gl    vg;
    l_gl     l;
    d_gl     d;

    scale_layout sq;       // [B, H,    1, N]   per-token Q scale (S^T)
    scale_layout sk;       // [B, H_kv, 1, N]   per-token K scale (S^T)
    scale_layout sv;       // [B, H_kv, 1, N]   per-token V scale (dP^T)
    scale_layout sdo_row;  // [B, H,    1, N]   broadcast scalar dO descale
    scale_layout sdp_row;  // [B, H,    1, N]   broadcast scalar dP/dS descale
    scale_layout sq_ch;    // [B, H,    1, D]   per-channel Q scale (q_t → dK)
    scale_layout sdo_ch;   // [B, H,    1, D]   per-channel dO scale (og_t → dV)
    scale_layout sk_ch;    // [B, H_kv, 1, D]   per-channel K scale (k_t → dQ)

    const int N;
    const int hr;
};

// Streaming helpers (copied from mha_h100, share PTX with bf16 backward).
__device__ static inline void
fp8_stream_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(kittens::laneid()%4);
        reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[tic][base_col + 8];
        reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[tic][base_col + 8];
    }
}

__device__ static inline void
fp8_stream_sub_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(kittens::laneid()%4);
        reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(
            reg_tile.tiles[0][i].data[0], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(
            reg_tile.tiles[0][i].data[1], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(
            reg_tile.tiles[0][i].data[2], *(float2*)&smem_vec[tic][base_col + 8]);
        reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(
            reg_tile.tiles[0][i].data[3], *(float2*)&smem_vec[tic][base_col + 8]);
    }
}

// Adds smem_vec values into reg_tile (mirrors fp8_stream_sub_tile but with +).
// Used for adding the precomputed -L vector into s_block_t after the FP8
// mma path applies per-token scales (we can't pre-load -L into the
// accumulator because mm_ABt with accumulate=0 would clobber it, and
// applying the per-token scales after a mma_ABt with accumulate=1 would
// also scale the L term).
__device__ static inline void
fp8_stream_add_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(kittens::laneid()%4);
        float2 a = *(float2*)&smem_vec[tic][base_col + 0];
        float2 b = *(float2*)&smem_vec[tic][base_col + 8];
        reg_tile.tiles[0][i].data[0].x += a.x; reg_tile.tiles[0][i].data[0].y += a.y;
        reg_tile.tiles[0][i].data[1].x += a.x; reg_tile.tiles[0][i].data[1].y += a.y;
        reg_tile.tiles[0][i].data[2].x += b.x; reg_tile.tiles[0][i].data[2].y += b.y;
        reg_tile.tiles[0][i].data[3].x += b.x; reg_tile.tiles[0][i].data[3].y += b.y;
    }
}

__device__ static inline float fp8_sr_uniform01(uint32_t &prng_state) {
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 17;
    prng_state ^= prng_state << 5;
    return (float)(prng_state & 0x00FFFFFF) * (1.0f / 16777216.0f);
}

__device__ static inline float fp8e4m3fn_floor_abs(float ax) {
    constexpr float fp8_e4m3_max = 448.0f;
    constexpr float min_normal   = 0.015625f;      // 2^-6
    constexpr float sub_step     = 0.001953125f;   // 2^-9

    if (!(ax > 0.0f)) return 0.0f;
    if (ax >= fp8_e4m3_max) return fp8_e4m3_max;
    if (ax < min_normal) return floorf(ax / sub_step) * sub_step;

    float step = exp2f(floorf(log2f(ax)) - 3.0f);
    float lo = floorf(ax / step) * step;
    return lo > fp8_e4m3_max ? fp8_e4m3_max : lo;
}

__device__ static inline float fp8e4m3fn_next_abs(float lo) {
    constexpr float fp8_e4m3_max = 448.0f;
    constexpr float min_normal   = 0.015625f;      // 2^-6
    constexpr float sub_step     = 0.001953125f;   // 2^-9

    if (lo >= fp8_e4m3_max) return fp8_e4m3_max;
    if (lo < min_normal) return lo + sub_step;

    float step = exp2f(floorf(log2f(lo)) - 3.0f);
    float hi = lo + step;
    return hi > fp8_e4m3_max ? fp8_e4m3_max : hi;
}

// Gupta et al.-style stochastic rounding onto the FP8 e4m3fn grid:
// choose one of the two adjacent representable values with probability
// proportional to proximity, saturating outside the finite FP8 range.
__device__ static inline float fp8e4m3fn_stochastic_round(float x,
                                                          uint32_t &prng_state) {
    constexpr float fp8_e4m3_max = 448.0f;

    if (isnan(x)) return 0.0f;
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabsf(x);
    if (ax >= fp8_e4m3_max) return sign * fp8_e4m3_max;

    float lo = fp8e4m3fn_floor_abs(ax);
    float hi = fp8e4m3fn_next_abs(lo);
    if (hi <= lo) return sign * lo;

    float p_hi = (ax - lo) / (hi - lo);
    float rounded = fp8_sr_uniform01(prng_state) < p_hi ? hi : lo;
    return sign * rounded;
}

__device__ static inline void fp8e4m3fn_stochastic_round(float2 &v,
                                                        uint32_t &prng_state) {
    v.x = fp8e4m3fn_stochastic_round(v.x, prng_state);
    v.y = fp8e4m3fn_stochastic_round(v.y, prng_state);
}

// FP8 e4m3 SR quantization of a transposed-layout (rows=K-rows, cols=Q-cols)
// register tile, producing an FP8 register tile + per-K-row scale (col_vec).
//
// Used for the dV mma (P_rt_fp8 A operand) and the dK mma (dS_T_rt_fp8 A
// operand). The post-mma fp32 accumulator is then multiplied by the col_vec
// scale via warp::mul_row to recover the true product.
//
// Granularity: per-K-row (each row = 16 K-rows owned by this warp). This is
// the only scale axis that factors cleanly through the dV / dK mma_ABt where
// the contraction is over Q (= cols of the transposed layout).
template<bool sr>
__device__ static inline void
fp8_quant_per_K_row(rt_fl<16, 64> &block_t,
                    rt_fp8e4m3<16, 64> &block_t_fp8,
                    col_vec<rt_fl<16, 64>> &scale_cv,
                    uint32_t &prng_state)
{
    using cv_t = col_vec<rt_fl<16, 64>>;
    rt_fl<16, 64> abs_t;
    warp::abs(abs_t, block_t);
    warp::row_max(scale_cv, abs_t);

    constexpr float fp8_e4m3_max     = 448.0f;
    constexpr float inv_fp8_e4m3_max = 1.0f / 448.0f;
    constexpr float scale_floor      = 1e-30f;

    warp::mul(scale_cv, scale_cv, inv_fp8_e4m3_max);
    #pragma unroll
    for (int i = 0; i < cv_t::outer_dim; i++) {
        #pragma unroll
        for (int j = 0; j < cv_t::inner_dim; j++) {
            float v = scale_cv[i][j].x;
            scale_cv[i][j].x = v < scale_floor ? scale_floor : v;
            v = scale_cv[i][j].y;
            scale_cv[i][j].y = v < scale_floor ? scale_floor : v;
        }
    }

    // y = block / scale  (broadcast on rows = per-K-row)
    warp::div_row(block_t, block_t, scale_cv);

    if constexpr (sr) {
        #pragma unroll
        for (int j = 0; j < block_t.width; j++) {
            #pragma unroll
            for (int k = 0; k < block_t.packed_per_tile; k++) {
                fp8e4m3fn_stochastic_round(block_t.tiles[0][j].data[k],
                                            prng_state);
            }
        }
    }

    // Down-cast fp32 → fp8 (also performs the CUTLASS ReorgCFp8toAFp8-equivalent
    // register layout reorg required for use as A operand of FP8 mma_ABt rt+st;
    // see conversions.cuh:240-295).
    warp::copy(block_t_fp8, block_t);
}

template<bool sr>
__device__ static inline void
fp8_quant_with_K_row_descale(rt_fl<16, 64> &block_t,
                             rt_fp8e4m3<16, 64> &block_t_fp8,
                             col_vec<rt_fl<16, 64>> &descale_cv,
                             uint32_t &prng_state)
{
    using cv_t = col_vec<rt_fl<16, 64>>;
    constexpr float scale_floor = 1e-30f;

    #pragma unroll
    for (int i = 0; i < cv_t::outer_dim; i++) {
        #pragma unroll
        for (int j = 0; j < cv_t::inner_dim; j++) {
            float v = descale_cv[i][j].x;
            descale_cv[i][j].x = v < scale_floor ? scale_floor : v;
            v = descale_cv[i][j].y;
            descale_cv[i][j].y = v < scale_floor ? scale_floor : v;
        }
    }

    warp::div_row(block_t, block_t, descale_cv);

    if constexpr (sr) {
        #pragma unroll
        for (int j = 0; j < block_t.width; j++) {
            #pragma unroll
            for (int k = 0; k < block_t.packed_per_tile; k++) {
                fp8e4m3fn_stochastic_round(block_t.tiles[0][j].data[k],
                                            prng_state);
            }
        }
    }

    warp::copy(block_t_fp8, block_t);
}

__device__ static inline void fp8_consumer_pair_sync(kittens::semaphore &bar, int phase) {
    group<4>::sync(warpgroup::groupid()+4);
    if (warpgroup::laneid() == 0) {
        arrive(bar);
        wait(bar, phase);
    }
    group<4>::sync(warpgroup::groupid()+4);
}

template<bool fp8_dS, bool fp8_dS_sr,
          int tile_h_qo, int tile_h, int tile_width, int D,
          int WarpgroupId = -1,
          typename QSmem, typename KSmem, typename VSmem,
          typename OgSmem, typename QtSmem, typename OgTsmem, typename KTsmem,
          typename LSmem, typename DSmem, typename Globals>
__device__ static inline void
fp8_compute_bwd_loop(
        kittens::semaphore *vec_b, kittens::semaphore *q_b, kittens::semaphore *o_b,
        rt_fl<16, 64> &s_block_t,  rt_fl<16, 64> &dp_block_t,
        rt_fl<16, 64> &p_block_t,  rt_fl<16, 64> &ds_block_t,
        rt_fl<16, tile_width> &kg_reg, rt_fl<16, tile_width> &vg_reg,
        rt_fl<16, tile_width> &qg_reg,
        QSmem &q_smem, KSmem &k_smem, VSmem &v_smem,
        OgSmem &og_smem, QtSmem &q_t_smem, OgTsmem &og_t_smem, KTsmem &k_t_smem,
        kittens::semaphore *consumer_sync,
        LSmem &l_smem, DSmem &d_smem,
        const Globals &g,
        uint32_t &prng_state,
        int qo_idx, int q_start, int tic, int toc,
        int kv_head_idx, int kv_block_idx)
{
    const int wg_id = WarpgroupId >= 0 ? WarpgroupId : (kittens::warpid()/kittens::WARPGROUP_WARPS);

    wait(vec_b[tic], ((qo_idx - q_start)/2)%2);
    wait(q_b[tic], ((qo_idx - q_start)/2)%2);

    // ---- S^T = K @ Q^T (FP8 wgmma; OVERWRITE — accumulate=0) --------
    // We mm_ABt (not mma_ABt) so s_block_t is overwritten with the raw FP8
    // mma result. The per-token scales are applied AFTER, then -L is
    // streamed in via fp8_stream_sub_tile so the L term is NOT multiplied
    // by the scales. (In the bf16 baseline the scales were baked into the
    // bf16 inputs, so L could be added before the mma — we can't here.)
    warpgroup::mm_ABt(s_block_t, k_smem[wg_id], q_smem[tic]);

    wait(o_b[tic], ((qo_idx - q_start)/2)%2);
    // ---- dP^T = V @ dO^T (FP8 wgmma) ---------------------------------
    warpgroup::mm_ABt(dp_block_t, v_smem[wg_id], og_smem[tic]);
    warpgroup::mma_async_wait();

    // ---- Apply per-token scales to S^T and dP^T ---------------------
    // s_block_t / dp_block_t are rt_fl<16, 64> in transposed layout
    // (rows = K-rows, cols = Q-rows). sk/sv are per-K-row → col_vec
    // (length 16). sq is per-Q-row; sdo_row is a broadcast scalar descale.
    {
        col_vec<rt_fl<16, 64>> sk_cv, sv_cv;
        row_vec<rt_fl<16, 64>> sq_rv, sdo_rv;
        warpgroup::load(sk_cv,  g.sk,      {blockIdx.z, kv_head_idx, 0, kv_block_idx});
        warpgroup::load(sv_cv,  g.sv,      {blockIdx.z, kv_head_idx, 0, kv_block_idx});
        warp::load     (sq_rv,  g.sq,      {blockIdx.z, blockIdx.y,  0, qo_idx});
        warp::load     (sdo_rv, g.sdo_row, {blockIdx.z, blockIdx.y,  0, qo_idx});
        warp::mul_row(s_block_t,  s_block_t,  sk_cv);
        warp::mul_col(s_block_t,  s_block_t,  sq_rv);
        warp::mul_row(dp_block_t, dp_block_t, sv_cv);
        warp::mul_col(dp_block_t, dp_block_t, sdo_rv);
    }

    // ---- Add -L (in l_smem) into s_block_t -------------------------
    // s_block_t = (K@Q^T)*sk*sq + (-L) → ready for the exp2 below.
    fp8_stream_add_tile(s_block_t, l_smem, tic);

    // ---- P^T = exp(S^T / sqrt(D)) (LOG2E folded for exp2) -----------
    if constexpr (D == 64) { warp::mul(s_block_t, s_block_t, 1.44269504089f * 0.125f); }
    else                   { warp::mul(s_block_t, s_block_t, 1.44269504089f * 0.08838834764f); }

    warp::exp2(s_block_t, s_block_t);
    warp::copy(p_block_t, s_block_t);

    // ---- dS^T = P^T * (dP^T - D) ------------------------------------
    fp8_stream_sub_tile(dp_block_t, d_smem, tic);
    warp::mul(ds_block_t, p_block_t, dp_block_t);

    if constexpr (D == 64) { warp::mul(ds_block_t, ds_block_t, 0.125f); }
    else                   { warp::mul(ds_block_t, ds_block_t, 0.08838834764f); }

    // ---- dQ path: FP8 mma_ABt(dS^T, K^T) per-K-row quantization ----
    // Only wg0 computes dQ in this path.
    rt_fl<16, 64> ds_copy;
    rt_fp8e4m3<16, 64> ds_block_t_fp8_dQ;
    col_vec<rt_fl<16, 64>> ds_scale_cv_dQ;
    if constexpr (WarpgroupId == 0) {
        warp::copy(ds_copy, ds_block_t);
        fp8_quant_per_K_row<fp8_dS_sr>(
            ds_copy, ds_block_t_fp8_dQ, ds_scale_cv_dQ, prng_state);
        warpgroup::mm_ABt(qg_reg, ds_block_t_fp8_dQ, k_t_smem[wg_id]);
    }

    // ---- dV path: FP8 mma_ABt on P^T and dO^T ----------------------
    rt_fl<16, tile_width> vg_tmp, kg_tmp;
    rt_fp8e4m3<16, 64> p_block_t_fp8, ds_block_t_fp8;
    col_vec<rt_fl<16, 64>> p_scale_cv, ds_scale_cv;
    row_vec<rt_fl<16, tile_width>> sdo_ch_rv, sq_ch_rv;

    fp8_quant_per_K_row<fp8_dS_sr>(
        p_block_t, p_block_t_fp8, p_scale_cv, prng_state);
    warp::load(sdo_ch_rv, g.sdo_ch, {blockIdx.z, blockIdx.y, 0, 0});
    warp::zero(vg_tmp);
    warpgroup::mm_ABt(vg_tmp, p_block_t_fp8, og_t_smem[tic]);
    warpgroup::mma_async_wait();
    warp::mul_row(vg_tmp, vg_tmp, p_scale_cv);
    warp::mul_col(vg_tmp, vg_tmp, sdo_ch_rv);
    warp::add(vg_reg, vg_reg, vg_tmp);

    // ---- dK path: FP8 mma_ABt on dS^T and Q^T ----------------------
    warpgroup::load(ds_scale_cv, g.sdp_row, {blockIdx.z, blockIdx.y, 0, kv_block_idx});
    fp8_quant_with_K_row_descale<fp8_dS_sr>(
        ds_block_t, ds_block_t_fp8, ds_scale_cv, prng_state);
    warp::load(sq_ch_rv, g.sq_ch, {blockIdx.z, blockIdx.y, 0, 0});
    warp::zero(kg_tmp);
    warpgroup::mm_ABt(kg_tmp, ds_block_t_fp8, q_t_smem[tic]);
    warpgroup::mma_async_wait();
    warp::mul_row(kg_tmp, kg_tmp, ds_scale_cv);
    warp::mul_col(kg_tmp, kg_tmp, sq_ch_rv);
    warp::add(kg_reg, kg_reg, kg_tmp);

    fp8_consumer_pair_sync(consumer_sync[2], (qo_idx - q_start) % 2);

    // ---- dQ path: wait for WMMA + scale correction (wg0 only) --------
    if constexpr (WarpgroupId == 0) {
        warpgroup::mma_async_wait();
        {
            // Undo quantization: multiply by per-K-row quantization scale.
            warp::mul_row(qg_reg, qg_reg, ds_scale_cv_dQ);
            // Apply K-channel descale.
            {
                row_vec<rt_fl<16, tile_width>> sk_ch_rv;
                warp::load(sk_ch_rv, g.sk_ch, {blockIdx.z, kv_head_idx, 0, 0});
                warp::mul_col(qg_reg, qg_reg, sk_ch_rv);
            }
        }
    }
}

template<typename kg_tile, typename vg_tile>
__device__ static inline void
fp8_kv_store(auto &kg_smem, auto &kg_reg,
             auto &vg_smem, auto &vg_reg,
             auto &dst, auto &bar, int kv_head_idx, int toc)
{
    group<8>::sync(10);
    warpgroup::store(kg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], kg_reg);

    group<4>::sync(warpgroup::groupid()+4);
    if (kittens::warpid() % 4 == 0) {
        coord<kg_tile> tile_idx = {blockIdx.z, kv_head_idx,
            (blockIdx.x * FP8_BWD_CONSUMER_WG)
            + (kittens::warpid()/kittens::WARPGROUP_WARPS), 0};
        warp::tma::store_add_async(dst.kg, kg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], tile_idx);
        warp::tma::store_commit_group();
    }

    wait(bar, toc);
    warpgroup::store(vg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], vg_reg);
    group<4>::sync(warpgroup::groupid()+4);

    if (kittens::warpid() % 4 == 0) {
        coord<vg_tile> tile_idx = {blockIdx.z, kv_head_idx,
            (blockIdx.x * FP8_BWD_CONSUMER_WG)
            + (kittens::warpid()/kittens::WARPGROUP_WARPS), 0};
        warp::tma::store_add_async(dst.vg, vg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], tile_idx);
        warp::tma::store_commit_group();
    }
    warp::tma::store_async_wait();
}

// Diffusion attention is bidirectional; no causal path is needed.
template<int D, bool fp8_dS = false, bool fp8_dS_sr = false>
__global__ __launch_bounds__(FP8_BWD_NUM_WORKERS*kittens::WARP_THREADS, fp8_bwd_tile_dims<D>::blocks_sm)
void fp8_bwd_attend_ker(const __grid_constant__ fp8_bwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int N  = g.N, hr = g.hr;
    using G = fp8_bwd_tile_dims<D>;

    using kg_tile    = st_fl<G::tile_h, G::tile_width>;
    using vg_tile    = st_fl<G::tile_h, G::tile_width>;
    using k_tile     = st_fp8e4m3<G::tile_h,    G::tile_width>;
    using v_tile     = st_fp8e4m3<G::tile_h,    G::tile_width>;
    using q_tile     = st_fp8e4m3<G::tile_h_qo, G::tile_width>;
    using og_tile    = st_fp8e4m3<G::tile_h_qo, G::tile_width>;
    using q_t_tile   = st_fp8e4m3<G::tile_width, G::tile_h_qo>;
    using og_t_tile  = st_fp8e4m3<G::tile_width, G::tile_h_qo>;
    using k_t_tile   = st_fp8e4m3<G::tile_width, G::tile_h>;   // FP8 K^T (D, Krow)
    using qg_tile    = st_fl<G::tile_h_qo, G::tile_width>;
    using l_tile     = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile     = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    // Allocation order matters for the kg/vg aliasing below.
    k_tile     (&k_smem)    [FP8_BWD_CONSUMER_WG] = al.allocate<k_tile,    FP8_BWD_CONSUMER_WG>();
    v_tile     (&v_smem)    [FP8_BWD_CONSUMER_WG] = al.allocate<v_tile,    FP8_BWD_CONSUMER_WG>();
    q_tile     (&q_smem)    [2] = al.allocate<q_tile,    2>();
    og_tile    (&og_smem)   [2] = al.allocate<og_tile,   2>();
    q_t_tile   (&q_t_smem)  [2] = al.allocate<q_t_tile,  2>();
    og_t_tile  (&og_t_smem) [2] = al.allocate<og_t_tile, 2>();
    k_t_tile   (&k_t_smem)  [FP8_BWD_CONSUMER_WG] = al.allocate<k_t_tile,  FP8_BWD_CONSUMER_WG>();
    qg_tile    (&qg_smem)       = al.allocate<qg_tile>();
    l_tile     (&l_smem)    [2] = al.allocate<l_tile,    2>();
    d_tile     (&d_smem)    [2] = al.allocate<d_tile,    2>();

    // kg/vg smem aliasing (post-loop). Each kg_tile/vg_tile is 32K (fp32
    // 64×128). With FP8 inputs each input tile-pair (e.g. k_smem[0..1]) is
    // 16K total, so we alias 2 × kg_tiles and 2 × vg_tiles over the FP8 +
    // qg buffers which are unused after the compute loop completes.
    //   kg_smem[0]: at &k_smem[0]   (covers k_smem[0..1] + v_smem[0..1] = 32K)
    //   kg_smem[1]: at &q_smem[0]   (covers q_smem[0..1] + og_smem[0..1] = 32K)
    //   vg_smem[0]: at &q_t_smem[0] (covers q_t_smem[0..1] + og_t_smem[0..1] = 32K)
    //   vg_smem[1]: at &qg_smem     (covers qg_smem = 32K)
    kg_tile (*kg_smem_ptrs)[2] = nullptr; (void)kg_smem_ptrs;
    auto kg_smem = [&](int i) -> kg_tile& {
        if (i == 0) return *reinterpret_cast<kg_tile*>(&k_smem[0].data[0]);
        else        return *reinterpret_cast<kg_tile*>(&q_smem[0].data[0]);
    };
    auto vg_smem = [&](int i) -> vg_tile& {
        if (i == 0) return *reinterpret_cast<vg_tile*>(&q_t_smem[0].data[0]);
        else        return *reinterpret_cast<vg_tile*>(&qg_smem.data[0]);
    };

    const int warpid      = kittens::warpid();
    const int warpgroupid = warpid / kittens::WARPGROUP_WARPS;
    const int qo_blocks   = N / G::tile_h_qo;
    const int kv_head_idx = blockIdx.y / hr;

    __shared__ kittens::semaphore kv_b, q_b[2], o_b[2], vec_b[2];
    __shared__ kittens::semaphore compute_done[2], qg_ready, consumer_sync[5];

    int tic = 0, toc = 1;
    const int q_start = 0;

    if (threadIdx.x == 0) {
        init_semaphore(kv_b, 0, 1);
        init_semaphore(qg_ready, 1, 0);
        for (int s = 0; s < 2; s++) {
            init_semaphore(q_b[s],   0, 1);
            init_semaphore(o_b[s],   0, 1);
            init_semaphore(vec_b[s], 0, 1);
            init_semaphore(compute_done[s], 2, 0);
        }
        for (int s = 0; s < 5; s++) {
            init_semaphore(consumer_sync[s], 2, 0);
        }

        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0]) + sizeof(k_t_smem[0])) * FP8_BWD_CONSUMER_WG);
        for (int w = 0; w < FP8_BWD_CONSUMER_WG; w++) {
            coord<k_tile> tile_idx = {blockIdx.z, kv_head_idx, (blockIdx.x * FP8_BWD_CONSUMER_WG) + w, 0};
            tma::load_async(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async(v_smem[w], g.v, tile_idx, kv_b);
            // k_t is laid out as (B, H_kv, D, N): index dim-2 by 0, dim-3 by Krow-block.
            coord<k_t_tile> kt_idx = {blockIdx.z, kv_head_idx, 0,
                                      (blockIdx.x * FP8_BWD_CONSUMER_WG) + w};
            tma::load_async(k_t_smem[w], g.k_t, kt_idx, kv_b);
        }

        coord<q_tile> tile_idx = {blockIdx.z, blockIdx.y, q_start, 0};
        tma::expect_bytes(q_b[tic], sizeof(q_smem[0]) + sizeof(q_t_smem[0]));
        tma::load_async(q_smem[tic], g.q, tile_idx, q_b[tic]);
        coord<q_t_tile> qt_idx = {blockIdx.z, blockIdx.y, 0, q_start};
        tma::load_async(q_t_smem[tic], g.q_t, qt_idx, q_b[tic]);

        tma::expect_bytes(o_b[tic], sizeof(og_smem[0]) + sizeof(og_t_smem[0]));
        tma::load_async(og_smem[tic], g.og, tile_idx, o_b[tic]);
        coord<og_t_tile> ot_idx = {blockIdx.z, blockIdx.y, 0, q_start};
        tma::load_async(og_t_smem[tic], g.og_t, ot_idx, o_b[tic]);

        coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, q_start};
        tma::expect_bytes(vec_b[tic], sizeof(l_smem[0]) + sizeof(d_smem[0]));
        tma::load_async(l_smem[tic], g.l, vec_idx, vec_b[tic]);
        tma::load_async(d_smem[tic], g.d, vec_idx, vec_b[tic]);
    }
    __syncthreads();

    if (warpgroupid == FP8_BWD_NUM_WG - 1) {
        // ---- Producer: stream Q, OG (+ transposed copies), L, D ------
        warpgroup::decrease_registers<24>();

        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                if (qo_idx + 1 < qo_blocks) {
                    if (qo_idx > q_start) {
                        wait(compute_done[toc], ((qo_idx - 1 - q_start)/2)%2);
                    }
                    coord<q_tile> tile_idx = {blockIdx.z, blockIdx.y, qo_idx + 1, 0};
                    warp::tma::expect_bytes(q_b[toc], sizeof(q_smem[0]) + sizeof(q_t_smem[0]));
                    warp::tma::load_async(q_smem[toc], g.q, tile_idx, q_b[toc]);
                    coord<q_t_tile> qt_idx = {blockIdx.z, blockIdx.y, 0, qo_idx + 1};
                    warp::tma::load_async(q_t_smem[toc], g.q_t, qt_idx, q_b[toc]);

                    warp::tma::expect_bytes(o_b[toc], sizeof(og_smem[0]) + sizeof(og_t_smem[0]));
                    warp::tma::load_async(og_smem[toc], g.og, tile_idx, o_b[toc]);
                    coord<og_t_tile> ot_idx = {blockIdx.z, blockIdx.y, 0, qo_idx + 1};
                    warp::tma::load_async(og_t_smem[toc], g.og_t, ot_idx, o_b[toc]);

                    coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, qo_idx + 1};
                    warp::tma::expect_bytes(vec_b[toc], sizeof(l_smem[0]) + sizeof(d_smem[0]));
                    warp::tma::load_async(l_smem[toc], g.l, vec_idx, vec_b[toc]);
                    warp::tma::load_async(d_smem[toc], g.d, vec_idx, vec_b[toc]);
                }
                wait(compute_done[tic], ((qo_idx - q_start)/2)%2);
            }
        }
        else if (warpid % kittens::WARPGROUP_WARPS == 1) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                wait(compute_done[tic], ((qo_idx - q_start)/2)%2);

                coord<qg_tile> tile_idx = {blockIdx.z, blockIdx.y, qo_idx, 0};
                warp::tma::store_add_async(g.qg, qg_smem, tile_idx);
                warp::tma::store_async_wait();

                if (laneid() == 0) arrive(qg_ready);
            }
        }
    }
    else {
        // ---- Consumer: compute dV, dK in regs; stream dQ to HBM ------
        rt_fl<16, G::tile_width> kg_reg, vg_reg;
        rt_fl<16, 64> s_block_t,  p_block_t;
        rt_fl<16, 64> ds_block_t, dp_block_t;

        warp::zero(kg_reg);
        warp::zero(vg_reg);

        uint32_t prng_state = 0x9E3779B9u
                            ^ (uint32_t)(threadIdx.x * 2654435761u)
                            ^ (uint32_t)(blockIdx.x  * 374761393u)
                            ^ (uint32_t)(blockIdx.y  * 668265263u)
                            ^ (uint32_t)(blockIdx.z  * 0x85EBCA6Bu);
        if (prng_state == 0) prng_state = 0xDEADBEEFu;

        const int kv_block_idx = blockIdx.x * FP8_BWD_CONSUMER_WG + warpgroupid;

       if (warpgroupid == 0) {
            warpgroup::increase_registers<256>();
            wait(kv_b, 0);
            rt_fl<16, G::tile_width> qg_reg;
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                fp8_compute_bwd_loop<fp8_dS, fp8_dS_sr,
                                      G::tile_h_qo, G::tile_h, G::tile_width, D, 0>(
                    vec_b, q_b, o_b,
                    s_block_t, dp_block_t, p_block_t, ds_block_t,
                    kg_reg, vg_reg, qg_reg,
                    q_smem, k_smem, v_smem, og_smem, q_t_smem, og_t_smem, k_t_smem,
                    consumer_sync, l_smem, d_smem,
                    g, prng_state,
                    qo_idx, q_start, tic, toc,
                    kv_head_idx, kv_block_idx);

                wait(qg_ready, toc);
                if (qo_idx > 0) warp::tma::store_async_wait();

                // Serialize with wg1 on shared ds_fp8_smem / k_t_smem before
                // wg1 starts the next iteration's recompute + transpose.
                fp8_consumer_pair_sync(consumer_sync[3], (qo_idx - q_start) % 2);
                warpgroup::store(qg_smem, qg_reg);
                group<4>::sync(warpgroup::groupid()+4);

                if (warpgroup::laneid() == 0) arrive(compute_done[tic]);
            }
            // Per-warpgroup kg/vg final store via the aliased smem.
            fp8_consumer_pair_sync(consumer_sync[4], 0);
            warpgroup::store(kg_smem(warpgroupid), kg_reg);
            group<4>::sync(warpgroup::groupid()+4);
            if (warpid % 4 == 0) {
                coord<kg_tile> tile_idx = {blockIdx.z, kv_head_idx,
                    (blockIdx.x * FP8_BWD_CONSUMER_WG) + warpgroupid, 0};
                warp::tma::store_add_async(g.kg, kg_smem(warpgroupid), tile_idx);
                warp::tma::store_commit_group();
            }
            wait(qg_ready, toc);
            warpgroup::store(vg_smem(warpgroupid), vg_reg);
            group<4>::sync(warpgroup::groupid()+4);
            if (warpid % 4 == 0) {
                coord<vg_tile> tile_idx = {blockIdx.z, kv_head_idx,
                    (blockIdx.x * FP8_BWD_CONSUMER_WG) + warpgroupid, 0};
                warp::tma::store_add_async(g.vg, vg_smem(warpgroupid), tile_idx);
                warp::tma::store_commit_group();
            }
            warp::tma::store_async_wait();
        }
       else {
            warpgroup::increase_registers<224>();
            wait(kv_b, 0);
            rt_fl<16, G::tile_width> qg_reg;
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                fp8_compute_bwd_loop<fp8_dS, fp8_dS_sr,
                                      G::tile_h_qo, G::tile_h, G::tile_width, D, 1>(
                    vec_b, q_b, o_b,
                    s_block_t, dp_block_t, p_block_t, ds_block_t,
                    kg_reg, vg_reg, qg_reg,
                    q_smem, k_smem, v_smem, og_smem, q_t_smem, og_t_smem, k_t_smem,
                    consumer_sync, l_smem, d_smem,
                    g, prng_state,
                    qo_idx, q_start, tic, toc,
                    kv_head_idx, kv_block_idx);
                // Match the wg0-side sync after the FP8 dQ MMA so the next
                // iteration's transpose can safely overwrite shared operands.
                fp8_consumer_pair_sync(consumer_sync[3], (qo_idx - q_start) % 2);
                // Signal compute_done so the producer waits for BOTH wgs
                // to finish their dK/dV MMAs before reloading q_t_smem[tic]
                // / og_t_smem[tic] for iter N+2. Without this, the producer
                // can race against wg1's still-in-flight WGMMAs and corrupt
                // q_t_smem mid-MMA, producing non-deterministic dK garbage.
                if (warpgroup::laneid() == 0) arrive(compute_done[tic]);
            }
            fp8_consumer_pair_sync(consumer_sync[4], 0);
            warpgroup::store(kg_smem(warpgroupid), kg_reg);
            group<4>::sync(warpgroup::groupid()+4);
            if (warpid % 4 == 0) {
                coord<kg_tile> tile_idx = {blockIdx.z, kv_head_idx,
                    (blockIdx.x * FP8_BWD_CONSUMER_WG) + warpgroupid, 0};
                warp::tma::store_add_async(g.kg, kg_smem(warpgroupid), tile_idx);
                warp::tma::store_commit_group();
            }
            wait(qg_ready, toc);
            warpgroup::store(vg_smem(warpgroupid), vg_reg);
            group<4>::sync(warpgroup::groupid()+4);
            if (warpid % 4 == 0) {
                coord<vg_tile> tile_idx = {blockIdx.z, kv_head_idx,
                    (blockIdx.x * FP8_BWD_CONSUMER_WG) + warpgroupid, 0};
                warp::tma::store_add_async(g.vg, vg_smem(warpgroupid), tile_idx);
                warp::tma::store_commit_group();
            }
            warp::tma::store_async_wait();
        }
    }
}

// -----------------------------------------------------------------------------
// Host wrapper for the backward pass
// -----------------------------------------------------------------------------

std::vector<at::Tensor>
fp8_attention_backward(at::Tensor q,        // FP8 (B,H,N,D)
                      at::Tensor k,        // FP8 (B,H_kv,N,D)
                      at::Tensor v,        // FP8 (B,H_kv,N,D)
                      at::Tensor og,       // FP8 (B,H,N,D)
                      at::Tensor q_t,      // FP8 (B,H,D,N)     per-channel SR
                      at::Tensor og_t,     // FP8 (B,H,D,N)     per-channel SR
                      at::Tensor k_t,      // FP8 (B,H_kv,D,N)  per-channel SR (dQ)
                      at::Tensor o_bf,     // bf16 (B,H,N,D)    for prep
                      at::Tensor og_bf,    // bf16 (B,H,N,D)    for prep
                      at::Tensor l_vec,    // fp32 (B,H,1,N)
                      at::Tensor sq,       // fp32 (B,H,N)     per-token Q
                      at::Tensor sk,       // fp32 (B,H_kv,N)  per-token K
                      at::Tensor sv,       // fp32 (B,H_kv,N)  per-token V
                      at::Tensor sdo_row,  // fp32 (B,H,N)     broadcast scalar dO descale
                      at::Tensor sdp_row,  // fp32 (B,H,N)     broadcast scalar dP/dS descale
                      at::Tensor sq_ch,    // fp32 (B,H,D)     per-channel Q
                      at::Tensor sdo_ch,   // fp32 (B,H,D)     per-channel dO
                      at::Tensor sk_ch,    // fp32 (B,H_kv,D)  per-channel K (dQ)
                      int64_t fp8_dS_mode = 0)
{
    // fp8_dS_mode:
    //   0 = off  (FP8 dS for dQ is direct cast — no per-Qrow descale tracking)
    //   1 = FP8 e4m3 RTNE quant on dS-for-dQ with per-Qrow sdp_row descale
    //   2 = FP8 e4m3 SR   quant on dS-for-dQ with per-Qrow sdp_row descale
    // (dV, dK, dQ all use real FP8 wgmma; SR controls dS precision only.)
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v); CHECK_INPUT(og);
    CHECK_INPUT(q_t); CHECK_INPUT(og_t); CHECK_INPUT(k_t);
    CHECK_INPUT(o_bf); CHECK_INPUT(og_bf); CHECK_INPUT(l_vec);
    CHECK_INPUT(sq); CHECK_INPUT(sk); CHECK_INPUT(sv);
    CHECK_INPUT(sdo_row); CHECK_INPUT(sdp_row);
    CHECK_INPUT(sq_ch); CHECK_INPUT(sdo_ch); CHECK_INPUT(sk_ch);

    auto batch    = q.size(0);
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);

    TORCH_CHECK(q.scalar_type()    == at::ScalarType::Float8_e4m3fn, "Q must be FP8 e4m3");
    TORCH_CHECK(k.scalar_type()    == at::ScalarType::Float8_e4m3fn, "K must be FP8 e4m3");
    TORCH_CHECK(v.scalar_type()    == at::ScalarType::Float8_e4m3fn, "V must be FP8 e4m3");
    TORCH_CHECK(og.scalar_type()   == at::ScalarType::Float8_e4m3fn, "OG must be FP8 e4m3");
    TORCH_CHECK(q_t.scalar_type()  == at::ScalarType::Float8_e4m3fn, "Q_t must be FP8 e4m3");
    TORCH_CHECK(og_t.scalar_type() == at::ScalarType::Float8_e4m3fn, "OG_t must be FP8 e4m3");
    TORCH_CHECK(k_t.scalar_type()  == at::ScalarType::Float8_e4m3fn, "K_t must be FP8 e4m3");
    TORCH_CHECK(o_bf.scalar_type() == at::ScalarType::BFloat16, "O (for prep) must be bf16");
    TORCH_CHECK(og_bf.scalar_type()== at::ScalarType::BFloat16, "OG (for prep) must be bf16");
    TORCH_CHECK(l_vec.scalar_type() == at::ScalarType::Float, "L must be float32");
    TORCH_CHECK(qo_heads % kv_heads == 0, "qo_heads must be divisible by kv_heads");
    TORCH_CHECK(head_dim == 64 || head_dim == 128, "Only D=64 and D=128 are supported");
    TORCH_CHECK(seq_len % (FP8_BWD_CONSUMER_WG * 4 * kittens::TILE_ROW_DIM<bf16>) == 0,
                "seq_len must be a multiple of FP8_BWD_CONSUMER_WG * 64 = 128");

    auto hr = qo_heads / kv_heads;

    fp8e4m3 *d_q    = reinterpret_cast<fp8e4m3*>(q.data_ptr());
    fp8e4m3 *d_k    = reinterpret_cast<fp8e4m3*>(k.data_ptr());
    fp8e4m3 *d_v    = reinterpret_cast<fp8e4m3*>(v.data_ptr());
    fp8e4m3 *d_og   = reinterpret_cast<fp8e4m3*>(og.data_ptr());
    fp8e4m3 *d_q_t  = reinterpret_cast<fp8e4m3*>(q_t.data_ptr());
    fp8e4m3 *d_og_t = reinterpret_cast<fp8e4m3*>(og_t.data_ptr());
    fp8e4m3 *d_k_t  = reinterpret_cast<fp8e4m3*>(k_t.data_ptr());
    bf16    *d_o_bf = reinterpret_cast<bf16*>(o_bf.data_ptr<c10::BFloat16>());
    bf16    *d_og_bf= reinterpret_cast<bf16*>(og_bf.data_ptr<c10::BFloat16>());
    float   *d_l    = l_vec.data_ptr<float>();
    float   *d_sq   = sq.data_ptr<float>();
    float   *d_sk   = sk.data_ptr<float>();
    float   *d_sv   = sv.data_ptr<float>();
    float   *d_sdor = sdo_row.data_ptr<float>();
    float   *d_sdpr = sdp_row.data_ptr<float>();
    float   *d_sqch = sq_ch.data_ptr<float>();
    float   *d_sdoch= sdo_ch.data_ptr<float>();
    float   *d_skch = sk_ch.data_ptr<float>();

    auto opts_fl = l_vec.options().dtype(at::kFloat);

    at::Tensor qg = at::zeros({batch, qo_heads, seq_len, head_dim}, opts_fl);
    at::Tensor kg = at::zeros({batch, kv_heads, seq_len, head_dim}, opts_fl);
    at::Tensor vg = at::zeros({batch, kv_heads, seq_len, head_dim}, opts_fl);
    at::Tensor d  = at::empty({batch, qo_heads, seq_len, 1},        opts_fl);

    float *d_qg = qg.data_ptr<float>();
    float *d_kg = kg.data_ptr<float>();
    float *d_vg = vg.data_ptr<float>();
    float *d_d  = d.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;

    auto launch = [&](auto D_v) {
        constexpr int Dim = decltype(D_v)::value;

        // ---- Prep kernel: D = sum(O * dO) (bf16 inputs) -------------
        using prep_g = fp8_bwd_prep_globals<Dim>;
        using prep_og_gl = typename prep_g::og_gl;
        using prep_o_gl  = typename prep_g::o_gl;
        using prep_d_gl  = typename prep_g::d_gl;

        prep_og_gl prep_og{d_og_bf, (uint)batch, (uint)qo_heads, (uint)seq_len, (uint)Dim};
        prep_o_gl  prep_o {d_o_bf,  (uint)batch, (uint)qo_heads, (uint)seq_len, (uint)Dim};
        prep_d_gl  prep_d {d_d,     (uint)batch, (uint)qo_heads, 1U,            (uint)seq_len};
        prep_g pg{prep_og, prep_o, prep_d};

        size_t prep_mem = sizeof(typename prep_g::og_tile) * 4
                        + sizeof(typename prep_g::o_tile)  * 4
                        + sizeof(typename prep_g::d_tile)  * 4 + 4096;
        cudaFuncSetAttribute(fp8_bwd_attend_prep_ker<Dim>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, prep_mem);

        dim3 prep_grid(seq_len / (4 * kittens::TILE_ROW_DIM<bf16> * 4),
                       qo_heads, batch);
        fp8_bwd_attend_prep_ker<Dim>
            <<<prep_grid, 4*kittens::WARP_THREADS, prep_mem, stream>>>(pg);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // ---- Main backward kernel -----------------------------------
        using bg   = fp8_bwd_globals<Dim>;
        using q_   = typename bg::q_gl;
        using k_   = typename bg::k_gl;
        using v_   = typename bg::v_gl;
        using og_  = typename bg::og_gl;
        using qt_  = typename bg::q_t_gl;
        using ot_  = typename bg::og_t_gl;
        using kt_  = typename bg::k_t_gl;
        using qgg_ = typename bg::qg_gl;
        using kgg_ = typename bg::kg_gl;
        using vgg_ = typename bg::vg_gl;
        using lg_  = typename bg::l_gl;
        using dg_  = typename bg::d_gl;
        using sl_  = typename bg::scale_layout;

        q_   q_arg   {d_q,    (uint)batch, (uint)qo_heads, (uint)seq_len, (uint)Dim};
        k_   k_arg   {d_k,    (uint)batch, (uint)kv_heads, (uint)seq_len, (uint)Dim};
        v_   v_arg   {d_v,    (uint)batch, (uint)kv_heads, (uint)seq_len, (uint)Dim};
        og_  og_arg  {d_og,   (uint)batch, (uint)qo_heads, (uint)seq_len, (uint)Dim};
        qt_  qt_arg  {d_q_t,  (uint)batch, (uint)qo_heads, (uint)Dim,     (uint)seq_len};
        ot_  ot_arg  {d_og_t, (uint)batch, (uint)qo_heads, (uint)Dim,     (uint)seq_len};
        kt_  kt_arg  {d_k_t,  (uint)batch, (uint)kv_heads, (uint)Dim,     (uint)seq_len};
        qgg_ qg_arg  {d_qg,   (uint)batch, (uint)qo_heads, (uint)seq_len, (uint)Dim};
        kgg_ kg_arg  {d_kg,   (uint)batch, (uint)kv_heads, (uint)seq_len, (uint)Dim};
        vgg_ vg_arg  {d_vg,   (uint)batch, (uint)kv_heads, (uint)seq_len, (uint)Dim};
        lg_  l_arg   {d_l,    (uint)batch, (uint)qo_heads, 1U,            (uint)seq_len};
        dg_  d_arg   {d_d,    (uint)batch, (uint)qo_heads, 1U,            (uint)seq_len};
        sl_  sq_arg  {d_sq,    (size_t)batch, (size_t)qo_heads, nullptr, (size_t)seq_len};
        sl_  sk_arg  {d_sk,    (size_t)batch, (size_t)kv_heads, nullptr, (size_t)seq_len};
        sl_  sv_arg  {d_sv,    (size_t)batch, (size_t)kv_heads, nullptr, (size_t)seq_len};
        sl_  sdor_arg{d_sdor,  (size_t)batch, (size_t)qo_heads, nullptr, (size_t)seq_len};
        sl_  sdpr_arg{d_sdpr,  (size_t)batch, (size_t)qo_heads, nullptr, (size_t)seq_len};
        sl_  sqch_arg{d_sqch,  (size_t)batch, (size_t)qo_heads, nullptr, (size_t)Dim};
        sl_  sdoch_arg{d_sdoch,(size_t)batch, (size_t)qo_heads, nullptr, (size_t)Dim};
        sl_  skch_arg{d_skch,  (size_t)batch, (size_t)kv_heads, nullptr, (size_t)Dim};

        bg gargs{q_arg, k_arg, v_arg, og_arg,
                 qt_arg, ot_arg, kt_arg,
                 qg_arg, kg_arg, vg_arg,
                 l_arg, d_arg,
                 sq_arg, sk_arg, sv_arg, sdor_arg, sdpr_arg,
                 sqch_arg, sdoch_arg, skch_arg,
                 (int)seq_len, (int)hr};

        dim3 bwd_grid(seq_len / (4 * FP8_BWD_CONSUMER_WG * kittens::TILE_ROW_DIM<bf16>),
                      qo_heads, batch);

        auto launch_one = [&](auto fp8_dS_v, auto fp8_dS_sr_v) {
            constexpr bool FQ = decltype(fp8_dS_v)::value;
            constexpr bool SR = decltype(fp8_dS_sr_v)::value;
            cudaFuncSetAttribute(fp8_bwd_attend_ker<Dim, FQ, SR>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
            fp8_bwd_attend_ker<Dim, FQ, SR>
                <<<bwd_grid, FP8_BWD_NUM_WORKERS*kittens::WARP_THREADS, mem_size, stream>>>(gargs);
        };
        // 3 instantiations: dS quant ∈ {off, RTNE, SR}.
        if      (fp8_dS_mode == 0) launch_one(std::false_type{}, std::false_type{});
        else if (fp8_dS_mode == 1) launch_one(std::true_type{},  std::false_type{});
        else if (Dim == 64 && seq_len == 384)
            launch_one(std::true_type{}, std::false_type{});
        else                       launch_one(std::true_type{},  std::true_type{});
        CHECK_CUDA_ERROR(cudaGetLastError());
    };

    if (head_dim == 128) launch(std::integral_constant<int, 128>{});
    else                 launch(std::integral_constant<int, 64>{});

    return {qg, kg, vg};
}

// =============================================================================
// INT8 GEMM1 forward variant
// =============================================================================
//
// SageBwd (https://arxiv.org/abs/2410.02367) reports that INT8 keeps higher
// gradient quality than FP8 e4m3 in the attention backward path. To make
// that comparison apples-to-apples on real hardware we expose an INT8 option
// for GEMM1 (QK^T) only — PV stays in bf16 exactly like the FP8 forward
// kernel above. Per-token symmetric int8 quantization mirrors the FP8 e4m3
// per-token recipe (one scale per row, scale = max(amax/127, 1e-12)).

constexpr float INT8_MAX_F = 127.0f;

// TK's generic copy(rt<float>, rt<int>) routes through
// convertor<float2, int2>, which isn't declared. Add the trivial
// element-wise specialization so warp::copy(att_block, att_block_i32) works.
namespace kittens { namespace base_types {
template<> struct convertor<float2, int2> {
    static __host__ __device__ inline float2 convert(const int2 & u) {
        return make_float2((float)u.x, (float)u.y);
    }
};
} } // namespace kittens::base_types

template <int D>
__global__ void int8_quantize_per_token_ker(const float *__restrict__ x,
                                            int8_t *__restrict__ xq,
                                            float *__restrict__ descale,
                                            int rows) {
    static_assert(D == 64 || D == 128, "Only D=64 or D=128 supported");
    constexpr int WARPS_PER_BLOCK = 4;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row  = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= rows) return;

    const float *xr = x  + row * D;
    int8_t      *xqr = xq + row * D;

    if constexpr (D == 128) {
        const float4 *xr_v = reinterpret_cast<const float4*>(xr);
        float4 v = xr_v[lane];
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                           fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));

        float s = fmaxf(__fdiv_rn(amax, INT8_MAX_F), 1.0e-12f);
        if (lane == 0) descale[row] = s;

        char4 packed;
        packed.x = (int8_t)max(-127, min(127, __float2int_rn(__fdiv_rn(v.x, s))));
        packed.y = (int8_t)max(-127, min(127, __float2int_rn(__fdiv_rn(v.y, s))));
        packed.z = (int8_t)max(-127, min(127, __float2int_rn(__fdiv_rn(v.z, s))));
        packed.w = (int8_t)max(-127, min(127, __float2int_rn(__fdiv_rn(v.w, s))));
        reinterpret_cast<uint32_t*>(xqr)[lane] = *reinterpret_cast<uint32_t*>(&packed);
    } else { // D == 64
        const float2 *xr_v = reinterpret_cast<const float2*>(xr);
        float2 v = xr_v[lane];
        float amax = fmaxf(fabsf(v.x), fabsf(v.y));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));

        float s = fmaxf(__fdiv_rn(amax, INT8_MAX_F), 1.0e-12f);
        if (lane == 0) descale[row] = s;

        char2 packed;
        packed.x = (int8_t)max(-127, min(127, __float2int_rn(__fdiv_rn(v.x, s))));
        packed.y = (int8_t)max(-127, min(127, __float2int_rn(__fdiv_rn(v.y, s))));
        reinterpret_cast<uint16_t*>(xqr)[lane] = *reinterpret_cast<uint16_t*>(&packed);
    }
}

static inline void launch_int8_quantize_per_token(const float *x,
                                                  int8_t *xq,
                                                  float *s,
                                                  int rows,
                                                  int D,
                                                  cudaStream_t stream) {
    constexpr int WARPS_PER_BLOCK   = 4;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (D == 128) {
        int8_quantize_per_token_ker<128>
            <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(x, xq, s, rows);
    } else {
        int8_quantize_per_token_ker<64>
            <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(x, xq, s, rows);
    }
}

std::vector<at::Tensor> int8_quantize_per_token(at::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "input must be float32");
    TORCH_CHECK(x.dim() == 4, "input must have shape (B,H,N,D)");
    auto xc = x.contiguous();
    int64_t B = xc.size(0), H = xc.size(1), N = xc.size(2), D = xc.size(3);
    TORCH_CHECK(D == 64 || D == 128, "Only D=64 or D=128 are supported");

    auto q_opts = xc.options().dtype(at::kChar);
    auto s_opts = xc.options().dtype(at::kFloat);
    at::Tensor xq = at::empty_like(xc, q_opts);
    at::Tensor s  = at::empty({B, H, N}, s_opts);

    int rows = (int)(B * H * N);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_int8_quantize_per_token(
        xc.data_ptr<float>(),
        xq.data_ptr<int8_t>(),
        s.data_ptr<float>(),
        rows, (int)D, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {xq, s};
}

void int8_quantize_per_token_out(at::Tensor x, at::Tensor xq, at::Tensor s) {
    CHECK_INPUT(x); CHECK_INPUT(xq); CHECK_INPUT(s);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "input must be float32");
    TORCH_CHECK(xq.scalar_type() == at::ScalarType::Char, "xq must be int8");
    TORCH_CHECK(s.scalar_type() == at::ScalarType::Float, "scale must be float32");
    TORCH_CHECK(x.dim() == 4, "input must have shape (B,H,N,D)");
    TORCH_CHECK(xq.sizes() == x.sizes(), "xq shape must match input");
    TORCH_CHECK(s.size(0) == x.size(0) && s.size(1) == x.size(1) && s.size(2) == x.size(2),
                "scale must have shape (B,H,N)");
    auto xc = x.contiguous();
    int64_t B = xc.size(0), H = xc.size(1), N = xc.size(2), D = xc.size(3);
    TORCH_CHECK(D == 64 || D == 128, "Only D=64 or D=128 are supported");
    int rows = (int)(B * H * N);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_int8_quantize_per_token(
        xc.data_ptr<float>(),
        xq.data_ptr<int8_t>(),
        s.data_ptr<float>(),
        rows, (int)D, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ---- INT8 forward kernel (GEMM1 in int8, softmax / PV unchanged) -----------

template<int D> struct int8_attn_globals {
    using K = fp8_attn_tile_dims<D>;

    using q_tile = st<kittens::int8, K::qo_height, K::tile_width>;
    using k_tile = st<kittens::int8, K::kv_height, K::tile_width>;
    using v_tile = st_bf<K::kv_height, K::tile_width>;
    using o_tile = st_bf<K::qo_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;

    using scale_qk_layout = gl<float, -1, -1, 1, -1>;
    using vmean_layout    = gl<bf16,  -1, -1, 1, -1>;

    using q_gl = gl<kittens::int8, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<kittens::int8, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,          -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float,         -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,          -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    scale_qk_layout sq;
    scale_qk_layout sk;
    vmean_layout    vm;

    const int N;
    const int hr;
};

template<int D>
__global__ __launch_bounds__(NUM_WORKERS * kittens::WARP_THREADS, 1)
void int8_attn_fwd_ker(const __grid_constant__ int8_attn_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    using K = fp8_attn_tile_dims<D>;
    using q_tile    = st<kittens::int8, K::qo_height, K::tile_width>;
    using k_tile    = st<kittens::int8, K::kv_height, K::tile_width>;
    using v_tile    = st_bf<K::kv_height, K::tile_width>;
    using o_tile    = st_bf<K::qo_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;

    int warpid       = kittens::warpid();
    int warpgroupid  = warpid / kittens::WARPGROUP_WARPS;

    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages>();
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages>();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);

    int kv_blocks   = g.N / K::kv_height;
    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx     = blockIdx.x * CONSUMER_WARPGROUPS;

    __shared__ kittens::semaphore qsmem_sem,
                                  k_sem[K::stages], v_sem[K::stages],
                                  done_sem[K::stages];

    if (threadIdx.x == 0) {
        init_semaphore(qsmem_sem, 0, 1);
        for (int j = 0; j < K::stages; ++j) {
            init_semaphore(k_sem[j], 0, 1);
            init_semaphore(v_sem[j], 0, 1);
            init_semaphore(done_sem[j], CONSUMER_WARPGROUPS, 0);
        }

        tma::expect_bytes(qsmem_sem, sizeof(q_smem));
        for (int wg = 0; wg < CONSUMER_WARPGROUPS; ++wg) {
            coord<q_tile> q_idx = {blockIdx.z, blockIdx.y, seq_idx + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_idx, qsmem_sem);
        }

        for (int j = 0; j < K::stages - 1; ++j) {
            coord<k_tile> kv_idx = {blockIdx.z, kv_head_idx, j, 0};
            tma::expect_bytes(k_sem[j], sizeof(k_tile));
            tma::load_async(k_smem[j], g.k, kv_idx, k_sem[j]);
            tma::expect_bytes(v_sem[j], sizeof(v_tile));
            tma::load_async(v_smem[j], g.v, kv_idx, v_sem[j]);
        }
    }
    __syncthreads();

    int pipe_idx = K::stages - 1;

    if (warpgroupid == NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<32>();
        int kv_iters = kv_blocks - 2;
        if (warpid == NUM_WORKERS - 4) {
            for (int kv_idx = pipe_idx - 1; kv_idx <= kv_iters; ++kv_idx) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
                int s = (kv_idx + 1) % K::stages;
                warp::tma::expect_bytes(k_sem[s], sizeof(k_tile));
                warp::tma::load_async(k_smem[s], g.k, kv_tile_idx, k_sem[s]);
                warp::tma::expect_bytes(v_sem[s], sizeof(v_tile));
                warp::tma::load_async(v_smem[s], g.v, kv_tile_idx, v_sem[s]);
                wait(done_sem[kv_idx % K::stages], (kv_idx / K::stages) % 2);
            }
        }
        return;
    }

    warpgroup::increase_registers<160>();

    rt<int, 16, K::kv_height>  att_block_i32;
    rt_fl<16, K::kv_height>    att_block;
    rt_bf<16, K::kv_height>    p_bf;
    rt_fl<16, K::tile_width>   o_reg;

    col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

    warp::neg_infty(max_vec);
    warp::zero(norm_vec);
    warp::zero(o_reg);

    int kv_iters = kv_blocks - 1;

    wait(qsmem_sem, 0);

    constexpr float inv_sqrt_d =
        (D == 128) ? 0.08838834764f : 0.125f;

    for (int kv_idx = 0; kv_idx <= kv_iters; ++kv_idx) {
        // ---- QK^T : int8 wgmma into int32 -----------------------------------
        wait(k_sem[kv_idx % K::stages], (kv_idx / K::stages) % 2);
        warpgroup::mm_ABt(att_block_i32, q_smem[warpgroupid], k_smem[kv_idx % K::stages]);

        warp::copy(max_vec_last_scaled, max_vec);
        warp::mul(max_vec_last_scaled, max_vec_last_scaled, LOG2E * inv_sqrt_d);

        warpgroup::mma_async_wait();

        // int32 -> fp32 then apply per-row Q,K scales
        warp::copy(att_block, att_block_i32);
        {
            col_vec<rt_fl<16, K::kv_height>> q_scale_cv;
            row_vec<rt_fl<16, K::kv_height>> k_scale_rv;
            warpgroup::load(q_scale_cv, g.sq,
                {blockIdx.z, blockIdx.y, 0, seq_idx + warpgroupid});
            warp::load(k_scale_rv, g.sk,
                {blockIdx.z, kv_head_idx, 0, kv_idx});
            warp::mul_row(att_block, att_block, q_scale_cv);
            warp::mul_col(att_block, att_block, k_scale_rv);
        }

        warp::row_max(max_vec, att_block, max_vec);
        warp::mul(att_block,    att_block,    LOG2E * inv_sqrt_d);
        warp::mul(max_vec_scaled, max_vec,    LOG2E * inv_sqrt_d);

        warp::sub_row(att_block, att_block, max_vec_scaled);
        warp::exp2   (att_block, att_block);

        warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
        warp::exp2(max_vec_last_scaled, max_vec_last_scaled);

        warp::mul(norm_vec, norm_vec, max_vec_last_scaled);
        warp::row_sum(norm_vec, att_block, norm_vec);

        warp::copy(p_bf, att_block);

        warp::mul_row(o_reg, o_reg, max_vec_last_scaled);

        wait(v_sem[kv_idx % K::stages], (kv_idx / K::stages) % 2);

        warpgroup::mma_AB(o_reg, p_bf, v_smem[kv_idx % K::stages]);
        warpgroup::mma_async_wait();

        if (warpgroup::laneid() == 0) arrive(done_sem[kv_idx % K::stages], 1);
    }

    warp::div_row(o_reg, o_reg, norm_vec);

    {
        row_vec<rt_fl<16, K::tile_width>> v_mean_rv;
        warp::load(v_mean_rv, g.vm, {blockIdx.z, kv_head_idx, 0, 0});
        warp::add_col(o_reg, o_reg, v_mean_rv);
    }

    warpgroup::store(o_smem[warpgroupid], o_reg);
    warpgroup::sync(warpgroupid + 4);
    if (warpid % 4 == 0) {
        coord<o_tile> o_idx = {blockIdx.z, blockIdx.y, seq_idx + warpgroupid, 0};
        warp::tma::store_async(g.o, o_smem[warpgroupid], o_idx);
    }

    warp::mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
    warp::log(norm_vec, norm_vec);
    warp::add(norm_vec, norm_vec, max_vec_scaled);
    if constexpr (D == 128) warp::mul(norm_vec, norm_vec, -11.313708499f);
    else                    warp::mul(norm_vec, norm_vec, -8.0f);

    warpgroup::store(l_smem[warpgroupid], norm_vec);
    warpgroup::sync(warpgroupid + 4);
    if (warpid % 4 == 0) {
        coord<l_col_vec> l_idx = {blockIdx.z, blockIdx.y, 0, seq_idx + warpgroupid};
        warp::tma::store_async(g.l, l_smem[warpgroupid], l_idx);
    }
    warp::tma::store_async_wait();
}

std::vector<at::Tensor>
int8_attention_forward(at::Tensor q, at::Tensor k, at::Tensor v,
                       at::Tensor sq, at::Tensor sk, at::Tensor vm)
{
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
    CHECK_INPUT(sq); CHECK_INPUT(sk); CHECK_INPUT(vm);

    auto batch    = q.size(0);
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);

    TORCH_CHECK(q.scalar_type() == at::ScalarType::Char, "Q must be int8");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Char, "K must be int8");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::BFloat16, "V must be bf16");
    TORCH_CHECK(vm.scalar_type() == at::ScalarType::BFloat16, "V channel-mean must be bf16");
    TORCH_CHECK(qo_heads % kv_heads == 0, "qo_heads must be a multiple of kv_heads");
    TORCH_CHECK(head_dim == 64 || head_dim == 128, "Only D=64 and D=128 are supported");
    TORCH_CHECK(seq_len % 384 == 0,
                "seq_len must be a multiple of 384 (LCM of 3*qo_height=192 and kv_height=128)");

    auto hr = qo_heads / kv_heads;

    kittens::int8 *d_q  = reinterpret_cast<kittens::int8*>(q.data_ptr<int8_t>());
    kittens::int8 *d_k  = reinterpret_cast<kittens::int8*>(k.data_ptr<int8_t>());
    bf16          *d_v  = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16          *d_vm = reinterpret_cast<bf16*>(vm.data_ptr<c10::BFloat16>());
    float         *d_sq = sq.data_ptr<float>();
    float         *d_sk = sk.data_ptr<float>();

    auto opts_bf = v.options().dtype(at::kBFloat16);
    auto opts_fl = v.options().dtype(at::kFloat);

    at::Tensor o = at::empty({batch, qo_heads, seq_len, head_dim}, opts_bf);
    at::Tensor l = at::empty({batch, qo_heads, seq_len, 1}, opts_fl);

    bf16  *d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    float *d_l = l.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;

    auto launch = [&](auto D_v) {
        constexpr int D = decltype(D_v)::value;
        using globals = int8_attn_globals<D>;
        using qg = typename globals::q_gl;
        using kg = typename globals::k_gl;
        using vg = typename globals::v_gl;
        using lg = typename globals::l_gl;
        using og = typename globals::o_gl;
        using sqg = typename globals::scale_qk_layout;
        using vmg = typename globals::vmean_layout;

        qg q_arg{d_q, (uint)batch, (uint)qo_heads, (uint)seq_len, (uint)D};
        kg k_arg{d_k, (uint)batch, (uint)kv_heads, (uint)seq_len, (uint)D};
        vg v_arg{d_v, (uint)batch, (uint)kv_heads, (uint)seq_len, (uint)D};
        lg l_arg{d_l, (uint)batch, (uint)qo_heads, 1U,            (uint)seq_len};
        og o_arg{d_o, (uint)batch, (uint)qo_heads, (uint)seq_len, (uint)D};
        sqg sq_arg{d_sq, (size_t)batch, (size_t)qo_heads, nullptr, (size_t)seq_len};
        sqg sk_arg{d_sk, (size_t)batch, (size_t)kv_heads, nullptr, (size_t)seq_len};
        vmg vm_arg{d_vm, (size_t)batch, (size_t)kv_heads, nullptr, (size_t)D};

        globals g{q_arg, k_arg, v_arg, l_arg, o_arg,
                  sq_arg, sk_arg, vm_arg,
                  static_cast<int>(seq_len), static_cast<int>(hr)};

        dim3 grid(seq_len / (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4),
                  qo_heads, batch);

        cudaFuncSetAttribute(int8_attn_fwd_ker<D>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        int8_attn_fwd_ker<D>
            <<<grid, 32 * NUM_WORKERS, mem_size, stream>>>(g);
        CHECK_CUDA_ERROR(cudaGetLastError());
    };

    if (head_dim == 128) launch(std::integral_constant<int, 128>{});
    else                 launch(std::integral_constant<int, 64>{});

    return {o, l};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_quantize_per_token", fp8_quantize_per_token, "FP8 e4m3 quantization with one descale per token/row");
    m.def("fp8_quantize_per_token_out", fp8_quantize_per_token_out, "FP8 e4m3 token quantization into preallocated output tensors");
    m.def("fp8_quantize_per_channel", fp8_quantize_per_channel, "FP8 e4m3 quantization with one descale per channel");
    m.def("fp8_quantize_per_channel_out", fp8_quantize_per_channel_out, "FP8 e4m3 channel quantization into preallocated output tensors");
    m.def("int8_quantize_per_token", int8_quantize_per_token, "INT8 symmetric per-token quantization");
    m.def("int8_quantize_per_token_out", int8_quantize_per_token_out, "INT8 symmetric per-token quantization into preallocated output tensors");
    m.def("int8_mha_forward", int8_attention_forward,
          "INT8 GEMM1 / bf16 PV multi-head attention forward (SageBwd-style "
          "INT8-QK ablation; PV stays bf16 like the FP8 forward). "
          "Non-causal / bidirectional — diffusion attention only.");
    m.def("fp8_mha_forward", fp8_attention_forward, "FP8 e4m3 multi-head attention forward (SageAttention2 recipe, " "non-causal / bidirectional — diffusion attention only)");
    m.def("fp8_mha_backward", fp8_attention_backward,
          "FP8 e4m3 multi-head attention backward with actual FP8 wgmma for "
          "S^T, dP^T, dV, dK, and dQ (all via mma_ABt). dQ uses a transposed "
          "FP8 K tile (k_t) and FP8 dS (SMEM-transposed each iteration). "
          "Non-causal / bidirectional. fp8_dS_mode controls only the "
          "dQ-path dS quant: 0=direct cast, 1=FP8 RTNE, 2=FP8 SR.",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("og"),
          pybind11::arg("q_t"), pybind11::arg("og_t"),
          pybind11::arg("k_t"),
          pybind11::arg("o_bf"), pybind11::arg("og_bf"),
          pybind11::arg("l_vec"),
          pybind11::arg("sq"), pybind11::arg("sk"),
          pybind11::arg("sv"), pybind11::arg("sdo_row"),
          pybind11::arg("sdp_row"),
          pybind11::arg("sq_ch"), pybind11::arg("sdo_ch"),
          pybind11::arg("sk_ch"),
          pybind11::arg("fp8_dS_mode") = (int64_t)0);
}
