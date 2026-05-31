#include "kittens.cuh"
#include "prototype.cuh"
#include <math.h>

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

__device__ static inline float fast_tanh(float x) {
  #if defined(__CUDA_ARCH__)
    #if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
      float y;
      asm volatile ( "tanh.approx.f32 %0, %1; " : "=f"(y) : "f"(x));
      return y;
    #else
      return ::tanhf(x);
    #endif
  #else
  return std::tanh(x);
  #endif
}

using namespace kittens;

static constexpr int PERSISTENT_GRID_BLOCKS = 128;

template<typename op, kittens::ducks::sv::all SV>
__device__ static inline void rt_sv_op(rt_fl<16, SV::length> &acc, const SV &vec) {
    #pragma unroll
    for (int i = 0; i < SV::tiles; i++) {
        float2 v0 = __bfloat1622float2(*(bf16_2*)&vec.data[16 * i + 0 + 2 * (laneid() % 4)]);
        acc.tiles[0][i].data[0] = op::template op<float2>(acc.tiles[0][i].data[0], v0);
        acc.tiles[0][i].data[1] = op::template op<float2>(acc.tiles[0][i].data[1], v0);
        float2 v1 = __bfloat1622float2(*(bf16_2*)&vec.data[16 * i + 8 + 2 * (laneid() % 4)]);
        acc.tiles[0][i].data[2] = op::template op<float2>(acc.tiles[0][i].data[2], v1);
        acc.tiles[0][i].data[3] = op::template op<float2>(acc.tiles[0][i].data[3], v1);
    }
}

template<typename op, kittens::ducks::st::all ST>
__device__ static inline void wg_rt_sv_op(rt_fl<16, ST::cols> &acc, const ST &tile) {
    static_assert(ST::rows == 64);
    #pragma unroll
    for (int i = 0; i < ST::cols / 16; i++) {
        acc.tiles[0][i].data[0] = op::template op<float2>(
            acc.tiles[0][i].data[0],
            __bfloat1622float2(*(bf16_2*)&tile[{warpgroup::warpid() * 16 + 0 + laneid() / 4, 16 * i + 0 + 2 * (laneid() % 4)}]));
        acc.tiles[0][i].data[1] = op::template op<float2>(
            acc.tiles[0][i].data[1],
            __bfloat1622float2(*(bf16_2*)&tile[{warpgroup::warpid() * 16 + 8 + laneid() / 4, 16 * i + 0 + 2 * (laneid() % 4)}]));
        acc.tiles[0][i].data[2] = op::template op<float2>(
            acc.tiles[0][i].data[2],
            __bfloat1622float2(*(bf16_2*)&tile[{warpgroup::warpid() * 16 + 0 + laneid() / 4, 16 * i + 8 + 2 * (laneid() % 4)}]));
        acc.tiles[0][i].data[3] = op::template op<float2>(
            acc.tiles[0][i].data[3],
            __bfloat1622float2(*(bf16_2*)&tile[{warpgroup::warpid() * 16 + 8 + laneid() / 4, 16 * i + 8 + 2 * (laneid() % 4)}]));
    }
}

// todo: visualze and ablate speed difference
template<kittens::ducks::sv::all SV> __device__ static inline void init_bias(rt_fl<16,SV::length> &acc, const SV &bias) {
    #pragma unroll
    for(int i = 0; i < SV::tiles; i++) {
        float2 tmp1 = __bfloat1622float2(*(bf16_2*)&bias.data[16*i + 0 + 2*(laneid()%4)]);
        acc.tiles[0][i].data[0].x = tmp1.x;
        acc.tiles[0][i].data[0].y = tmp1.y;
        acc.tiles[0][i].data[1].x = tmp1.x;
        acc.tiles[0][i].data[1].y = tmp1.y;
        float2 tmp2 = __bfloat1622float2(*(bf16_2*)&bias.data[16*i + 8 + 2*(laneid()%4)]);
        acc.tiles[0][i].data[2].x = tmp2.x;
        acc.tiles[0][i].data[2].y = tmp2.y;
        acc.tiles[0][i].data[3].x = tmp2.x;
        acc.tiles[0][i].data[3].y = tmp2.y;
    }
}

template<int WIDTH>
__device__ static inline void apply_gelu(rt_fl<16, WIDTH> &acc) {
    #pragma unroll
    for(int i = 0; i < acc.width; i++) {
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            float f = acc.tiles[0][i].data[j].x, g = acc.tiles[0][i].data[j].y;
            acc.tiles[0][i].data[j].x = f * 0.5f * (1.0f + fast_tanh(f * 0.79788456f * (1.f + f * f * 0.044715f)));
            acc.tiles[0][i].data[j].y = g * 0.5f * (1.0f + fast_tanh(g * 0.79788456f * (1.f + g * g * 0.044715f)));
        }
    }
}

using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  bias_vec       = sv_bf<64*N_BLOCK>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    using  bias_global    = gl<bf16, 1, 1, 1, -1, bias_vec>;
    struct globals        { global_layout A, B, C, preact; bias_global bias; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct scratch_block  { bias_vec bias; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};
template<int M_BLOCK, int N_BLOCK>
struct modulated_matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  bias_vec       = sv_bf<64*N_BLOCK>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    using  bias_global    = gl<bf16, 1, 1, 1, -1, bias_vec>;
    struct globals        {
        global_layout A, B, C, preact;
        bias_global bias;
        const bf16 *shift;
        const bf16 *scale;
        int tokens_per_sample;
        int K;
    };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct scratch_block  { bias_vec bias; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};
template<int M_BLOCK, int N_BLOCK>
struct ln_adaln_matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  bias_vec       = sv_bf<64*N_BLOCK>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    using  bias_global    = gl<bf16, 1, 1, 1, -1, bias_vec>;
    struct globals        {
        global_layout A, B, C, preact;
        bias_global bias;
        const bf16 *shift;
        const bf16 *scale;
        const float *mean;
        const float *rstd;
        int tokens_per_sample;
        int K;
    };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct scratch_block  { bias_vec bias; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};
template<int M_BLOCK, int N_BLOCK>
struct gated_linear_layout {
    using  base_tile      = st_bf<64, 64>;
    using  wide_tile      = st_bf<64, 64*N_BLOCK>;
    using  bias_vec       = sv_bf<64*N_BLOCK>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    using  wide_global    = gl<bf16, 1, 1, -1, -1, wide_tile>;
    using  bias_global    = gl<bf16, 1, 1, 1, -1, bias_vec>;
    using  gate_global    = gl<bf16, 1, 1, -1, -1, bias_vec>;
    struct globals        {
        global_layout A, B;
        wide_global C, preact, residual;
        bias_global bias;
        gate_global gate;
        int tokens_per_sample;
    };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct scratch_block  { bias_vec bias, gate; };
    struct finish_block   { wide_tile c[M_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};
template<int M_BLOCK, int N_BLOCK>
struct gated_linear_out_layout {
    using  base_tile      = st_bf<64, 64>;
    using  wide_tile      = st_bf<64, 64*N_BLOCK>;
    using  bias_vec       = sv_bf<64*N_BLOCK>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    using  wide_global    = gl<bf16, 1, 1, -1, -1, wide_tile>;
    using  bias_global    = gl<bf16, 1, 1, 1, -1, bias_vec>;
    using  gate_global    = gl<bf16, 1, 1, -1, -1, bias_vec>;
    struct globals        {
        global_layout A, B;
        wide_global C, residual;
        bias_global bias;
        gate_global gate;
        int tokens_per_sample;
    };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct scratch_block  { bias_vec bias, gate; };
    struct finish_block   { wide_tile c[M_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};

template<ducks::st::all ST>
__device__ static inline void apply_adaln_modulation(
    ST &tile,
    const bf16 *__restrict__ shift,
    const bf16 *__restrict__ scale,
    int row_tile,
    int k_tile,
    int tokens_per_sample,
    int K
) {
    #pragma unroll
    for (int i = warpgroup::laneid(); i < ST::num_elements; i += 128) {
        int row = i / ST::cols;
        int col = i % ST::cols;
        int global_row = row_tile * ST::rows + row;
        int global_col = k_tile * ST::cols + col;
        int batch = global_row / tokens_per_sample;
        float x = __bfloat162float(tile[{row, col}]);
        float sh = __bfloat162float(shift[batch * K + global_col]);
        float sc = __bfloat162float(scale[batch * K + global_col]);
        tile[{row, col}] = __float2bfloat16(x * (1.0f + sc) + sh);
    }
    warpgroup::sync(0);
}

template<ducks::st::all ST>
__device__ static inline void apply_ln_adaln_modulation(
    ST &tile,
    const bf16 *__restrict__ shift,
    const bf16 *__restrict__ scale,
    const float *__restrict__ mean,
    const float *__restrict__ rstd,
    int row_tile,
    int k_tile,
    int tokens_per_sample,
    int K
) {
    #pragma unroll
    for (int i = warpgroup::laneid(); i < ST::num_elements; i += 128) {
        int row = i / ST::cols;
        int col = i % ST::cols;
        int global_row = row_tile * ST::rows + row;
        int global_col = k_tile * ST::cols + col;
        int batch = global_row / tokens_per_sample;
        float x = __bfloat162float(tile[{row, col}]);
        float sh = __bfloat162float(shift[batch * K + global_col]);
        float sc = __bfloat162float(scale[batch * K + global_col]);
        float z = (x - mean[global_row]) * rstd[global_row];
        tile[{row, col}] = __float2bfloat16(z * (1.0f + sc) + sh);
    }
    warpgroup::sync(0);
}

template<ducks::st::all ST>
__device__ static inline void apply_gated_residual_epilogue(
    ST &tile,
    bf16 *__restrict__ projected_out,
    const bf16 *__restrict__ residual,
    const bf16 *__restrict__ gate,
    int row_tile,
    int col_tile,
    int tokens_per_sample,
    int N
) {
    #pragma unroll
    for (int i = warpgroup::laneid(); i < ST::num_elements; i += 128) {
        int row = i / ST::cols;
        int col = i % ST::cols;
        int global_row = row_tile * ST::rows + row;
        int global_col = col_tile * 64 + col;
        int batch = global_row / tokens_per_sample;
        size_t idx = size_t(global_row) * N + global_col;
        float projected = __bfloat162float(tile[{row, col}]);
        projected_out[idx] = tile[{row, col}];
        float base = __bfloat162float(residual[idx]);
        float g = __bfloat162float(gate[batch * N + global_col]);
        tile[{row, col}] = __float2bfloat16(base + g * projected);
    }
    warpgroup::sync(0);
}




__device__ __forceinline__ float warp_sum_float(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum_128_float(float value) {
    __shared__ float warp_sums[4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    value = warp_sum_float(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 4 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        value = warp_sum_float(value);
    }
    if (threadIdx.x == 0) {
        warp_sums[0] = value;
    }
    __syncthreads();
    return warp_sums[0];
}

__device__ __forceinline__ float2 block_sum_128_float2(float2 value) {
    __shared__ float2 warp_sums[4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    value.x = warp_sum_float(value.x);
    value.y = warp_sum_float(value.y);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 4 ? warp_sums[lane] : make_float2(0.0f, 0.0f);
    if (warp == 0) {
        value.x = warp_sum_float(value.x);
        value.y = warp_sum_float(value.y);
    }
    if (threadIdx.x == 0) {
        warp_sums[0] = value;
    }
    __syncthreads();
    return warp_sums[0];
}

__global__ __launch_bounds__(128, 4) void layernorm_adaln_forward_k1024_vec2_kernel(
    bf16 *__restrict__ out,
    float *__restrict__ mean_out,
    float *__restrict__ rstd_out,
    const bf16 *__restrict__ x,
    const bf16 *__restrict__ shift,
    const bf16 *__restrict__ scale,
    int tokens_per_sample,
    float eps
) {
    constexpr int K = 1024;
    constexpr int K2 = K / 2;
    using bf16_2 = __nv_bfloat162;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int batch = row / tokens_per_sample;
    size_t pair_base = size_t(row) * K2;
    size_t param_pair_base = size_t(batch) * K2;
    const bf16_2 *__restrict__ x2 = reinterpret_cast<const bf16_2*>(x);
    const bf16_2 *__restrict__ shift2 = reinterpret_cast<const bf16_2*>(shift);
    const bf16_2 *__restrict__ scale2 = reinterpret_cast<const bf16_2*>(scale);
    bf16_2 *__restrict__ out2 = reinterpret_cast<bf16_2*>(out);

    float2 stats = make_float2(0.0f, 0.0f);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int pair_col = tid + j * 128;
        float2 v = __bfloat1622float2(x2[pair_base + pair_col]);
        stats.x += v.x + v.y;
        stats.y += v.x * v.x + v.y * v.y;
    }
    stats = block_sum_128_float2(stats);
    float mean = stats.x * (1.0f / K);
    float var = stats.y * (1.0f / K) - mean * mean;
    var = fmaxf(var, 0.0f);
    float rstd = rsqrtf(var + eps);
    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int pair_col = tid + j * 128;
        size_t pair_idx = pair_base + pair_col;
        float2 xv = __bfloat1622float2(x2[pair_idx]);
        float2 sh = __bfloat1622float2(shift2[param_pair_base + pair_col]);
        float2 sc = __bfloat1622float2(scale2[param_pair_base + pair_col]);
        float out_x = ((xv.x - mean) * rstd) * (1.0f + sc.x) + sh.x;
        float out_y = ((xv.y - mean) * rstd) * (1.0f + sc.y) + sh.y;
        out2[pair_idx] = __floats2bfloat162_rn(out_x, out_y);
    }
}

__global__ __launch_bounds__(128, 4) void layernorm_adaln_forward_k1024_vec2_persistent_kernel(
    bf16 *__restrict__ out,
    float *__restrict__ mean_out,
    float *__restrict__ rstd_out,
    const bf16 *__restrict__ x,
    const bf16 *__restrict__ shift,
    const bf16 *__restrict__ scale,
    int M,
    int tokens_per_sample,
    float eps
) {
    constexpr int K = 1024;
    constexpr int K2 = K / 2;
    using bf16_2 = __nv_bfloat162;

    int tid = threadIdx.x;
    const bf16_2 *__restrict__ x2 = reinterpret_cast<const bf16_2*>(x);
    const bf16_2 *__restrict__ shift2 = reinterpret_cast<const bf16_2*>(shift);
    const bf16_2 *__restrict__ scale2 = reinterpret_cast<const bf16_2*>(scale);
    bf16_2 *__restrict__ out2 = reinterpret_cast<bf16_2*>(out);

    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        int batch = row / tokens_per_sample;
        size_t pair_base = size_t(row) * K2;
        size_t param_pair_base = size_t(batch) * K2;

        float2 stats = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int pair_col = tid + j * 128;
            float2 v = __bfloat1622float2(x2[pair_base + pair_col]);
            stats.x += v.x + v.y;
            stats.y += v.x * v.x + v.y * v.y;
        }
        stats = block_sum_128_float2(stats);
        float mean = stats.x * (1.0f / K);
        float var = stats.y * (1.0f / K) - mean * mean;
        var = fmaxf(var, 0.0f);
        float rstd = rsqrtf(var + eps);
        if (tid == 0) {
            mean_out[row] = mean;
            rstd_out[row] = rstd;
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int pair_col = tid + j * 128;
            size_t pair_idx = pair_base + pair_col;
            float2 xv = __bfloat1622float2(x2[pair_idx]);
            float2 sh = __bfloat1622float2(shift2[param_pair_base + pair_col]);
            float2 sc = __bfloat1622float2(scale2[param_pair_base + pair_col]);
            float out_x = ((xv.x - mean) * rstd) * (1.0f + sc.x) + sh.x;
            float out_y = ((xv.y - mean) * rstd) * (1.0f + sc.y) + sh.y;
            out2[pair_idx] = __floats2bfloat162_rn(out_x, out_y);
        }
    }
}

__global__ __launch_bounds__(128, 4) void layernorm_adaln_forward_k1024_warp4_kernel(
    bf16 *__restrict__ out,
    float *__restrict__ mean_out,
    float *__restrict__ rstd_out,
    const bf16 *__restrict__ x,
    const bf16 *__restrict__ shift,
    const bf16 *__restrict__ scale,
    int M,
    int tokens_per_sample,
    float eps
) {
    constexpr int K = 1024;
    constexpr int K2 = K / 2;
    using bf16_2 = __nv_bfloat162;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * 4 + warp;
    if (row >= M) {
        return;
    }

    int batch = row / tokens_per_sample;
    size_t pair_base = size_t(row) * K2;
    size_t param_pair_base = size_t(batch) * K2;
    const bf16_2 *__restrict__ x2 = reinterpret_cast<const bf16_2*>(x);
    const bf16_2 *__restrict__ shift2 = reinterpret_cast<const bf16_2*>(shift);
    const bf16_2 *__restrict__ scale2 = reinterpret_cast<const bf16_2*>(scale);
    bf16_2 *__restrict__ out2 = reinterpret_cast<bf16_2*>(out);

    float sum = 0.0f;
    float sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int pair_col = lane + j * 32;
        float2 v = __bfloat1622float2(x2[pair_base + pair_col]);
        sum += v.x + v.y;
        sq += v.x * v.x + v.y * v.y;
    }
    sum = warp_sum_float(sum);
    sq = warp_sum_float(sq);
    sum = __shfl_sync(0xffffffff, sum, 0);
    sq = __shfl_sync(0xffffffff, sq, 0);
    float mean = sum * (1.0f / K);
    float var = sq * (1.0f / K) - mean * mean;
    var = fmaxf(var, 0.0f);
    float rstd = rsqrtf(var + eps);
    if (lane == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int pair_col = lane + j * 32;
        size_t pair_idx = pair_base + pair_col;
        float2 xv = __bfloat1622float2(x2[pair_idx]);
        float2 sh = __bfloat1622float2(shift2[param_pair_base + pair_col]);
        float2 sc = __bfloat1622float2(scale2[param_pair_base + pair_col]);
        float out_x = ((xv.x - mean) * rstd) * (1.0f + sc.x) + sh.x;
        float out_y = ((xv.y - mean) * rstd) * (1.0f + sc.y) + sh.y;
        out2[pair_idx] = __floats2bfloat162_rn(out_x, out_y);
    }
}

__global__ void layernorm_adaln_forward_kernel(
    bf16 *__restrict__ out,
    float *__restrict__ mean_out,
    float *__restrict__ rstd_out,
    const bf16 *__restrict__ x,
    const bf16 *__restrict__ shift,
    const bf16 *__restrict__ scale,
    int M,
    int K,
    int tokens_per_sample,
    float eps
) {
    extern __shared__ float smem[];
    float *sum_s = smem;
    float *sq_s = smem + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int batch = row / tokens_per_sample;

    float sum = 0.0f;
    float sq = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        float v = __bfloat162float(x[size_t(row) * K + col]);
        sum += v;
        sq += v * v;
    }
    sum_s[tid] = sum;
    sq_s[tid] = sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_s[tid] += sum_s[tid + stride];
            sq_s[tid] += sq_s[tid + stride];
        }
        __syncthreads();
    }

    float mean = sum_s[0] / K;
    float var = sq_s[0] / K - mean * mean;
    var = fmaxf(var, 0.0f);
    float rstd = rsqrtf(var + eps);
    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    for (int col = tid; col < K; col += blockDim.x) {
        size_t idx = size_t(row) * K + col;
        float xhat = (__bfloat162float(x[idx]) - mean) * rstd;
        float sh = __bfloat162float(shift[batch * K + col]);
        float sc = __bfloat162float(scale[batch * K + col]);
        out[idx] = __float2bfloat16(xhat * (1.0f + sc) + sh);
    }
}

__global__ void layernorm_stats_kernel(
    float *__restrict__ mean_out,
    float *__restrict__ rstd_out,
    const bf16 *__restrict__ x,
    int M,
    int K,
    float eps
) {
    extern __shared__ float smem[];
    float *sum_s = smem;
    float *sq_s = smem + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    float sq = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        float v = __bfloat162float(x[size_t(row) * K + col]);
        sum += v;
        sq += v * v;
    }
    sum_s[tid] = sum;
    sq_s[tid] = sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_s[tid] += sum_s[tid + stride];
            sq_s[tid] += sq_s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = sum_s[0] / K;
        float var = sq_s[0] / K - mean * mean;
        var = fmaxf(var, 0.0f);
        mean_out[row] = mean;
        rstd_out[row] = rsqrtf(var + eps);
    }
}

__global__ void layernorm_adaln_backward_kernel(
    bf16 *__restrict__ dx,
    const bf16 *__restrict__ grad,
    const bf16 *__restrict__ x,
    const bf16 *__restrict__ scale,
    const float *__restrict__ mean,
    const float *__restrict__ rstd,
    int M,
    int K,
    int tokens_per_sample
) {
    extern __shared__ float smem[];
    float *sum_dy = smem;
    float *sum_dy_xhat = smem + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int batch = row / tokens_per_sample;
    float m = mean[row];
    float rs = rstd[row];

    float local_sum = 0.0f;
    float local_sum_xhat = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        size_t idx = size_t(row) * K + col;
        float xhat = (__bfloat162float(x[idx]) - m) * rs;
        float g = __bfloat162float(grad[idx]);
        float sc = __bfloat162float(scale[batch * K + col]);
        float dnorm = g * (1.0f + sc);
        local_sum += dnorm;
        local_sum_xhat += dnorm * xhat;
    }
    sum_dy[tid] = local_sum;
    sum_dy_xhat[tid] = local_sum_xhat;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_dy[tid] += sum_dy[tid + stride];
            sum_dy_xhat[tid] += sum_dy_xhat[tid + stride];
        }
        __syncthreads();
    }

    float s1 = sum_dy[0];
    float s2 = sum_dy_xhat[0];
    float inv_k = 1.0f / K;
    for (int col = tid; col < K; col += blockDim.x) {
        size_t idx = size_t(row) * K + col;
        float xv = __bfloat162float(x[idx]);
        float xhat = (xv - m) * rs;
        float g = __bfloat162float(grad[idx]);
        float sc = __bfloat162float(scale[batch * K + col]);
        float dnorm = g * (1.0f + sc);
        float dxv = (dnorm - s1 * inv_k - xhat * s2 * inv_k) * rs;
        dx[idx] = __float2bfloat16(dxv);
    }
}

__global__ __launch_bounds__(128, 4) void layernorm_adaln_backward_k1024_warp4_kernel(
    bf16 *__restrict__ dx,
    const bf16 *__restrict__ grad,
    const bf16 *__restrict__ x,
    const bf16 *__restrict__ scale,
    const float *__restrict__ mean,
    const float *__restrict__ rstd,
    int M,
    int tokens_per_sample
) {
    constexpr int K = 1024;
    constexpr int K2 = K / 2;
    using bf16_2 = __nv_bfloat162;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * 4 + warp;
    if (row >= M) {
        return;
    }

    int batch = row / tokens_per_sample;
    float m = mean[row];
    float rs = rstd[row];
    size_t pair_base = size_t(row) * K2;
    size_t param_pair_base = size_t(batch) * K2;
    const bf16_2 *__restrict__ x2 = reinterpret_cast<const bf16_2*>(x);
    const bf16_2 *__restrict__ grad2 = reinterpret_cast<const bf16_2*>(grad);
    const bf16_2 *__restrict__ scale2 = reinterpret_cast<const bf16_2*>(scale);
    bf16_2 *__restrict__ dx2 = reinterpret_cast<bf16_2*>(dx);

    float s1 = 0.0f;
    float s2 = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int pair_col = lane + j * 32;
        size_t pair_idx = pair_base + pair_col;
        float2 xv = __bfloat1622float2(x2[pair_idx]);
        float2 gv = __bfloat1622float2(grad2[pair_idx]);
        float2 sc = __bfloat1622float2(scale2[param_pair_base + pair_col]);
        float xhat_x = (xv.x - m) * rs;
        float xhat_y = (xv.y - m) * rs;
        float dnorm_x = gv.x * (1.0f + sc.x);
        float dnorm_y = gv.y * (1.0f + sc.y);
        s1 += dnorm_x + dnorm_y;
        s2 += dnorm_x * xhat_x + dnorm_y * xhat_y;
    }
    s1 = warp_sum_float(s1);
    s2 = warp_sum_float(s2);
    s1 = __shfl_sync(0xffffffff, s1, 0);
    s2 = __shfl_sync(0xffffffff, s2, 0);

    float inv_k = 1.0f / K;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int pair_col = lane + j * 32;
        size_t pair_idx = pair_base + pair_col;
        float2 xv = __bfloat1622float2(x2[pair_idx]);
        float2 gv = __bfloat1622float2(grad2[pair_idx]);
        float2 sc = __bfloat1622float2(scale2[param_pair_base + pair_col]);
        float xhat_x = (xv.x - m) * rs;
        float xhat_y = (xv.y - m) * rs;
        float dnorm_x = gv.x * (1.0f + sc.x);
        float dnorm_y = gv.y * (1.0f + sc.y);
        float dx_x = (dnorm_x - s1 * inv_k - xhat_x * s2 * inv_k) * rs;
        float dx_y = (dnorm_y - s1 * inv_k - xhat_y * s2 * inv_k) * rs;
        dx2[pair_idx] = __floats2bfloat162_rn(dx_x, dx_y);
    }
}

__global__ void layernorm_adaln_param_backward_kernel(
    float *__restrict__ dshift,
    float *__restrict__ dscale,
    const bf16 *__restrict__ grad,
    const bf16 *__restrict__ x,
    const float *__restrict__ mean,
    const float *__restrict__ rstd,
    int K,
    int tokens_per_sample
) {
    constexpr int COLS = 16;
    constexpr int TOK_THREADS = 16;
    __shared__ float shift_sums[TOK_THREADS][COLS];
    __shared__ float scale_sums[TOK_THREADS][COLS];

    int col = blockIdx.x * COLS + threadIdx.x;
    int batch = blockIdx.y;
    int ty = threadIdx.y;

    float shift_acc = 0.0f;
    float scale_acc = 0.0f;
    if (col < K) {
        int row_base = batch * tokens_per_sample;
        for (int tok = ty; tok < tokens_per_sample; tok += TOK_THREADS) {
            int row = row_base + tok;
            size_t idx = size_t(row) * K + col;
            float xhat = (__bfloat162float(x[idx]) - mean[row]) * rstd[row];
            float g = __bfloat162float(grad[idx]);
            shift_acc += g;
            scale_acc += g * xhat;
        }
    }

    shift_sums[ty][threadIdx.x] = shift_acc;
    scale_sums[ty][threadIdx.x] = scale_acc;
    __syncthreads();

    if (ty == 0 && col < K) {
        float ds = 0.0f;
        float dc = 0.0f;
        #pragma unroll
        for (int i = 0; i < TOK_THREADS; i++) {
            ds += shift_sums[i][threadIdx.x];
            dc += scale_sums[i][threadIdx.x];
        }
        dshift[batch * K + col] = ds;
        dscale[batch * K + col] = dc;
    }
}

__global__ __launch_bounds__(512, 2) void layernorm_adaln_param_backward_k1024_cols32_kernel(
    float *__restrict__ dshift,
    float *__restrict__ dscale,
    const bf16 *__restrict__ grad,
    const bf16 *__restrict__ x,
    const float *__restrict__ mean,
    const float *__restrict__ rstd,
    int tokens_per_sample
) {
    constexpr int K = 1024;
    constexpr int COLS = 32;
    constexpr int TOK_THREADS = 16;
    __shared__ float shift_sums[TOK_THREADS][COLS];
    __shared__ float scale_sums[TOK_THREADS][COLS];

    int col = blockIdx.x * COLS + threadIdx.x;
    int batch = blockIdx.y;
    int ty = threadIdx.y;

    float shift_acc = 0.0f;
    float scale_acc = 0.0f;
    int row_base = batch * tokens_per_sample;
    for (int tok = ty; tok < tokens_per_sample; tok += TOK_THREADS) {
        int row = row_base + tok;
        size_t idx = size_t(row) * K + col;
        float xhat = (__bfloat162float(x[idx]) - mean[row]) * rstd[row];
        float g = __bfloat162float(grad[idx]);
        shift_acc += g;
        scale_acc += g * xhat;
    }

    shift_sums[ty][threadIdx.x] = shift_acc;
    scale_sums[ty][threadIdx.x] = scale_acc;
    __syncthreads();

    if (ty == 0) {
        float ds = 0.0f;
        float dc = 0.0f;
        #pragma unroll
        for (int i = 0; i < TOK_THREADS; i++) {
            ds += shift_sums[i][threadIdx.x];
            dc += scale_sums[i][threadIdx.x];
        }
        dshift[batch * K + col] = ds;
        dscale[batch * K + col] = dc;
    }
}

__global__ __launch_bounds__(512, 2) void layernorm_adaln_param_backward_k1024_cols16_tok32_kernel(
    float *__restrict__ dshift,
    float *__restrict__ dscale,
    const bf16 *__restrict__ grad,
    const bf16 *__restrict__ x,
    const float *__restrict__ mean,
    const float *__restrict__ rstd,
    int tokens_per_sample
) {
    constexpr int K = 1024;
    constexpr int COLS = 16;
    constexpr int TOK_THREADS = 32;
    __shared__ float shift_sums[TOK_THREADS][COLS];
    __shared__ float scale_sums[TOK_THREADS][COLS];

    int col = blockIdx.x * COLS + threadIdx.x;
    int batch = blockIdx.y;
    int ty = threadIdx.y;

    float shift_acc = 0.0f;
    float scale_acc = 0.0f;
    int row_base = batch * tokens_per_sample;
    for (int tok = ty; tok < tokens_per_sample; tok += TOK_THREADS) {
        int row = row_base + tok;
        size_t idx = size_t(row) * K + col;
        float xhat = (__bfloat162float(x[idx]) - mean[row]) * rstd[row];
        float g = __bfloat162float(grad[idx]);
        shift_acc += g;
        scale_acc += g * xhat;
    }

    shift_sums[ty][threadIdx.x] = shift_acc;
    scale_sums[ty][threadIdx.x] = scale_acc;
    __syncthreads();

    if (ty == 0) {
        float ds = 0.0f;
        float dc = 0.0f;
        #pragma unroll
        for (int i = 0; i < TOK_THREADS; i++) {
            ds += shift_sums[i][threadIdx.x];
            dc += scale_sums[i][threadIdx.x];
        }
        dshift[batch * K + col] = ds;
        dscale[batch * K + col] = dc;
    }
}

__global__ void adaln_modulate_backward_kernel(
    bf16 *__restrict__ dx,
    float *__restrict__ dshift,
    float *__restrict__ dscale,
    const bf16 *__restrict__ grad,
    const bf16 *__restrict__ x,
    const bf16 *__restrict__ scale,
    int K,
    int tokens_per_sample
) {
    constexpr int COLS = 16;
    constexpr int TOK_THREADS = 16;
    __shared__ float shift_sums[TOK_THREADS][COLS];
    __shared__ float scale_sums[TOK_THREADS][COLS];

    int col = blockIdx.x * COLS + threadIdx.x;
    int batch = blockIdx.y;
    int ty = threadIdx.y;

    float shift_acc = 0.0f;
    float scale_acc = 0.0f;
    if (col < K) {
        float sc = __bfloat162float(scale[batch * K + col]);
        for (int tok = ty; tok < tokens_per_sample; tok += TOK_THREADS) {
            int row = batch * tokens_per_sample + tok;
            size_t idx = size_t(row) * K + col;
            float g = __bfloat162float(grad[idx]);
            float xv = __bfloat162float(x[idx]);
            dx[idx] = __float2bfloat16(g * (1.0f + sc));
            shift_acc += g;
            scale_acc += g * xv;
        }
    }

    shift_sums[ty][threadIdx.x] = shift_acc;
    scale_sums[ty][threadIdx.x] = scale_acc;
    __syncthreads();

    if (ty == 0 && col < K) {
        float ds = 0.0f;
        float dc = 0.0f;
        #pragma unroll
        for (int i = 0; i < TOK_THREADS; i++) {
            ds += shift_sums[i][threadIdx.x];
            dc += scale_sums[i][threadIdx.x];
        }
        dshift[batch * K + col] = ds;
        dscale[batch * K + col] = dc;
    }
}

__global__ void adaln_modulate_kernel(
    bf16 *__restrict__ out,
    const bf16 *__restrict__ x,
    const bf16 *__restrict__ shift,
    const bf16 *__restrict__ scale,
    int M,
    int K,
    int tokens_per_sample
) {
    size_t idx = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    size_t total = size_t(M) * K;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        size_t i = idx + j;
        if (i >= total) return;
        int col = i % K;
        int row = i / K;
        int batch = row / tokens_per_sample;
        float v = __bfloat162float(x[i]);
        float sh = __bfloat162float(shift[batch * K + col]);
        float sc = __bfloat162float(scale[batch * K + col]);
        out[i] = __float2bfloat16(v * (1.0f + sc) + sh);
    }
}

__global__ void gated_residual_forward_kernel(
    bf16 *__restrict__ out,
    const bf16 *__restrict__ x,
    const bf16 *__restrict__ h,
    const bf16 *__restrict__ gate,
    int M,
    int K,
    int tokens_per_sample
) {
    size_t idx = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    size_t total = size_t(M) * K;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        size_t i = idx + j;
        if (i >= total) return;
        int col = i % K;
        int row = i / K;
        int batch = row / tokens_per_sample;
        float xv = __bfloat162float(x[i]);
        float hv = __bfloat162float(h[i]);
        float gv = __bfloat162float(gate[batch * K + col]);
        out[i] = __float2bfloat16(xv + gv * hv);
    }
}

__global__ void gated_residual_forward_vec2_kernel(
    bf16 *__restrict__ out,
    const bf16 *__restrict__ x,
    const bf16 *__restrict__ h,
    const bf16 *__restrict__ gate,
    int M,
    int K,
    int tokens_per_sample
) {
    using bf16_2 = __nv_bfloat162;
    size_t pair_idx = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    size_t total_pairs = size_t(M) * K / 2;
    const bf16_2 *__restrict__ x2 = reinterpret_cast<const bf16_2*>(x);
    const bf16_2 *__restrict__ h2 = reinterpret_cast<const bf16_2*>(h);
    const bf16_2 *__restrict__ gate2 = reinterpret_cast<const bf16_2*>(gate);
    bf16_2 *__restrict__ out2 = reinterpret_cast<bf16_2*>(out);
    int K2 = K / 2;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        size_t p = pair_idx + j;
        if (p >= total_pairs) return;
        int row = p / K2;
        int col_pair = p - size_t(row) * K2;
        int batch = row / tokens_per_sample;
        float2 xv = __bfloat1622float2(x2[p]);
        float2 hv = __bfloat1622float2(h2[p]);
        float2 gv = __bfloat1622float2(gate2[size_t(batch) * K2 + col_pair]);
        out2[p] = __floats2bfloat162_rn(xv.x + gv.x * hv.x, xv.y + gv.y * hv.y);
    }
}

__global__ void gated_residual_backward_kernel(
    bf16 *__restrict__ dx,
    bf16 *__restrict__ dh,
    float *__restrict__ dgate,
    const bf16 *__restrict__ grad,
    const bf16 *__restrict__ h,
    const bf16 *__restrict__ gate,
    int K,
    int tokens_per_sample
) {
    constexpr int COLS = 16;
    constexpr int TOK_THREADS = 16;
    __shared__ float gate_sums[TOK_THREADS][COLS];

    int col = blockIdx.x * COLS + threadIdx.x;
    int batch = blockIdx.y;
    int ty = threadIdx.y;

    float acc = 0.0f;
    if (col < K) {
        float gv = __bfloat162float(gate[batch * K + col]);
        int row_base = batch * tokens_per_sample;
        for (int tok = ty; tok < tokens_per_sample; tok += TOK_THREADS) {
            int row = row_base + tok;
            size_t idx = size_t(row) * K + col;
            float go = __bfloat162float(grad[idx]);
            float hv = __bfloat162float(h[idx]);
            dx[idx] = __float2bfloat16(go);
            dh[idx] = __float2bfloat16(go * gv);
            acc += go * hv;
        }
    }

    gate_sums[ty][threadIdx.x] = acc;
    __syncthreads();

    if (ty == 0 && col < K) {
        float dg = 0.0f;
        #pragma unroll
        for (int i = 0; i < TOK_THREADS; i++) {
            dg += gate_sums[i][threadIdx.x];
        }
        dgate[batch * K + col] = dg;
    }
}

__global__ void gated_residual_backward_no_dx_kernel(
    bf16 *__restrict__ dh,
    float *__restrict__ dgate,
    const bf16 *__restrict__ grad,
    const bf16 *__restrict__ h,
    const bf16 *__restrict__ gate,
    int K,
    int tokens_per_sample
) {
    constexpr int COLS = 16;
    constexpr int TOK_THREADS = 16;
    __shared__ float gate_sums[TOK_THREADS][COLS];

    int col = blockIdx.x * COLS + threadIdx.x;
    int batch = blockIdx.y;
    int ty = threadIdx.y;

    float acc = 0.0f;
    if (col < K) {
        float gv = __bfloat162float(gate[batch * K + col]);
        int row_base = batch * tokens_per_sample;
        for (int tok = ty; tok < tokens_per_sample; tok += TOK_THREADS) {
            int row = row_base + tok;
            size_t idx = size_t(row) * K + col;
            float go = __bfloat162float(grad[idx]);
            float hv = __bfloat162float(h[idx]);
            dh[idx] = __float2bfloat16(go * gv);
            acc += go * hv;
        }
    }

    gate_sums[ty][threadIdx.x] = acc;
    __syncthreads();

    if (ty == 0 && col < K) {
        float dg = 0.0f;
        #pragma unroll
        for (int i = 0; i < TOK_THREADS; i++) {
            dg += gate_sums[i][threadIdx.x];
        }
        dgate[batch * K + col] = dg;
    }
}

__global__ void gated_residual_backward_no_dx_db_kernel(
    bf16 *__restrict__ dh,
    float *__restrict__ dgate,
    float *__restrict__ dbias,
    const bf16 *__restrict__ grad,
    const bf16 *__restrict__ h,
    const bf16 *__restrict__ gate,
    int K,
    int tokens_per_sample
) {
    constexpr int COLS = 16;
    constexpr int TOK_THREADS = 16;
    __shared__ float gate_sums[TOK_THREADS][COLS];
    __shared__ float bias_sums[TOK_THREADS][COLS];

    int col = blockIdx.x * COLS + threadIdx.x;
    int batch = blockIdx.y;
    int ty = threadIdx.y;

    float gate_acc = 0.0f;
    float bias_acc = 0.0f;
    if (col < K) {
        float gv = __bfloat162float(gate[batch * K + col]);
        int row_base = batch * tokens_per_sample;
        for (int tok = ty; tok < tokens_per_sample; tok += TOK_THREADS) {
            int row = row_base + tok;
            size_t idx = size_t(row) * K + col;
            float go = __bfloat162float(grad[idx]);
            float hv = __bfloat162float(h[idx]);
            float dhv = go * gv;
            bf16 dh_bf16 = __float2bfloat16(dhv);
            dh[idx] = dh_bf16;
            gate_acc += go * hv;
            bias_acc += __bfloat162float(dh_bf16);
        }
    }

    gate_sums[ty][threadIdx.x] = gate_acc;
    bias_sums[ty][threadIdx.x] = bias_acc;
    __syncthreads();

    if (ty == 0 && col < K) {
        float dg = 0.0f;
        float db = 0.0f;
        #pragma unroll
        for (int i = 0; i < TOK_THREADS; i++) {
            dg += gate_sums[i][threadIdx.x];
            db += bias_sums[i][threadIdx.x];
        }
        dgate[batch * K + col] = dg;
        atomicAdd(&dbias[col], db);
    }
}

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? PERSISTENT_GRID_BLOCKS : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if (threadIdx.x == 0) {
            args.globals.A.template prefetch_tma<typename layout::base_tile>();
            args.globals.B.template prefetch_tma<typename layout::base_tile>();
            args.globals.C.template prefetch_tma<typename layout::base_tile>();
            args.globals.preact.template prefetch_tma<typename layout::base_tile>();
        }
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::elect_leader()) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            group<NUM_CONSUMER_WARPS>::load(args.scratch.bias, args.globals.bias, {args.common.coord.y / N_BLOCK});
            group<NUM_CONSUMER_WARPS>::sync(0);
            init_bias(args.state.accum, args.scratch.bias);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(args.state.accum, args.input.a[warpgroup::groupid()], reinterpret_cast<wide_tile&>(args.input.b) );
            warpgroup::mma_async_wait();
            if (warp::elect_leader()) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            // Store pre-activation (before GELU) for backward pass
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.preact, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y + i});
            }
            // Apply GELU in registers while TMA reads preact from smem
            apply_gelu(args.state.accum);
            if (warpgroup::elect_leader())
                tma::store_async_read_wait();
            warpgroup::sync(warpgroup::groupid() + 4);
            // Store post-activation
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y + i});
            }
            // Overlap: reinit accum while TMA reads C from smem
            init_bias(args.state.accum, args.scratch.bias);
            if (warpgroup::elect_leader())
                tma::store_async_read_wait();
            if (warp::elect_leader()) arrive(args.finish_finished);
        }
    };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct native_matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? PERSISTENT_GRID_BLOCKS : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if (threadIdx.x == 0) {
            args.globals.A.template prefetch_tma<typename layout::base_tile>();
            args.globals.B.template prefetch_tma<typename layout::base_tile>();
            args.globals.C.template prefetch_tma<typename layout::base_tile>();
            args.globals.preact.template prefetch_tma<typename layout::base_tile>();
        }
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::elect_leader()) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            group<NUM_CONSUMER_WARPS>::load(args.scratch.bias, args.globals.bias, {args.common.coord.y / N_BLOCK});
            group<NUM_CONSUMER_WARPS>::sync(0);
            init_bias(args.state.accum, args.scratch.bias);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            using tall_tile = st_bf<64*N_BLOCK, 64>;
            warpgroup::mma_ABt(args.state.accum, args.input.a[warpgroup::groupid()], reinterpret_cast<tall_tile&>(args.input.b));
            warpgroup::mma_async_wait();
            if (warp::elect_leader()) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.preact, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y + i});
            }
            apply_gelu(args.state.accum);
            if (warpgroup::elect_leader())
                tma::store_async_read_wait();
            warpgroup::sync(warpgroup::groupid() + 4);
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y + i});
            }
            init_bias(args.state.accum, args.scratch.bias);
            if (warpgroup::elect_leader())
                tma::store_async_read_wait();
            if (warp::elect_leader()) arrive(args.finish_finished);
        }
    };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct native_linear_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? PERSISTENT_GRID_BLOCKS : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if (threadIdx.x == 0) {
            args.globals.A.template prefetch_tma<typename layout::base_tile>();
            args.globals.B.template prefetch_tma<typename layout::base_tile>();
            args.globals.C.template prefetch_tma<typename layout::base_tile>();
        }
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::elect_leader()) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            group<NUM_CONSUMER_WARPS>::load(args.scratch.bias, args.globals.bias, {args.common.coord.y / N_BLOCK});
            group<NUM_CONSUMER_WARPS>::sync(0);
            init_bias(args.state.accum, args.scratch.bias);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            using tall_tile = st_bf<64*N_BLOCK, 64>;
            warpgroup::mma_ABt(args.state.accum, args.input.a[warpgroup::groupid()], reinterpret_cast<tall_tile&>(args.input.b));
            warpgroup::mma_async_wait();
            if (warp::elect_leader()) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y + i});
                tma::store_async_read_wait();
            }
            init_bias(args.state.accum, args.scratch.bias);
            if (warp::elect_leader()) arrive(args.finish_finished);
        }
    };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct gated_linear_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = gated_linear_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? PERSISTENT_GRID_BLOCKS : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if (threadIdx.x == 0) {
            args.globals.A.template prefetch_tma<typename layout::base_tile>();
            args.globals.B.template prefetch_tma<typename layout::base_tile>();
            args.globals.C.template prefetch_tma<typename layout::wide_tile>();
            args.globals.preact.template prefetch_tma<typename layout::wide_tile>();
            args.globals.residual.template prefetch_tma<typename layout::wide_tile>();
        }
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::elect_leader()) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            int batch = (args.common.coord.x * 64) / args.globals.tokens_per_sample;
            group<NUM_CONSUMER_WARPS>::load(args.scratch.bias, args.globals.bias, {args.common.coord.y / N_BLOCK});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.gate, args.globals.gate, {batch, args.common.coord.y / N_BLOCK});
            group<NUM_CONSUMER_WARPS>::sync(0);
            init_bias(args.state.accum, args.scratch.bias);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            using tall_tile = st_bf<64*N_BLOCK, 64>;
            warpgroup::mma_ABt(args.state.accum, args.input.a[warpgroup::groupid()], reinterpret_cast<tall_tile&>(args.input.b));
            warpgroup::mma_async_wait();
            if (warp::elect_leader()) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            int wide_col = args.common.coord.y / N_BLOCK;
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                tma::store_async(args.globals.preact, args.finish.c[warpgroup::groupid()], {args.common.coord.x, wide_col});
                tma::store_async_read_wait();
            }
            rt_sv_op<base_ops::mul>(args.state.accum, args.scratch.gate);
            warpgroup::load_async(args.finish.c[warpgroup::groupid()], args.globals.residual, {args.common.coord.x, wide_col});
            warpgroup::load_async_wait(warpgroup::groupid());
            wg_rt_sv_op<base_ops::sum>(args.state.accum, args.finish.c[warpgroup::groupid()]);
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()], {args.common.coord.x, wide_col});
                tma::store_async_read_wait();
            }
            init_bias(args.state.accum, args.scratch.bias);
            if (warp::elect_leader()) arrive(args.finish_finished);
        }
    };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct gated_linear_out_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = gated_linear_out_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? PERSISTENT_GRID_BLOCKS : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if (threadIdx.x == 0) {
            args.globals.A.template prefetch_tma<typename layout::base_tile>();
            args.globals.B.template prefetch_tma<typename layout::base_tile>();
            args.globals.C.template prefetch_tma<typename layout::wide_tile>();
            args.globals.residual.template prefetch_tma<typename layout::wide_tile>();
        }
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::elect_leader()) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            int batch = (args.common.coord.x * 64) / args.globals.tokens_per_sample;
            group<NUM_CONSUMER_WARPS>::load(args.scratch.bias, args.globals.bias, {args.common.coord.y / N_BLOCK});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.gate, args.globals.gate, {batch, args.common.coord.y / N_BLOCK});
            group<NUM_CONSUMER_WARPS>::sync(0);
            init_bias(args.state.accum, args.scratch.bias);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            using tall_tile = st_bf<64*N_BLOCK, 64>;
            warpgroup::mma_ABt(args.state.accum, args.input.a[warpgroup::groupid()], reinterpret_cast<tall_tile&>(args.input.b));
            warpgroup::mma_async_wait();
            if (warp::elect_leader()) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            int wide_col = args.common.coord.y / N_BLOCK;
            rt_sv_op<base_ops::mul>(args.state.accum, args.scratch.gate);
            warpgroup::load_async(args.finish.c[warpgroup::groupid()], args.globals.residual, {args.common.coord.x, wide_col});
            warpgroup::load_async_wait(warpgroup::groupid());
            wg_rt_sv_op<base_ops::sum>(args.state.accum, args.finish.c[warpgroup::groupid()]);
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()], {args.common.coord.x, wide_col});
                tma::store_async_read_wait();
            }
            init_bias(args.state.accum, args.scratch.bias);
            if (warp::elect_leader()) arrive(args.finish_finished);
        }
    };
};


template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct modulated_matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = modulated_matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? PERSISTENT_GRID_BLOCKS : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if (threadIdx.x == 0) {
            args.globals.A.template prefetch_tma<typename layout::base_tile>();
            args.globals.B.template prefetch_tma<typename layout::base_tile>();
            args.globals.C.template prefetch_tma<typename layout::base_tile>();
            args.globals.preact.template prefetch_tma<typename layout::base_tile>();
        }
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::elect_leader()) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            group<NUM_CONSUMER_WARPS>::load(args.scratch.bias, args.globals.bias, {args.common.coord.y / N_BLOCK});
            group<NUM_CONSUMER_WARPS>::sync(0);
            init_bias(args.state.accum, args.scratch.bias);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            apply_adaln_modulation(
                args.input.a[warpgroup::groupid()],
                args.globals.shift,
                args.globals.scale,
                args.common.coord.x,
                args.iter,
                args.globals.tokens_per_sample,
                args.globals.K
            );
            warpgroup::mma_AB(args.state.accum, args.input.a[warpgroup::groupid()], reinterpret_cast<wide_tile&>(args.input.b) );
            warpgroup::mma_async_wait();
            if (warp::elect_leader()) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.preact, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y + i});
            }
            apply_gelu(args.state.accum);
            if (warpgroup::elect_leader())
                tma::store_async_read_wait();
            warpgroup::sync(warpgroup::groupid() + 4);
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y + i});
            }
            init_bias(args.state.accum, args.scratch.bias);
            if (warpgroup::elect_leader())
                tma::store_async_read_wait();
            if (warp::elect_leader()) arrive(args.finish_finished);
        }
    };
};

template<bool APPLY_GELU, int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct ln_adaln_linear_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = ln_adaln_matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? PERSISTENT_GRID_BLOCKS : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if (threadIdx.x == 0) {
            args.globals.A.template prefetch_tma<typename layout::base_tile>();
            args.globals.B.template prefetch_tma<typename layout::base_tile>();
            args.globals.C.template prefetch_tma<typename layout::base_tile>();
            if constexpr (APPLY_GELU) {
                args.globals.preact.template prefetch_tma<typename layout::base_tile>();
            }
        }
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::elect_leader()) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            group<NUM_CONSUMER_WARPS>::load(args.scratch.bias, args.globals.bias, {args.common.coord.y / N_BLOCK});
            group<NUM_CONSUMER_WARPS>::sync(0);
            init_bias(args.state.accum, args.scratch.bias);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            using tall_tile = st_bf<64*N_BLOCK, 64>;
            apply_ln_adaln_modulation(
                args.input.a[warpgroup::groupid()],
                args.globals.shift,
                args.globals.scale,
                args.globals.mean,
                args.globals.rstd,
                args.common.coord.x,
                args.iter,
                args.globals.tokens_per_sample,
                args.globals.K
            );
            warpgroup::mma_ABt(args.state.accum, args.input.a[warpgroup::groupid()], reinterpret_cast<tall_tile&>(args.input.b));
            warpgroup::mma_async_wait();
            if (warp::elect_leader()) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if constexpr (APPLY_GELU) {
                if (warpgroup::elect_leader()) {
                    for (int i = 0; i < N_BLOCK; i++)
                        tma::store_async(args.globals.preact, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y + i});
                }
                apply_gelu(args.state.accum);
                if (warpgroup::elect_leader())
                    tma::store_async_read_wait();
                warpgroup::sync(warpgroup::groupid() + 4);
                warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
                warpgroup::sync(warpgroup::groupid() + 4);
            }
            if (warpgroup::elect_leader()) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y + i});
                tma::store_async_read_wait();
            }
            init_bias(args.state.accum, args.scratch.bias);
            if (warp::elect_leader()) arrive(args.finish_finished);
        }
    };
};

#ifndef TORCH_COMPILE
#include <iostream>
#include <cuda_bf16.h>

#include "../common.cuh"
template<typename mmt>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, bf16 *d_bias, bf16 *d_preact,
               size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using global_layout = typename mmt::layout::global_layout;
    using bias_global   = typename mmt::layout::bias_global;
    using globals  = typename mmt::layout::globals;
    global_layout Ag{d_A, nullptr, nullptr, M, K};
    global_layout Bg{d_B, nullptr, nullptr, K, N};
    global_layout Cg{d_C, nullptr, nullptr, M, N};
    global_layout PREg{d_preact, nullptr, nullptr, M, N};
    bias_global BIASg{d_bias, nullptr, nullptr, nullptr, N};
    globals G{Ag, Bg, Cg, PREg, BIASg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

template<typename mmt>
double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << mmt::M_BLOCK*64 << "x" << mmt::N_BLOCK*64 << "\n";

    sleep_ms(500);

    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = 2 * (size_t(M) * K + size_t(N) * K + size_t(M) * N);
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    std::vector<__nv_bfloat16*> d_A(arg_group_count);
    std::vector<__nv_bfloat16*> d_B(arg_group_count);
    std::vector<__nv_bfloat16*> d_C(arg_group_count);
    std::vector<__nv_bfloat16*> d_bias(arg_group_count);
    std::vector<__nv_bfloat16*> d_preact(arg_group_count);

    __nv_bfloat16* d_C_ref;

    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K*sizeof(__nv_bfloat16));
        cudaMalloc(&d_B[i], K*N*sizeof(__nv_bfloat16));
        cudaMalloc(&d_C[i], M*N*sizeof(__nv_bfloat16));
        cudaMalloc(&d_bias[i], 1*N*sizeof(__nv_bfloat16));
        cudaMalloc(&d_preact[i], M*N*sizeof(__nv_bfloat16));
    }

    cudaMalloc(&d_C_ref, M*N*sizeof(__nv_bfloat16));

    uint64_t seed = 42;

    for (int i = 0; i < arg_group_count; i++) {
        fill<__nv_bfloat16, FillMode::RANDOM>(d_A[i], M*K, seed + i*100, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::RANDOM>(d_B[i], K*N, seed + i*100 + 1, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_C[i], M*N, 0.0f);
        fill<__nv_bfloat16, FillMode::RANDOM>(d_bias[i], 1*N, seed + i*100 + 1, -1.0f, 1.0f);
    }

    cudaDeviceSynchronize();

    reference_linear<__nv_bfloat16, __nv_bfloat16, false>(d_C_ref, d_A[0], d_B[0], d_bias[0], M, N, K);

    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        MAX_SHARED_MEMORY-1024);

    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);

    int num_warmups = ncu ? 0 : 500;
    int num_iters = ncu ? 1 : 100;

    for(int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        inner_run<mmt>(d_A[idx], d_B[idx], d_C[idx], d_bias[idx], d_preact[idx],
                       M, N, K, grid, block);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        inner_run<mmt>(d_A[idx], d_B[idx], d_C[idx], d_bias[idx], d_preact[idx],
                       M, N, K, grid, block);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double us = ms * 1000.0 / num_iters;
    double flops = 2.0 * M * N * K;
    double tflops = (flops / us) / 1e6;

    std::cout << "Average kernel execution time: " << us << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    check_correctness(d_C[0], d_C_ref, M * N);

    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
        cudaFree(d_bias[i]);
        cudaFree(d_preact[i]);
    }

    cudaFree(d_C_ref);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

int main() {
    int N;
    N = 4096;
    run_benchmark<matmul_template<2,4,8>>(N, N, N);
    return 0;
}

#else
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "pyutils/torchutils.cuh"

void gemm_custom_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &bias,
    const at::Tensor &preact
) {
    using mmt = matmul_template<2,4,8>;
    using globals = typename mmt::layout::globals;
    using global_layout = typename mmt::layout::global_layout;
    using bias_global = typename mmt::layout::bias_global;

    kittens::py::device_check(A, B, C, bias, preact);

    globals G{
        kittens::py::tensor_to_gl<global_layout>(A),
        kittens::py::tensor_to_gl<global_layout>(B),
        kittens::py::tensor_to_gl<global_layout>(C),
        kittens::py::tensor_to_gl<global_layout>(preact),
        kittens::py::tensor_to_gl<bias_global>(bias)
    };

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    dim3 grid = mmt::grid(M, N, K);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    prototype::lcf::kernel<mmt><<<grid, block, smem, stream>>>(G);
}

template<int M_BLOCK, int N_BLOCK, int SUPER_M>
void gemm_custom_native_variant_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &bias,
    const at::Tensor &preact
) {
    using mmt = native_matmul_template<M_BLOCK, N_BLOCK, SUPER_M>;
    using globals = typename mmt::layout::globals;
    using global_layout = typename mmt::layout::global_layout;
    using bias_global = typename mmt::layout::bias_global;

    kittens::py::device_check(A, B, C, bias, preact);

    globals G{
        kittens::py::tensor_to_gl<global_layout>(A),
        kittens::py::tensor_to_gl<global_layout>(B),
        kittens::py::tensor_to_gl<global_layout>(C),
        kittens::py::tensor_to_gl<global_layout>(preact),
        kittens::py::tensor_to_gl<bias_global>(bias)
    };

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    dim3 grid = mmt::grid(M, N, K);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    prototype::lcf::kernel<mmt><<<grid, block, smem, stream>>>(G);
}

void gemm_custom_native_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &bias,
    const at::Tensor &preact
) {
    gemm_custom_native_variant_entrypoint<2, 4, 8>(A, B, C, bias, preact);
}

template<int M_BLOCK, int N_BLOCK, int SUPER_M>
void gemm_linear_native_variant_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &bias
) {
    using mmt = native_linear_template<M_BLOCK, N_BLOCK, SUPER_M>;
    using globals = typename mmt::layout::globals;
    using global_layout = typename mmt::layout::global_layout;
    using bias_global = typename mmt::layout::bias_global;

    kittens::py::device_check(A, B, C, bias);

    globals G{
        kittens::py::tensor_to_gl<global_layout>(A),
        kittens::py::tensor_to_gl<global_layout>(B),
        kittens::py::tensor_to_gl<global_layout>(C),
        kittens::py::tensor_to_gl<global_layout>(C),
        kittens::py::tensor_to_gl<bias_global>(bias)
    };

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    dim3 grid = mmt::grid(M, N, K);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    prototype::lcf::kernel<mmt><<<grid, block, smem, stream>>>(G);
}

void gemm_linear_native_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &bias
) {
    gemm_linear_native_variant_entrypoint<2, 4, 8>(A, B, C, bias);
}

void gemm_linear_gated_residual_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &residual,
    const at::Tensor &gate,
    const at::Tensor &C,
    const at::Tensor &projected,
    const at::Tensor &bias,
    int64_t tokens_per_sample
) {
    using mmt = gated_linear_template<2,4,8>;
    using globals = typename mmt::layout::globals;
    using global_layout = typename mmt::layout::global_layout;
    using wide_global = typename mmt::layout::wide_global;
    using bias_global = typename mmt::layout::bias_global;
    using gate_global = typename mmt::layout::gate_global;

    kittens::py::device_check(A, B, residual, gate, C, projected, bias);
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16);
    TORCH_CHECK(residual.dtype() == torch::kBFloat16 && gate.dtype() == torch::kBFloat16);
    TORCH_CHECK(C.dtype() == torch::kBFloat16 && projected.dtype() == torch::kBFloat16 && bias.dtype() == torch::kBFloat16);
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && residual.is_contiguous());
    TORCH_CHECK(gate.is_contiguous() && C.is_contiguous() && projected.is_contiguous() && bias.is_contiguous());
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && residual.dim() == 2 && C.dim() == 2 && projected.dim() == 2);
    TORCH_CHECK(gate.dim() == 2 && bias.dim() == 1);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    int batch = gate.size(0);
    TORCH_CHECK(B.size(1) == K);
    TORCH_CHECK(residual.size(0) == M && residual.size(1) == N);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N);
    TORCH_CHECK(projected.size(0) == M && projected.size(1) == N);
    TORCH_CHECK(gate.size(1) == N);
    TORCH_CHECK(bias.numel() == N);
    TORCH_CHECK(M == batch * tokens_per_sample);
    TORCH_CHECK((M % 128) == 0 && (N % 256) == 0 && (K % 64) == 0);

    globals G{
        kittens::py::tensor_to_gl<global_layout>(A),
        kittens::py::tensor_to_gl<global_layout>(B),
        kittens::py::tensor_to_gl<wide_global>(C),
        kittens::py::tensor_to_gl<wide_global>(projected),
        kittens::py::tensor_to_gl<wide_global>(residual),
        kittens::py::tensor_to_gl<bias_global>(bias),
        kittens::py::tensor_to_gl<gate_global>(gate),
        static_cast<int>(tokens_per_sample)
    };

    dim3 grid = mmt::grid(M, N, K);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    prototype::lcf::kernel<mmt><<<grid, block, smem, stream>>>(G);
}

void gemm_linear_gated_residual_out_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &residual,
    const at::Tensor &gate,
    const at::Tensor &C,
    const at::Tensor &bias,
    int64_t tokens_per_sample
) {
    using mmt = gated_linear_out_template<2,4,8>;
    using globals = typename mmt::layout::globals;
    using global_layout = typename mmt::layout::global_layout;
    using wide_global = typename mmt::layout::wide_global;
    using bias_global = typename mmt::layout::bias_global;
    using gate_global = typename mmt::layout::gate_global;

    kittens::py::device_check(A, B, residual, gate, C, bias);
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16);
    TORCH_CHECK(residual.dtype() == torch::kBFloat16 && gate.dtype() == torch::kBFloat16);
    TORCH_CHECK(C.dtype() == torch::kBFloat16 && bias.dtype() == torch::kBFloat16);
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && residual.is_contiguous());
    TORCH_CHECK(gate.is_contiguous() && C.is_contiguous() && bias.is_contiguous());
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && residual.dim() == 2 && C.dim() == 2);
    TORCH_CHECK(gate.dim() == 2 && bias.dim() == 1);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    int batch = gate.size(0);
    TORCH_CHECK(B.size(1) == K);
    TORCH_CHECK(residual.size(0) == M && residual.size(1) == N);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N);
    TORCH_CHECK(gate.size(1) == N);
    TORCH_CHECK(bias.numel() == N);
    TORCH_CHECK(M == batch * tokens_per_sample);
    TORCH_CHECK((M % 128) == 0 && (N % 256) == 0 && (K % 64) == 0);
    TORCH_CHECK((tokens_per_sample % 64) == 0);

    globals G{
        kittens::py::tensor_to_gl<global_layout>(A),
        kittens::py::tensor_to_gl<global_layout>(B),
        kittens::py::tensor_to_gl<wide_global>(C),
        kittens::py::tensor_to_gl<wide_global>(residual),
        kittens::py::tensor_to_gl<bias_global>(bias),
        kittens::py::tensor_to_gl<gate_global>(gate),
        static_cast<int>(tokens_per_sample)
    };

    dim3 grid = mmt::grid(M, N, K);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    prototype::lcf::kernel<mmt><<<grid, block, smem, stream>>>(G);
}



void adaln_modulate_entrypoint(
    const at::Tensor &A,
    const at::Tensor &shift,
    const at::Tensor &scale,
    const at::Tensor &out,
    int64_t tokens_per_sample
) {
    kittens::py::device_check(A, shift, scale, out);
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && shift.dtype() == torch::kBFloat16);
    TORCH_CHECK(scale.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16);
    TORCH_CHECK(A.is_contiguous() && shift.is_contiguous() && scale.is_contiguous() && out.is_contiguous());
    TORCH_CHECK(A.dim() == 2 && out.dim() == 2 && shift.dim() == 2 && scale.dim() == 2);
    TORCH_CHECK(tokens_per_sample > 0);
    int M = A.size(0);
    int K = A.size(1);
    int batch = shift.size(0);
    TORCH_CHECK(out.size(0) == M && out.size(1) == K);
    TORCH_CHECK(scale.size(0) == batch && shift.size(1) == K && scale.size(1) == K);
    TORCH_CHECK(M == batch * tokens_per_sample);

    size_t total = size_t(M) * K;
    int threads = 256;
    int blocks = (total + threads * 4 - 1) / (threads * 4);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    adaln_modulate_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<bf16*>(out.data_ptr()),
        reinterpret_cast<const bf16*>(A.data_ptr()),
        reinterpret_cast<const bf16*>(shift.data_ptr()),
        reinterpret_cast<const bf16*>(scale.data_ptr()),
        M,
        K,
        static_cast<int>(tokens_per_sample)
    );
}


void adaln_modulate_backward_entrypoint(
    const at::Tensor &grad,
    const at::Tensor &A,
    const at::Tensor &scale,
    const at::Tensor &dA,
    const at::Tensor &dshift,
    const at::Tensor &dscale,
    int64_t tokens_per_sample
) {
    kittens::py::device_check(grad, A, scale, dA, dshift, dscale);
    TORCH_CHECK(grad.dtype() == torch::kBFloat16 && A.dtype() == torch::kBFloat16);
    TORCH_CHECK(scale.dtype() == torch::kBFloat16 && dA.dtype() == torch::kBFloat16);
    TORCH_CHECK(dshift.dtype() == torch::kFloat32 && dscale.dtype() == torch::kFloat32);
    TORCH_CHECK(grad.is_contiguous() && A.is_contiguous() && scale.is_contiguous());
    TORCH_CHECK(dA.is_contiguous() && dshift.is_contiguous() && dscale.is_contiguous());
    TORCH_CHECK(grad.dim() == 2 && A.dim() == 2 && dA.dim() == 2);
    TORCH_CHECK(scale.dim() == 2 && dshift.dim() == 2 && dscale.dim() == 2);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = A.size(0);
    int K = A.size(1);
    int batch = scale.size(0);
    TORCH_CHECK(grad.size(0) == M && grad.size(1) == K);
    TORCH_CHECK(dA.size(0) == M && dA.size(1) == K);
    TORCH_CHECK(scale.size(1) == K);
    TORCH_CHECK(dshift.size(0) == batch && dshift.size(1) == K);
    TORCH_CHECK(dscale.size(0) == batch && dscale.size(1) == K);
    TORCH_CHECK(M == batch * tokens_per_sample);

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, batch);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    adaln_modulate_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<bf16*>(dA.data_ptr()),
        reinterpret_cast<float*>(dshift.data_ptr()),
        reinterpret_cast<float*>(dscale.data_ptr()),
        reinterpret_cast<const bf16*>(grad.data_ptr()),
        reinterpret_cast<const bf16*>(A.data_ptr()),
        reinterpret_cast<const bf16*>(scale.data_ptr()),
        K,
        static_cast<int>(tokens_per_sample)
    );
}

void gated_residual_entrypoint(
    const at::Tensor &x,
    const at::Tensor &h,
    const at::Tensor &gate,
    const at::Tensor &out,
    int64_t tokens_per_sample
) {
    kittens::py::device_check(x, h, gate, out);
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && h.dtype() == torch::kBFloat16);
    TORCH_CHECK(gate.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16);
    TORCH_CHECK(x.is_contiguous() && h.is_contiguous() && gate.is_contiguous() && out.is_contiguous());
    TORCH_CHECK(x.dim() == 2 && h.dim() == 2 && out.dim() == 2 && gate.dim() == 2);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = x.size(0);
    int K = x.size(1);
    int batch = gate.size(0);
    TORCH_CHECK(h.size(0) == M && h.size(1) == K);
    TORCH_CHECK(out.size(0) == M && out.size(1) == K);
    TORCH_CHECK(gate.size(1) == K);
    TORCH_CHECK(M == batch * tokens_per_sample);

    size_t total = size_t(M) * K;
    int threads = 256;
    int blocks = (total + threads * 4 - 1) / (threads * 4);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if ((K % 2) == 0) {
        int pair_blocks = (size_t(M) * K / 2 + threads * 4 - 1) / (threads * 4);
        gated_residual_forward_vec2_kernel<<<pair_blocks, threads, 0, stream>>>(
            reinterpret_cast<bf16*>(out.data_ptr()),
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<const bf16*>(h.data_ptr()),
            reinterpret_cast<const bf16*>(gate.data_ptr()),
            M,
            K,
            static_cast<int>(tokens_per_sample)
        );
    } else {
        gated_residual_forward_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<bf16*>(out.data_ptr()),
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<const bf16*>(h.data_ptr()),
            reinterpret_cast<const bf16*>(gate.data_ptr()),
            M,
            K,
            static_cast<int>(tokens_per_sample)
        );
    }
}

void gated_residual_backward_entrypoint(
    const at::Tensor &grad,
    const at::Tensor &h,
    const at::Tensor &gate,
    const at::Tensor &dx,
    const at::Tensor &dh,
    const at::Tensor &dgate,
    int64_t tokens_per_sample
) {
    kittens::py::device_check(grad, h, gate, dx, dh, dgate);
    TORCH_CHECK(grad.dtype() == torch::kBFloat16 && h.dtype() == torch::kBFloat16);
    TORCH_CHECK(gate.dtype() == torch::kBFloat16 && dx.dtype() == torch::kBFloat16 && dh.dtype() == torch::kBFloat16);
    TORCH_CHECK(dgate.dtype() == torch::kFloat32);
    TORCH_CHECK(grad.is_contiguous() && h.is_contiguous() && gate.is_contiguous());
    TORCH_CHECK(dx.is_contiguous() && dh.is_contiguous() && dgate.is_contiguous());
    TORCH_CHECK(grad.dim() == 2 && h.dim() == 2 && dx.dim() == 2 && dh.dim() == 2 && gate.dim() == 2 && dgate.dim() == 2);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = grad.size(0);
    int K = grad.size(1);
    int batch = gate.size(0);
    TORCH_CHECK(h.size(0) == M && h.size(1) == K);
    TORCH_CHECK(dx.size(0) == M && dx.size(1) == K);
    TORCH_CHECK(dh.size(0) == M && dh.size(1) == K);
    TORCH_CHECK(gate.size(1) == K);
    TORCH_CHECK(dgate.size(0) == batch && dgate.size(1) == K);
    TORCH_CHECK(M == batch * tokens_per_sample);

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, batch);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gated_residual_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<bf16*>(dx.data_ptr()),
        reinterpret_cast<bf16*>(dh.data_ptr()),
        reinterpret_cast<float*>(dgate.data_ptr()),
        reinterpret_cast<const bf16*>(grad.data_ptr()),
        reinterpret_cast<const bf16*>(h.data_ptr()),
        reinterpret_cast<const bf16*>(gate.data_ptr()),
        K,
        static_cast<int>(tokens_per_sample)
    );
}

void gated_residual_backward_no_dx_entrypoint(
    const at::Tensor &grad,
    const at::Tensor &h,
    const at::Tensor &gate,
    const at::Tensor &dh,
    const at::Tensor &dgate,
    int64_t tokens_per_sample
) {
    kittens::py::device_check(grad, h, gate, dh, dgate);
    TORCH_CHECK(grad.dtype() == torch::kBFloat16 && h.dtype() == torch::kBFloat16);
    TORCH_CHECK(gate.dtype() == torch::kBFloat16 && dh.dtype() == torch::kBFloat16);
    TORCH_CHECK(dgate.dtype() == torch::kFloat32);
    TORCH_CHECK(grad.is_contiguous() && h.is_contiguous() && gate.is_contiguous());
    TORCH_CHECK(dh.is_contiguous() && dgate.is_contiguous());
    TORCH_CHECK(grad.dim() == 2 && h.dim() == 2 && dh.dim() == 2 && gate.dim() == 2 && dgate.dim() == 2);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = grad.size(0);
    int K = grad.size(1);
    int batch = gate.size(0);
    TORCH_CHECK(h.size(0) == M && h.size(1) == K);
    TORCH_CHECK(dh.size(0) == M && dh.size(1) == K);
    TORCH_CHECK(gate.size(1) == K);
    TORCH_CHECK(dgate.size(0) == batch && dgate.size(1) == K);
    TORCH_CHECK(M == batch * tokens_per_sample);

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, batch);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gated_residual_backward_no_dx_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<bf16*>(dh.data_ptr()),
        reinterpret_cast<float*>(dgate.data_ptr()),
        reinterpret_cast<const bf16*>(grad.data_ptr()),
        reinterpret_cast<const bf16*>(h.data_ptr()),
        reinterpret_cast<const bf16*>(gate.data_ptr()),
        K,
        static_cast<int>(tokens_per_sample)
    );
}

void gated_residual_backward_no_dx_db_entrypoint(
    const at::Tensor &grad,
    const at::Tensor &h,
    const at::Tensor &gate,
    const at::Tensor &dh,
    const at::Tensor &dgate,
    const at::Tensor &dbias,
    int64_t tokens_per_sample
) {
    kittens::py::device_check(grad, h, gate, dh, dgate, dbias);
    TORCH_CHECK(grad.dtype() == torch::kBFloat16 && h.dtype() == torch::kBFloat16);
    TORCH_CHECK(gate.dtype() == torch::kBFloat16 && dh.dtype() == torch::kBFloat16);
    TORCH_CHECK(dgate.dtype() == torch::kFloat32 && dbias.dtype() == torch::kFloat32);
    TORCH_CHECK(grad.is_contiguous() && h.is_contiguous() && gate.is_contiguous());
    TORCH_CHECK(dh.is_contiguous() && dgate.is_contiguous() && dbias.is_contiguous());
    TORCH_CHECK(grad.dim() == 2 && h.dim() == 2 && dh.dim() == 2 && gate.dim() == 2 && dgate.dim() == 2 && dbias.dim() == 1);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = grad.size(0);
    int K = grad.size(1);
    int batch = gate.size(0);
    TORCH_CHECK(h.size(0) == M && h.size(1) == K);
    TORCH_CHECK(dh.size(0) == M && dh.size(1) == K);
    TORCH_CHECK(gate.size(1) == K);
    TORCH_CHECK(dgate.size(0) == batch && dgate.size(1) == K);
    TORCH_CHECK(dbias.size(0) == K);
    TORCH_CHECK(M == batch * tokens_per_sample);

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, batch);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(dbias.data_ptr(), 0, K * sizeof(float), stream);
    gated_residual_backward_no_dx_db_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<bf16*>(dh.data_ptr()),
        reinterpret_cast<float*>(dgate.data_ptr()),
        reinterpret_cast<float*>(dbias.data_ptr()),
        reinterpret_cast<const bf16*>(grad.data_ptr()),
        reinterpret_cast<const bf16*>(h.data_ptr()),
        reinterpret_cast<const bf16*>(gate.data_ptr()),
        K,
        static_cast<int>(tokens_per_sample)
    );
}


void layernorm_adaln_entrypoint(
    const at::Tensor &A,
    const at::Tensor &shift,
    const at::Tensor &scale,
    const at::Tensor &out,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    int64_t tokens_per_sample,
    double eps
) {
    kittens::py::device_check(A, shift, scale, out, mean, rstd);
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && shift.dtype() == torch::kBFloat16);
    TORCH_CHECK(scale.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16);
    TORCH_CHECK(mean.dtype() == torch::kFloat32 && rstd.dtype() == torch::kFloat32);
    TORCH_CHECK(A.is_contiguous() && shift.is_contiguous() && scale.is_contiguous());
    TORCH_CHECK(out.is_contiguous() && mean.is_contiguous() && rstd.is_contiguous());
    TORCH_CHECK(A.dim() == 2 && out.dim() == 2 && shift.dim() == 2 && scale.dim() == 2);
    TORCH_CHECK(mean.dim() == 1 && rstd.dim() == 1);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = A.size(0);
    int K = A.size(1);
    int batch = shift.size(0);
    TORCH_CHECK(out.size(0) == M && out.size(1) == K);
    TORCH_CHECK(scale.size(0) == batch && shift.size(1) == K && scale.size(1) == K);
    TORCH_CHECK(mean.size(0) == M && rstd.size(0) == M);
    TORCH_CHECK(M == batch * tokens_per_sample);

    int threads = 256;
    int smem = threads * 2 * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (K == 1024) {
        layernorm_adaln_forward_k1024_vec2_kernel<<<M, 128, 0, stream>>>(
            reinterpret_cast<bf16*>(out.data_ptr()),
            reinterpret_cast<float*>(mean.data_ptr()),
            reinterpret_cast<float*>(rstd.data_ptr()),
            reinterpret_cast<const bf16*>(A.data_ptr()),
            reinterpret_cast<const bf16*>(shift.data_ptr()),
            reinterpret_cast<const bf16*>(scale.data_ptr()),
            static_cast<int>(tokens_per_sample),
            static_cast<float>(eps)
        );
    } else {
        layernorm_adaln_forward_kernel<<<M, threads, smem, stream>>>(
            reinterpret_cast<bf16*>(out.data_ptr()),
            reinterpret_cast<float*>(mean.data_ptr()),
            reinterpret_cast<float*>(rstd.data_ptr()),
            reinterpret_cast<const bf16*>(A.data_ptr()),
            reinterpret_cast<const bf16*>(shift.data_ptr()),
            reinterpret_cast<const bf16*>(scale.data_ptr()),
            M,
            K,
            static_cast<int>(tokens_per_sample),
            static_cast<float>(eps)
        );
    }
}

void layernorm_adaln_persistent_entrypoint(
    const at::Tensor &A,
    const at::Tensor &shift,
    const at::Tensor &scale,
    const at::Tensor &out,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    int64_t tokens_per_sample,
    double eps
) {
    kittens::py::device_check(A, shift, scale, out, mean, rstd);
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && shift.dtype() == torch::kBFloat16);
    TORCH_CHECK(scale.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16);
    TORCH_CHECK(mean.dtype() == torch::kFloat32 && rstd.dtype() == torch::kFloat32);
    TORCH_CHECK(A.is_contiguous() && shift.is_contiguous() && scale.is_contiguous());
    TORCH_CHECK(out.is_contiguous() && mean.is_contiguous() && rstd.is_contiguous());
    TORCH_CHECK(A.dim() == 2 && out.dim() == 2 && shift.dim() == 2 && scale.dim() == 2);
    TORCH_CHECK(mean.dim() == 1 && rstd.dim() == 1);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = A.size(0);
    int K = A.size(1);
    int batch = shift.size(0);
    TORCH_CHECK(K == 1024);
    TORCH_CHECK(out.size(0) == M && out.size(1) == K);
    TORCH_CHECK(scale.size(0) == batch && shift.size(1) == K && scale.size(1) == K);
    TORCH_CHECK(mean.size(0) == M && rstd.size(0) == M);
    TORCH_CHECK(M == batch * tokens_per_sample);

    int device = A.get_device();
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int blocks = M < props.multiProcessorCount * 32 ? M : props.multiProcessorCount * 32;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    layernorm_adaln_forward_k1024_vec2_persistent_kernel<<<blocks, 128, 0, stream>>>(
        reinterpret_cast<bf16*>(out.data_ptr()),
        reinterpret_cast<float*>(mean.data_ptr()),
        reinterpret_cast<float*>(rstd.data_ptr()),
        reinterpret_cast<const bf16*>(A.data_ptr()),
        reinterpret_cast<const bf16*>(shift.data_ptr()),
        reinterpret_cast<const bf16*>(scale.data_ptr()),
        M,
        static_cast<int>(tokens_per_sample),
        static_cast<float>(eps)
    );
}

void layernorm_adaln_warp4_entrypoint(
    const at::Tensor &A,
    const at::Tensor &shift,
    const at::Tensor &scale,
    const at::Tensor &out,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    int64_t tokens_per_sample,
    double eps
) {
    kittens::py::device_check(A, shift, scale, out, mean, rstd);
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && shift.dtype() == torch::kBFloat16);
    TORCH_CHECK(scale.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16);
    TORCH_CHECK(mean.dtype() == torch::kFloat32 && rstd.dtype() == torch::kFloat32);
    TORCH_CHECK(A.is_contiguous() && shift.is_contiguous() && scale.is_contiguous());
    TORCH_CHECK(out.is_contiguous() && mean.is_contiguous() && rstd.is_contiguous());
    TORCH_CHECK(A.dim() == 2 && out.dim() == 2 && shift.dim() == 2 && scale.dim() == 2);
    TORCH_CHECK(mean.dim() == 1 && rstd.dim() == 1);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = A.size(0);
    int K = A.size(1);
    int batch = shift.size(0);
    TORCH_CHECK(K == 1024);
    TORCH_CHECK(out.size(0) == M && out.size(1) == K);
    TORCH_CHECK(scale.size(0) == batch && shift.size(1) == K && scale.size(1) == K);
    TORCH_CHECK(mean.size(0) == M && rstd.size(0) == M);
    TORCH_CHECK(M == batch * tokens_per_sample);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    layernorm_adaln_forward_k1024_warp4_kernel<<<(M + 3) / 4, 128, 0, stream>>>(
        reinterpret_cast<bf16*>(out.data_ptr()),
        reinterpret_cast<float*>(mean.data_ptr()),
        reinterpret_cast<float*>(rstd.data_ptr()),
        reinterpret_cast<const bf16*>(A.data_ptr()),
        reinterpret_cast<const bf16*>(shift.data_ptr()),
        reinterpret_cast<const bf16*>(scale.data_ptr()),
        M,
        static_cast<int>(tokens_per_sample),
        static_cast<float>(eps)
    );
}

void layernorm_stats_entrypoint(
    const at::Tensor &A,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    double eps
) {
    kittens::py::device_check(A, mean, rstd);
    TORCH_CHECK(A.dtype() == torch::kBFloat16);
    TORCH_CHECK(mean.dtype() == torch::kFloat32 && rstd.dtype() == torch::kFloat32);
    TORCH_CHECK(A.is_contiguous() && mean.is_contiguous() && rstd.is_contiguous());
    TORCH_CHECK(A.dim() == 2 && mean.dim() == 1 && rstd.dim() == 1);

    int M = A.size(0);
    int K = A.size(1);
    TORCH_CHECK(mean.size(0) == M && rstd.size(0) == M);

    int threads = 256;
    int smem = threads * 2 * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    layernorm_stats_kernel<<<M, threads, smem, stream>>>(
        reinterpret_cast<float*>(mean.data_ptr()),
        reinterpret_cast<float*>(rstd.data_ptr()),
        reinterpret_cast<const bf16*>(A.data_ptr()),
        M,
        K,
        static_cast<float>(eps)
    );
}

void layernorm_adaln_backward_entrypoint(
    const at::Tensor &grad,
    const at::Tensor &A,
    const at::Tensor &scale,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    const at::Tensor &dA,
    const at::Tensor &dshift,
    const at::Tensor &dscale,
    int64_t tokens_per_sample
) {
    kittens::py::device_check(grad, A, scale, mean, rstd, dA, dshift, dscale);
    TORCH_CHECK(grad.dtype() == torch::kBFloat16 && A.dtype() == torch::kBFloat16);
    TORCH_CHECK(scale.dtype() == torch::kBFloat16 && dA.dtype() == torch::kBFloat16);
    TORCH_CHECK(mean.dtype() == torch::kFloat32 && rstd.dtype() == torch::kFloat32);
    TORCH_CHECK(dshift.dtype() == torch::kFloat32 && dscale.dtype() == torch::kFloat32);
    TORCH_CHECK(grad.is_contiguous() && A.is_contiguous() && scale.is_contiguous());
    TORCH_CHECK(mean.is_contiguous() && rstd.is_contiguous() && dA.is_contiguous());
    TORCH_CHECK(dshift.is_contiguous() && dscale.is_contiguous());
    TORCH_CHECK(grad.dim() == 2 && A.dim() == 2 && dA.dim() == 2);
    TORCH_CHECK(scale.dim() == 2 && dshift.dim() == 2 && dscale.dim() == 2);
    TORCH_CHECK(mean.dim() == 1 && rstd.dim() == 1);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = A.size(0);
    int K = A.size(1);
    int batch = scale.size(0);
    TORCH_CHECK(grad.size(0) == M && grad.size(1) == K);
    TORCH_CHECK(dA.size(0) == M && dA.size(1) == K);
    TORCH_CHECK(mean.size(0) == M && rstd.size(0) == M);
    TORCH_CHECK(scale.size(1) == K);
    TORCH_CHECK(dshift.size(0) == batch && dshift.size(1) == K);
    TORCH_CHECK(dscale.size(0) == batch && dscale.size(1) == K);
    TORCH_CHECK(M == batch * tokens_per_sample);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;
    int smem = threads * 2 * sizeof(float);
    layernorm_adaln_backward_kernel<<<M, threads, smem, stream>>>(
        reinterpret_cast<bf16*>(dA.data_ptr()),
        reinterpret_cast<const bf16*>(grad.data_ptr()),
        reinterpret_cast<const bf16*>(A.data_ptr()),
        reinterpret_cast<const bf16*>(scale.data_ptr()),
        reinterpret_cast<const float*>(mean.data_ptr()),
        reinterpret_cast<const float*>(rstd.data_ptr()),
        M,
        K,
        static_cast<int>(tokens_per_sample)
    );
    if (K == 1024) {
        if (tokens_per_sample >= 2048) {
            dim3 param_block(16, 32);
            dim3 param_grid(1024 / param_block.x, batch);
            layernorm_adaln_param_backward_k1024_cols16_tok32_kernel<<<param_grid, param_block, 0, stream>>>(
                reinterpret_cast<float*>(dshift.data_ptr()),
                reinterpret_cast<float*>(dscale.data_ptr()),
                reinterpret_cast<const bf16*>(grad.data_ptr()),
                reinterpret_cast<const bf16*>(A.data_ptr()),
                reinterpret_cast<const float*>(mean.data_ptr()),
                reinterpret_cast<const float*>(rstd.data_ptr()),
                static_cast<int>(tokens_per_sample)
            );
        } else {
            dim3 param_block(32, 16);
            dim3 param_grid(1024 / param_block.x, batch);
            layernorm_adaln_param_backward_k1024_cols32_kernel<<<param_grid, param_block, 0, stream>>>(
                reinterpret_cast<float*>(dshift.data_ptr()),
                reinterpret_cast<float*>(dscale.data_ptr()),
                reinterpret_cast<const bf16*>(grad.data_ptr()),
                reinterpret_cast<const bf16*>(A.data_ptr()),
                reinterpret_cast<const float*>(mean.data_ptr()),
                reinterpret_cast<const float*>(rstd.data_ptr()),
                static_cast<int>(tokens_per_sample)
            );
        }
    } else {
        dim3 param_block(16, 16);
        dim3 param_grid((K + param_block.x - 1) / param_block.x, batch);
        layernorm_adaln_param_backward_kernel<<<param_grid, param_block, 0, stream>>>(
            reinterpret_cast<float*>(dshift.data_ptr()),
            reinterpret_cast<float*>(dscale.data_ptr()),
            reinterpret_cast<const bf16*>(grad.data_ptr()),
            reinterpret_cast<const bf16*>(A.data_ptr()),
            reinterpret_cast<const float*>(mean.data_ptr()),
            reinterpret_cast<const float*>(rstd.data_ptr()),
            K,
            static_cast<int>(tokens_per_sample)
        );
    }
}

void layernorm_adaln_backward_warp4_entrypoint(
    const at::Tensor &grad,
    const at::Tensor &A,
    const at::Tensor &scale,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    const at::Tensor &dA,
    const at::Tensor &dshift,
    const at::Tensor &dscale,
    int64_t tokens_per_sample
) {
    kittens::py::device_check(grad, A, scale, mean, rstd, dA, dshift, dscale);
    TORCH_CHECK(grad.dtype() == torch::kBFloat16 && A.dtype() == torch::kBFloat16);
    TORCH_CHECK(scale.dtype() == torch::kBFloat16 && dA.dtype() == torch::kBFloat16);
    TORCH_CHECK(mean.dtype() == torch::kFloat32 && rstd.dtype() == torch::kFloat32);
    TORCH_CHECK(dshift.dtype() == torch::kFloat32 && dscale.dtype() == torch::kFloat32);
    TORCH_CHECK(grad.is_contiguous() && A.is_contiguous() && scale.is_contiguous());
    TORCH_CHECK(mean.is_contiguous() && rstd.is_contiguous() && dA.is_contiguous());
    TORCH_CHECK(dshift.is_contiguous() && dscale.is_contiguous());
    TORCH_CHECK(grad.dim() == 2 && A.dim() == 2 && dA.dim() == 2);
    TORCH_CHECK(scale.dim() == 2 && dshift.dim() == 2 && dscale.dim() == 2);
    TORCH_CHECK(mean.dim() == 1 && rstd.dim() == 1);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = A.size(0);
    int K = A.size(1);
    int batch = scale.size(0);
    TORCH_CHECK(K == 1024);
    TORCH_CHECK(grad.size(0) == M && grad.size(1) == K);
    TORCH_CHECK(dA.size(0) == M && dA.size(1) == K);
    TORCH_CHECK(mean.size(0) == M && rstd.size(0) == M);
    TORCH_CHECK(scale.size(1) == K);
    TORCH_CHECK(dshift.size(0) == batch && dshift.size(1) == K);
    TORCH_CHECK(dscale.size(0) == batch && dscale.size(1) == K);
    TORCH_CHECK(M == batch * tokens_per_sample);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    layernorm_adaln_backward_k1024_warp4_kernel<<<(M + 3) / 4, 128, 0, stream>>>(
        reinterpret_cast<bf16*>(dA.data_ptr()),
        reinterpret_cast<const bf16*>(grad.data_ptr()),
        reinterpret_cast<const bf16*>(A.data_ptr()),
        reinterpret_cast<const bf16*>(scale.data_ptr()),
        reinterpret_cast<const float*>(mean.data_ptr()),
        reinterpret_cast<const float*>(rstd.data_ptr()),
        M,
        static_cast<int>(tokens_per_sample)
    );
    if (tokens_per_sample >= 2048) {
        dim3 param_block(16, 32);
        dim3 param_grid(1024 / param_block.x, batch);
        layernorm_adaln_param_backward_k1024_cols16_tok32_kernel<<<param_grid, param_block, 0, stream>>>(
            reinterpret_cast<float*>(dshift.data_ptr()),
            reinterpret_cast<float*>(dscale.data_ptr()),
            reinterpret_cast<const bf16*>(grad.data_ptr()),
            reinterpret_cast<const bf16*>(A.data_ptr()),
            reinterpret_cast<const float*>(mean.data_ptr()),
            reinterpret_cast<const float*>(rstd.data_ptr()),
            static_cast<int>(tokens_per_sample)
        );
    } else {
        dim3 param_block(32, 16);
        dim3 param_grid(1024 / param_block.x, batch);
        layernorm_adaln_param_backward_k1024_cols32_kernel<<<param_grid, param_block, 0, stream>>>(
            reinterpret_cast<float*>(dshift.data_ptr()),
            reinterpret_cast<float*>(dscale.data_ptr()),
            reinterpret_cast<const bf16*>(grad.data_ptr()),
            reinterpret_cast<const bf16*>(A.data_ptr()),
            reinterpret_cast<const float*>(mean.data_ptr()),
            reinterpret_cast<const float*>(rstd.data_ptr()),
            static_cast<int>(tokens_per_sample)
        );
    }
}

template<bool APPLY_GELU>
void gemm_ln_adaln_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &bias,
    const at::Tensor &preact,
    const at::Tensor &shift,
    const at::Tensor &scale,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    int64_t tokens_per_sample
) {
    using mmt = ln_adaln_linear_template<APPLY_GELU, 2, 4, 8>;
    using globals = typename mmt::layout::globals;
    using global_layout = typename mmt::layout::global_layout;
    using bias_global = typename mmt::layout::bias_global;

    kittens::py::device_check(A, B, C, bias, preact, shift, scale, mean, rstd);
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16);
    TORCH_CHECK(C.dtype() == torch::kBFloat16 && bias.dtype() == torch::kBFloat16);
    TORCH_CHECK(preact.dtype() == torch::kBFloat16 && shift.dtype() == torch::kBFloat16 && scale.dtype() == torch::kBFloat16);
    TORCH_CHECK(mean.dtype() == torch::kFloat32 && rstd.dtype() == torch::kFloat32);
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous() && bias.is_contiguous());
    TORCH_CHECK(preact.is_contiguous() && shift.is_contiguous() && scale.is_contiguous());
    TORCH_CHECK(mean.is_contiguous() && rstd.is_contiguous());
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2 && preact.dim() == 2);
    TORCH_CHECK(shift.dim() == 2 && scale.dim() == 2 && mean.dim() == 1 && rstd.dim() == 1);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    int batch = shift.size(0);
    TORCH_CHECK(B.size(1) == K);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N);
    TORCH_CHECK(preact.size(0) == M && preact.size(1) == N);
    TORCH_CHECK(bias.numel() == N);
    TORCH_CHECK(mean.size(0) == M && rstd.size(0) == M);
    TORCH_CHECK(scale.size(0) == batch && shift.size(1) == K && scale.size(1) == K);
    TORCH_CHECK(M == batch * tokens_per_sample);
    TORCH_CHECK((M % 128) == 0 && (N % 256) == 0 && (K % 64) == 0);

    globals G{
        kittens::py::tensor_to_gl<global_layout>(A),
        kittens::py::tensor_to_gl<global_layout>(B),
        kittens::py::tensor_to_gl<global_layout>(C),
        kittens::py::tensor_to_gl<global_layout>(preact),
        kittens::py::tensor_to_gl<bias_global>(bias),
        reinterpret_cast<const bf16*>(shift.data_ptr()),
        reinterpret_cast<const bf16*>(scale.data_ptr()),
        reinterpret_cast<const float*>(mean.data_ptr()),
        reinterpret_cast<const float*>(rstd.data_ptr()),
        static_cast<int>(tokens_per_sample),
        K
    };

    dim3 grid = mmt::grid(M, N, K);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    prototype::lcf::kernel<mmt><<<grid, block, smem, stream>>>(G);
}

void gemm_custom_adaln_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &bias,
    const at::Tensor &preact,
    const at::Tensor &shift,
    const at::Tensor &scale,
    int64_t tokens_per_sample
) {
    using mmt = modulated_matmul_template<2,4,8>;
    using globals = typename mmt::layout::globals;
    using global_layout = typename mmt::layout::global_layout;
    using bias_global = typename mmt::layout::bias_global;

    kittens::py::device_check(A, B, C, bias, preact, shift, scale);
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16);
    TORCH_CHECK(C.dtype() == torch::kBFloat16 && bias.dtype() == torch::kBFloat16);
    TORCH_CHECK(preact.dtype() == torch::kBFloat16 && shift.dtype() == torch::kBFloat16 && scale.dtype() == torch::kBFloat16);
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous() && bias.is_contiguous());
    TORCH_CHECK(preact.is_contiguous() && shift.is_contiguous() && scale.is_contiguous());
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2 && preact.dim() == 2);
    TORCH_CHECK(shift.dim() == 2 && scale.dim() == 2);
    TORCH_CHECK(tokens_per_sample > 0);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    int batch = shift.size(0);
    TORCH_CHECK(B.size(0) == K);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N);
    TORCH_CHECK(preact.size(0) == M && preact.size(1) == N);
    TORCH_CHECK(bias.numel() == N);
    TORCH_CHECK(scale.size(0) == batch && shift.size(1) == K && scale.size(1) == K);
    TORCH_CHECK(M == batch * tokens_per_sample);
    TORCH_CHECK((M % 128) == 0 && (N % 256) == 0 && (K % 64) == 0);

    globals G{
        kittens::py::tensor_to_gl<global_layout>(A),
        kittens::py::tensor_to_gl<global_layout>(B),
        kittens::py::tensor_to_gl<global_layout>(C),
        kittens::py::tensor_to_gl<global_layout>(preact),
        kittens::py::tensor_to_gl<bias_global>(bias),
        reinterpret_cast<const bf16*>(shift.data_ptr()),
        reinterpret_cast<const bf16*>(scale.data_ptr()),
        static_cast<int>(tokens_per_sample),
        K
    };

    dim3 grid = mmt::grid(M, N, K);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    prototype::lcf::kernel<mmt><<<grid, block, smem, stream>>>(G);
}

PYBIND11_MODULE(_C, m) {
    m.def("gemm_custom", &gemm_custom_entrypoint);
    m.def("gemm_custom_native", &gemm_custom_native_entrypoint);
    m.def("gemm_custom_native_m2n4s4", &gemm_custom_native_variant_entrypoint<2,4,4>);
    m.def("gemm_custom_native_m2n4s16", &gemm_custom_native_variant_entrypoint<2,4,16>);
    m.def("gemm_custom_native_m2n2s8", &gemm_custom_native_variant_entrypoint<2,2,8>);
    m.def("gemm_custom_native_m1n4s8", &gemm_custom_native_variant_entrypoint<1,4,8>);
    m.def("gemm_linear_native", &gemm_linear_native_entrypoint);
    m.def("gemm_linear_native_m2n4s4", &gemm_linear_native_variant_entrypoint<2,4,4>);
    m.def("gemm_linear_native_m2n4s16", &gemm_linear_native_variant_entrypoint<2,4,16>);
    m.def("gemm_linear_native_m2n2s8", &gemm_linear_native_variant_entrypoint<2,2,8>);
    m.def("gemm_linear_native_m1n4s8", &gemm_linear_native_variant_entrypoint<1,4,8>);
    m.def("gemm_linear_gated_residual", &gemm_linear_gated_residual_entrypoint);
    m.def("gemm_linear_gated_residual_out", &gemm_linear_gated_residual_out_entrypoint);
    m.def("adaln_modulate", &adaln_modulate_entrypoint);
    m.def("adaln_modulate_backward", &adaln_modulate_backward_entrypoint);
    m.def("gated_residual", &gated_residual_entrypoint);
    m.def("gated_residual_backward", &gated_residual_backward_entrypoint);
    m.def("gated_residual_backward_no_dx", &gated_residual_backward_no_dx_entrypoint);
    m.def("gated_residual_backward_no_dx_db", &gated_residual_backward_no_dx_db_entrypoint);
    m.def("layernorm_adaln", &layernorm_adaln_entrypoint);
    m.def("layernorm_adaln_persistent", &layernorm_adaln_persistent_entrypoint);
    m.def("layernorm_adaln_warp4", &layernorm_adaln_warp4_entrypoint);
    m.def("layernorm_stats", &layernorm_stats_entrypoint);
    m.def("layernorm_adaln_backward", &layernorm_adaln_backward_entrypoint);
    m.def("layernorm_adaln_backward_warp4", &layernorm_adaln_backward_warp4_entrypoint);
    m.def("gemm_linear_ln_adaln", &gemm_ln_adaln_entrypoint<false>);
    m.def("gemm_gelu_ln_adaln", &gemm_ln_adaln_entrypoint<true>);
    m.def("gemm_custom_adaln", &gemm_custom_adaln_entrypoint);
}
#endif
