#define TORCH_COMPILE
#include "../fp8_h100/fp8_h100_gemm.cu"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <pybind11/pybind11.h>
#include <limits>

namespace {

constexpr int K1024 = 1024;
constexpr int K4096 = 4096;
constexpr float FP8_E4M3_MAX_F = 448.0f;

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float warp_vals[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) {
        warp_vals[warp] = v;
    }
    __syncthreads();
    v = threadIdx.x < 8 ? warp_vals[lane] : 0.0f;
    if (warp == 0) {
        v = warp_sum(v);
    }
    return v;
}

__device__ __forceinline__ float block_sum_all(float v) {
    __shared__ float result;
    float sum = block_sum(v);
    if (threadIdx.x == 0) {
        result = sum;
    }
    __syncthreads();
    return result;
}

__device__ __forceinline__ void block_sum2_all(float &a, float &b) {
    __shared__ float result_a;
    __shared__ float result_b;
    float sum_a = block_sum(a);
    __syncthreads();
    float sum_b = block_sum(b);
    if (threadIdx.x == 0) {
        result_a = sum_a;
        result_b = sum_b;
    }
    __syncthreads();
    a = result_a;
    b = result_b;
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float warp_vals[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) {
        warp_vals[warp] = v;
    }
    __syncthreads();
    v = threadIdx.x < 8 ? warp_vals[lane] : 0.0f;
    if (warp == 0) {
        v = warp_max(v);
    }
    return v;
}

__device__ __forceinline__ void atomic_max_positive_float(float *addr, float value) {
    atomicMax(reinterpret_cast<unsigned int *>(addr), __float_as_uint(value));
}

__global__ void ln_adaln_quantize_stats_k1024_kernel(
    fp8e4m3 *__restrict__ out,
    float *__restrict__ mean,
    float *__restrict__ rstd,
    float *__restrict__ global_amax,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ shift,
    const __nv_bfloat16 *__restrict__ scale,
    float inv_quant_scale,
    int rows,
    int tokens_per_sample,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int batch = row / tokens_per_sample;
    const __nv_bfloat16 *x_row = x + static_cast<size_t>(row) * K1024;
    const __nv_bfloat16 *shift_row = shift + static_cast<size_t>(batch) * K1024;
    const __nv_bfloat16 *scale_row = scale + static_cast<size_t>(batch) * K1024;
    fp8e4m3 *out_row = out + static_cast<size_t>(row) * K1024;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    #pragma unroll
    for (int col = threadIdx.x; col < K1024; col += blockDim.x) {
        float xv = __bfloat162float(x_row[col]);
        local_sum += xv;
        local_sumsq += xv * xv;
    }
    block_sum2_all(local_sum, local_sumsq);
    float inv_k = 1.0f / static_cast<float>(K1024);
    float mu = local_sum * inv_k;
    float variance = fmaxf(local_sumsq * inv_k - mu * mu, 0.0f);
    float rs = rsqrtf(variance + eps);
    __syncthreads();

    float local_amax = 0.0f;
    #pragma unroll
    for (int col = threadIdx.x; col < K1024; col += blockDim.x) {
        float xv = __bfloat162float(x_row[col]);
        float sh = __bfloat162float(shift_row[col]);
        float sc = __bfloat162float(scale_row[col]);
        float z = (xv - mu) * rs;
        float y = z * (1.0f + sc) + sh;
        local_amax = fmaxf(local_amax, fabsf(y));
        float q = fminf(fmaxf(y * inv_quant_scale, -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        out_row[col] = fp8e4m3(q);
    }

    float amax = block_max(local_amax);
    if (threadIdx.x == 0) {
        mean[row] = mu;
        rstd[row] = rs;
        atomic_max_positive_float(global_amax, amax);
    }
}

__global__ void ln_adaln_quantize_stats_delayed_k1024_kernel(
    fp8e4m3 *__restrict__ out,
    float *__restrict__ row_amax,
    float *__restrict__ mean,
    float *__restrict__ rstd,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ shift,
    const __nv_bfloat16 *__restrict__ scale,
    const float *__restrict__ quant_scale,
    int rows,
    int tokens_per_sample,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int batch = row / tokens_per_sample;
    const __nv_bfloat16 *x_row = x + static_cast<size_t>(row) * K1024;
    const __nv_bfloat16 *shift_row = shift + static_cast<size_t>(batch) * K1024;
    const __nv_bfloat16 *scale_row = scale + static_cast<size_t>(batch) * K1024;
    fp8e4m3 *out_row = out + static_cast<size_t>(row) * K1024;
    const float q_scale = quant_scale[0];

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    #pragma unroll
    for (int col = threadIdx.x; col < K1024; col += blockDim.x) {
        float xv = __bfloat162float(x_row[col]);
        local_sum += xv;
        local_sumsq += xv * xv;
    }
    block_sum2_all(local_sum, local_sumsq);
    float inv_k = 1.0f / static_cast<float>(K1024);
    float mu = local_sum * inv_k;
    float variance = fmaxf(local_sumsq * inv_k - mu * mu, 0.0f);
    float rs = rsqrtf(variance + eps);
    __syncthreads();

    float local_amax = 0.0f;
    #pragma unroll
    for (int col = threadIdx.x; col < K1024; col += blockDim.x) {
        float xv = __bfloat162float(x_row[col]);
        float sh = __bfloat162float(shift_row[col]);
        float sc = __bfloat162float(scale_row[col]);
        float z = (xv - mu) * rs;
        float y = z * (1.0f + sc) + sh;
        local_amax = fmaxf(local_amax, fabsf(y));
        float q = fminf(fmaxf(y * q_scale, -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        out_row[col] = fp8e4m3(q);
    }

    float amax = block_max(local_amax);
    if (threadIdx.x == 0) {
        row_amax[row] = amax;
        mean[row] = mu;
        rstd[row] = rs;
    }
}

__global__ void ln_adaln_quantize_stats_vec_delayed_k1024_kernel(
    fp8e4m3 *__restrict__ out,
    float *__restrict__ row_amax,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ shift,
    const __nv_bfloat16 *__restrict__ scale,
    const float *__restrict__ quant_scale,
    int rows,
    int tokens_per_sample,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int batch = row / tokens_per_sample;
    const auto *x_row = reinterpret_cast<const __nv_bfloat162 *>(x + static_cast<size_t>(row) * K1024);
    const auto *shift_row = reinterpret_cast<const __nv_bfloat162 *>(shift + static_cast<size_t>(batch) * K1024);
    const auto *scale_row = reinterpret_cast<const __nv_bfloat162 *>(scale + static_cast<size_t>(batch) * K1024);
    auto *out_row = out + static_cast<size_t>(row) * K1024;
    const float q_scale = quant_scale[0];

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    #pragma unroll
    for (int pair = threadIdx.x; pair < K1024 / 2; pair += blockDim.x) {
        float2 xv = __bfloat1622float2(x_row[pair]);
        local_sum += xv.x + xv.y;
        local_sumsq += xv.x * xv.x + xv.y * xv.y;
    }
    block_sum2_all(local_sum, local_sumsq);
    float inv_k = 1.0f / static_cast<float>(K1024);
    float mu = local_sum * inv_k;
    float variance = fmaxf(local_sumsq * inv_k - mu * mu, 0.0f);
    float rs = rsqrtf(variance + eps);
    __syncthreads();

    float local_amax = 0.0f;
    #pragma unroll
    for (int pair = threadIdx.x; pair < K1024 / 2; pair += blockDim.x) {
        float2 xv = __bfloat1622float2(x_row[pair]);
        float2 sh = __bfloat1622float2(shift_row[pair]);
        float2 sc = __bfloat1622float2(scale_row[pair]);
        float y0 = ((xv.x - mu) * rs) * (1.0f + sc.x) + sh.x;
        float y1 = ((xv.y - mu) * rs) * (1.0f + sc.y) + sh.y;
        local_amax = fmaxf(local_amax, fmaxf(fabsf(y0), fabsf(y1)));
        out_row[2 * pair + 0] = fp8e4m3(fminf(fmaxf(y0 * q_scale, -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F));
        out_row[2 * pair + 1] = fp8e4m3(fminf(fmaxf(y1 * q_scale, -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F));
    }

    float amax = block_max(local_amax);
    if (threadIdx.x == 0) {
        row_amax[row] = amax;
    }
}

__global__ void ln_adaln_quantize_precomputed_vec_k1024_kernel(
    fp8e4m3 *__restrict__ out,
    float *__restrict__ row_amax,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ shift,
    const __nv_bfloat16 *__restrict__ scale,
    const float *__restrict__ mean,
    const float *__restrict__ rstd,
    const float *__restrict__ quant_scale,
    int rows,
    int tokens_per_sample
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int batch = row / tokens_per_sample;
    const auto *x_row = reinterpret_cast<const __nv_bfloat162 *>(x + static_cast<size_t>(row) * K1024);
    const auto *shift_row = reinterpret_cast<const __nv_bfloat162 *>(shift + static_cast<size_t>(batch) * K1024);
    const auto *scale_row = reinterpret_cast<const __nv_bfloat162 *>(scale + static_cast<size_t>(batch) * K1024);
    auto *out_row = out + static_cast<size_t>(row) * K1024;
    const float mu = mean[row];
    const float rs = rstd[row];
    const float q_scale = quant_scale[0];

    float local_amax = 0.0f;
    #pragma unroll
    for (int pair = threadIdx.x; pair < K1024 / 2; pair += blockDim.x) {
        float2 xv = __bfloat1622float2(x_row[pair]);
        float2 sh = __bfloat1622float2(shift_row[pair]);
        float2 sc = __bfloat1622float2(scale_row[pair]);
        float y0 = ((xv.x - mu) * rs) * (1.0f + sc.x) + sh.x;
        float y1 = ((xv.y - mu) * rs) * (1.0f + sc.y) + sh.y;
        local_amax = fmaxf(local_amax, fmaxf(fabsf(y0), fabsf(y1)));
        out_row[2 * pair + 0] = fp8e4m3(fminf(fmaxf(y0 * q_scale, -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F));
        out_row[2 * pair + 1] = fp8e4m3(fminf(fmaxf(y1 * q_scale, -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F));
    }

    float amax = block_max(local_amax);
    if (threadIdx.x == 0) {
        row_amax[row] = amax;
    }
}

__global__ void ln_adaln_quantize_k1024_kernel(
    fp8e4m3 *__restrict__ out,
    float *__restrict__ global_amax,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ shift,
    const __nv_bfloat16 *__restrict__ scale,
    const float *__restrict__ mean,
    const float *__restrict__ rstd,
    float inv_quant_scale,
    int rows,
    int tokens_per_sample
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int batch = row / tokens_per_sample;
    const __nv_bfloat16 *x_row = x + static_cast<size_t>(row) * K1024;
    const __nv_bfloat16 *shift_row = shift + static_cast<size_t>(batch) * K1024;
    const __nv_bfloat16 *scale_row = scale + static_cast<size_t>(batch) * K1024;
    fp8e4m3 *out_row = out + static_cast<size_t>(row) * K1024;

    float mu = mean[row];
    float rs = rstd[row];
    float local_amax = 0.0f;

    #pragma unroll
    for (int col = threadIdx.x; col < K1024; col += blockDim.x) {
        float xv = __bfloat162float(x_row[col]);
        float sh = __bfloat162float(shift_row[col]);
        float sc = __bfloat162float(scale_row[col]);
        float z = (xv - mu) * rs;
        float y = z * (1.0f + sc) + sh;
        local_amax = fmaxf(local_amax, fabsf(y));
        float q = fminf(fmaxf(y * inv_quant_scale, -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        out_row[col] = fp8e4m3(q);
    }

    float amax = block_max(local_amax);
    if (threadIdx.x == 0) {
        atomic_max_positive_float(global_amax, amax);
    }
}

__device__ __forceinline__ float gelu_tanh_approx(float x) {
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    float x3 = x * x * x;
    float t;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(kAlpha * (x + kBeta * x3)));
    return 0.5f * x * (1.0f + t);
}

__global__ void bias_gelu_quantize_k4096_kernel(
    fp8e4m3 *__restrict__ out,
    float *__restrict__ row_amax,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ bias,
    const float *__restrict__ quant_scale,
    int rows
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const __nv_bfloat16 *x_row = x + static_cast<size_t>(row) * K4096;
    fp8e4m3 *out_row = out + static_cast<size_t>(row) * K4096;
    float q_scale = quant_scale[0];
    float local_amax = 0.0f;

    for (int pair = threadIdx.x; pair < K4096 / 2; pair += blockDim.x) {
        auto xv2 = *reinterpret_cast<const __nv_bfloat162 *>(x_row + 2 * pair);
        auto bv2 = *reinterpret_cast<const __nv_bfloat162 *>(bias + 2 * pair);
        float2 xv = __bfloat1622float2(xv2);
        float2 bv = __bfloat1622float2(bv2);

        float y0 = gelu_tanh_approx(xv.x + bv.x);
        float y1 = gelu_tanh_approx(xv.y + bv.y);
        local_amax = fmaxf(local_amax, fmaxf(fabsf(y0), fabsf(y1)));

        out_row[2 * pair + 0] = fp8e4m3(fminf(fmaxf(y0 * q_scale, -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F));
        out_row[2 * pair + 1] = fp8e4m3(fminf(fmaxf(y1 * q_scale, -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F));
    }

    float row_max = block_max(local_amax);
    if (threadIdx.x == 0) {
        row_amax[row] = row_max;
    }
}


template<bool WriteRowwise, bool WriteTranspose>
__global__ void bf16_quantize_transpose_delayed_kernel(
    fp8e4m3 *__restrict__ out,
    fp8e4m3 *__restrict__ out_t,
    float *__restrict__ row_amax,
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ quant_scale,
    int rows,
    int cols
) {
    constexpr int TILE = 16;
    __shared__ fp8e4m3 tile[TILE][TILE + 1];
    __shared__ float abs_tile[TILE][TILE];

    int local_col = threadIdx.x;
    int local_row = threadIdx.y;
    int row = blockIdx.y * TILE + local_row;
    int col = blockIdx.x * TILE + local_col;

    fp8e4m3 qval(0.0f);
    float abs_v = 0.0f;
    if (row < rows && col < cols) {
        float v = __bfloat162float(x[static_cast<size_t>(row) * cols + col]);
        abs_v = fabsf(v);
        float q = fminf(fmaxf(v * quant_scale[0], -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        qval = fp8e4m3(q);
        if constexpr (WriteRowwise) {
            out[static_cast<size_t>(row) * cols + col] = qval;
        }
    }
    tile[local_row][local_col] = qval;
    abs_tile[local_row][local_col] = abs_v;
    __syncthreads();

    if (local_col == 0 && row < rows) {
        float local_amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            local_amax = fmaxf(local_amax, abs_tile[local_row][i]);
        }
        atomic_max_positive_float(row_amax + row, local_amax);
    }

    if constexpr (WriteTranspose) {
        int trans_row = blockIdx.x * TILE + local_row;
        int trans_col = blockIdx.y * TILE + local_col;
        if (trans_row < cols && trans_col < rows) {
            out_t[static_cast<size_t>(trans_row) * rows + trans_col] = tile[local_col][local_row];
        }
    }
}



template<bool WriteRowwise, bool WriteTranspose>
__global__ void bf16_quantize_transpose_db_delayed_kernel(
    fp8e4m3 *__restrict__ out,
    fp8e4m3 *__restrict__ out_t,
    float *__restrict__ row_amax,
    float *__restrict__ db,
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ quant_scale,
    int rows,
    int cols
) {
    constexpr int TILE = 16;
    __shared__ fp8e4m3 tile[TILE][TILE + 1];
    __shared__ float abs_tile[TILE][TILE];
    __shared__ float val_tile[TILE][TILE];

    int local_col = threadIdx.x;
    int local_row = threadIdx.y;
    int row = blockIdx.y * TILE + local_row;
    int col = blockIdx.x * TILE + local_col;

    fp8e4m3 qval(0.0f);
    float abs_v = 0.0f;
    float v = 0.0f;
    if (row < rows && col < cols) {
        v = __bfloat162float(x[static_cast<size_t>(row) * cols + col]);
        abs_v = fabsf(v);
        float q = fminf(fmaxf(v * quant_scale[0], -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        qval = fp8e4m3(q);
        if constexpr (WriteRowwise) {
            out[static_cast<size_t>(row) * cols + col] = qval;
        }
    }
    tile[local_row][local_col] = qval;
    abs_tile[local_row][local_col] = abs_v;
    val_tile[local_row][local_col] = v;
    __syncthreads();

    if (local_col == 0 && row < rows) {
        float local_amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            local_amax = fmaxf(local_amax, abs_tile[local_row][i]);
        }
        atomic_max_positive_float(row_amax + row, local_amax);
    }

    if (local_row == 0 && col < cols) {
        float local_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            local_sum += val_tile[i][local_col];
        }
        atomicAdd(db + col, local_sum);
    }

    if constexpr (WriteTranspose) {
        int trans_row = blockIdx.x * TILE + local_row;
        int trans_col = blockIdx.y * TILE + local_col;
        if (trans_row < cols && trans_col < rows) {
            out_t[static_cast<size_t>(trans_row) * rows + trans_col] = tile[local_col][local_row];
        }
    }
}

template<bool WriteRowwise, bool WriteTranspose>
__global__ void gate_bwd_quantize_transpose_delayed_kernel(
    fp8e4m3 *__restrict__ out,
    fp8e4m3 *__restrict__ out_t,
    float *__restrict__ row_amax,
    float *__restrict__ dgate,
    const __nv_bfloat16 *__restrict__ grad_out,
    const __nv_bfloat16 *__restrict__ branch_out,
    const __nv_bfloat16 *__restrict__ gate,
    const float *__restrict__ quant_scale,
    int rows,
    int cols,
    int tokens_per_sample
) {
    constexpr int TILE = 16;
    __shared__ fp8e4m3 tile[TILE][TILE + 1];
    __shared__ float abs_tile[TILE][TILE];
    __shared__ float gate_grad_tile[TILE][TILE];

    int local_col = threadIdx.x;
    int local_row = threadIdx.y;
    int row = blockIdx.y * TILE + local_row;
    int col = blockIdx.x * TILE + local_col;

    fp8e4m3 qval(0.0f);
    float abs_v = 0.0f;
    float gate_grad = 0.0f;
    if (row < rows && col < cols) {
        int batch = row / tokens_per_sample;
        size_t idx = static_cast<size_t>(row) * cols + col;
        float go = __bfloat162float(grad_out[idx]);
        float bo = __bfloat162float(branch_out[idx]);
        float gv = __bfloat162float(gate[static_cast<size_t>(batch) * cols + col]);
        float branch_grad = go * gv;
        gate_grad = go * bo;
        abs_v = fabsf(branch_grad);
        float q = fminf(fmaxf(branch_grad * quant_scale[0], -FP8_E4M3_MAX_F), FP8_E4M3_MAX_F);
        qval = fp8e4m3(q);
        if constexpr (WriteRowwise) {
            out[idx] = qval;
        }
    }
    tile[local_row][local_col] = qval;
    abs_tile[local_row][local_col] = abs_v;
    gate_grad_tile[local_row][local_col] = gate_grad;
    __syncthreads();

    if (local_col == 0 && row < rows) {
        float local_amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            local_amax = fmaxf(local_amax, abs_tile[local_row][i]);
        }
        atomic_max_positive_float(row_amax + row, local_amax);
    }

    if (local_row == 0) {
        int base_row = blockIdx.y * TILE;
        int batch = base_row / tokens_per_sample;
        if (col < cols) {
            float local_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < TILE; ++i) {
                local_sum += gate_grad_tile[i][local_col];
            }
            atomicAdd(dgate + static_cast<size_t>(batch) * cols + col, local_sum);
        }
    }

    if constexpr (WriteTranspose) {
        int trans_row = blockIdx.x * TILE + local_row;
        int trans_col = blockIdx.y * TILE + local_col;
        if (trans_row < cols && trans_col < rows) {
            out_t[static_cast<size_t>(trans_row) * rows + trans_col] = tile[local_col][local_row];
        }
    }
}

__global__ void delayed_scaling_update_kernel(
    float *__restrict__ scale,
    float *__restrict__ scale_inv,
    float *__restrict__ amax_history,
    int *__restrict__ hist_idx,
    float *__restrict__ global_amax,
    const float *__restrict__ row_amax,
    int rows,
    int history_len,
    float eps
) {
    float local = 0.0f;
    for (int idx = threadIdx.x; idx < rows; idx += blockDim.x) {
        local = fmaxf(local, row_amax[idx]);
    }
    float current_amax = block_max(local);
    if (threadIdx.x == 0) {
        int slot = hist_idx[0];
        if (slot < 0 || slot >= history_len) {
            slot = 0;
        }
        amax_history[slot] = current_amax;
        hist_idx[0] = (slot + 1) % history_len;

        float hist_amax = 0.0f;
        for (int i = 0; i < history_len; ++i) {
            hist_amax = fmaxf(hist_amax, amax_history[i]);
        }
        float safe_amax = fmaxf(hist_amax, eps);
        scale[0] = FP8_E4M3_MAX_F / safe_amax;
        scale_inv[0] = safe_amax / FP8_E4M3_MAX_F;
        global_amax[0] = current_amax;
    }
}

__global__ void reduce_amax_kernel(
    float *__restrict__ block_amax,
    const float *__restrict__ row_amax,
    int rows
) {
    float local = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows; idx += blockDim.x * gridDim.x) {
        local = fmaxf(local, row_amax[idx]);
    }
    float amax = block_max(local);
    if (threadIdx.x == 0) {
        block_amax[blockIdx.x] = amax;
    }
}

__global__ void finalize_amax_kernel(
    float *__restrict__ global_amax,
    const float *__restrict__ block_amax,
    int blocks
) {
    float local = 0.0f;
    for (int idx = threadIdx.x; idx < blocks; idx += blockDim.x) {
        local = fmaxf(local, block_amax[idx]);
    }
    float amax = block_max(local);
    if (threadIdx.x == 0) {
        global_amax[0] = amax;
    }
}

#define TK_FP8_ACC_OPERANDS(d) \
    "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), \
    "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), \
    "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), \
    "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
    "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), \
    "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), \
    "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), \
    "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), \
    "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), \
    "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), \
    "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), \
    "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), \
    "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), \
    "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), \
    "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), \
    "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])

#define TK_FP8_ACC_OPERANDS_32(d) \
    "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), \
    "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), \
    "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), \
    "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
    "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), \
    "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), \
    "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), \
    "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])

__device__ __forceinline__ void fp8_wgmma_fence(float (&accum)[64]) {
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        asm volatile("" : "+f"(accum[i]) :: "memory");
    }
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void fp8_wgmma_fence(float (&accum)[32]) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        asm volatile("" : "+f"(accum[i]) :: "memory");
    }
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void fp8_wgmma_m64n128k32_ss(
    float (&accum)[64],
    uint64_t a_desc,
    uint64_t b_desc,
    int scale_d
) {
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p, 1, %67;\n"
        "}\n"
        : TK_FP8_ACC_OPERANDS(accum)
        : "l"(a_desc), "l"(b_desc), "r"(scale_d), "n"(1)
    );
}

__device__ __forceinline__ void fp8_wgmma_m64n64k32_ss(
    float (&accum)[32],
    uint64_t a_desc,
    uint64_t b_desc,
    int scale_d
) {
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, %35;\n"
        "}\n"
        : TK_FP8_ACC_OPERANDS_32(accum)
        : "l"(a_desc), "l"(b_desc), "r"(scale_d), "n"(1)
    );
}

__device__ __forceinline__ void zero_accum(float (&accum)[64]) {
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        accum[i] = 0.0f;
    }
}

__device__ __forceinline__ void zero_accum(float (&accum)[32]) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        accum[i] = 0.0f;
    }
}

__device__ __forceinline__ void promote_accum(float (&dst)[64], const float (&src)[64]) {
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        dst[i] += src[i];
    }
}

__device__ __forceinline__ void promote_accum(float (&dst)[32], const float (&src)[32]) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        dst[i] += src[i];
    }
}

template <typename ST>
__device__ __forceinline__ void store_accum_bf16_64x128(ST &dst, const float (&accum)[64], float dequant_scale) {
    const int lane = ::kittens::laneid();
    const int row = warpgroup::warpid() * 16 + lane / 4;
    const int col_pair = 2 * (lane % 4);

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 16 + col_pair;
        dst[{row, col + 0}] = __float2bfloat16(accum[j * 8 + 0] * dequant_scale);
        dst[{row, col + 1}] = __float2bfloat16(accum[j * 8 + 1] * dequant_scale);
        dst[{row + 8, col + 0}] = __float2bfloat16(accum[j * 8 + 2] * dequant_scale);
        dst[{row + 8, col + 1}] = __float2bfloat16(accum[j * 8 + 3] * dequant_scale);
        dst[{row, col + 8}] = __float2bfloat16(accum[j * 8 + 4] * dequant_scale);
        dst[{row, col + 9}] = __float2bfloat16(accum[j * 8 + 5] * dequant_scale);
        dst[{row + 8, col + 8}] = __float2bfloat16(accum[j * 8 + 6] * dequant_scale);
        dst[{row + 8, col + 9}] = __float2bfloat16(accum[j * 8 + 7] * dequant_scale);
    }
}

template <typename ST>
__device__ __forceinline__ void store_accum_bf16_64x64(ST &dst, const float (&accum)[32], float dequant_scale) {
    const int lane = ::kittens::laneid();
    const int row = warpgroup::warpid() * 16 + lane / 4;
    const int col_pair = 2 * (lane % 4);

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = j * 16 + col_pair;
        dst[{row, col + 0}] = __float2bfloat16(accum[j * 8 + 0] * dequant_scale);
        dst[{row, col + 1}] = __float2bfloat16(accum[j * 8 + 1] * dequant_scale);
        dst[{row + 8, col + 0}] = __float2bfloat16(accum[j * 8 + 2] * dequant_scale);
        dst[{row + 8, col + 1}] = __float2bfloat16(accum[j * 8 + 3] * dequant_scale);
        dst[{row, col + 8}] = __float2bfloat16(accum[j * 8 + 4] * dequant_scale);
        dst[{row, col + 9}] = __float2bfloat16(accum[j * 8 + 5] * dequant_scale);
        dst[{row + 8, col + 8}] = __float2bfloat16(accum[j * 8 + 6] * dequant_scale);
        dst[{row + 8, col + 9}] = __float2bfloat16(accum[j * 8 + 7] * dequant_scale);
    }
}

template<kittens::ducks::sv::all SV>
__device__ __forceinline__ void init_bias(rt_fl<16, SV::length> &acc, const SV &bias) {
    #pragma unroll
    for (int i = 0; i < SV::tiles; i++) {
        float2 tmp1 = __bfloat1622float2(*(bf16_2*)&bias.data[16 * i + 0 + 2 * (laneid() % 4)]);
        acc.tiles[0][i].data[0].x = tmp1.x;
        acc.tiles[0][i].data[0].y = tmp1.y;
        acc.tiles[0][i].data[1].x = tmp1.x;
        acc.tiles[0][i].data[1].y = tmp1.y;
        float2 tmp2 = __bfloat1622float2(*(bf16_2*)&bias.data[16 * i + 8 + 2 * (laneid() % 4)]);
        acc.tiles[0][i].data[2].x = tmp2.x;
        acc.tiles[0][i].data[2].y = tmp2.y;
        acc.tiles[0][i].data[3].x = tmp2.x;
        acc.tiles[0][i].data[3].y = tmp2.y;
    }
}

} // namespace


namespace fp8_fp32out {

struct matmul_layout {
    using  a_tile         = st_fp8e4m3<64,  128>;
    using  b_tile         = st_fp8e4m3<128, 128>;
    using  c_tile         = st_fl<64,  128>;
    using  a_layout       = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout       = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout       = gl<float, 1, 1, -1, -1, c_tile>;
    struct globals        { a_layout A; b_layout B; c_layout C; };
    struct input_block    { a_tile a[2]; b_tile b; };
    struct finish_block   { c_tile c[2]; };
    struct common_state   { int2 coord; };
    struct consumer_state {
        rt_fl<16, c_tile::cols> tc_accum;
        rt_fl<16, c_tile::cols> fp32_accum;
    };
};

template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 128 : M*N/(2*layout::c_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        if (task_id < super_rows * Cblocks) {
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        } else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A, {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B, {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            warp::zero(args.state.tc_accum);
            warp::zero(args.state.fp32_accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.tc_accum,
                args.input.a[warpgroup::groupid()],
                args.input.b
            );
            warpgroup::mma_async_wait();
            warp::add(args.state.fp32_accum, args.state.fp32_accum, args.state.tc_accum);
            warp::zero(args.state.tc_accum);
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.fp32_accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()], {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            warp::zero(args.state.fp32_accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, float *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

} // namespace fp8_fp32out

namespace fp8_bf16out {

struct matmul_layout {
    using  a_tile         = st_fp8e4m3<64,  128>;
    using  b_tile         = st_fp8e4m3<128, 128>;
    using  c_tile         = st_bf<64,  128>;
    using  a_layout       = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout       = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout       = gl<bf16, 1, 1, -1, -1, c_tile>;
    struct globals        { a_layout A; b_layout B; c_layout C; };
    struct input_block    { a_tile a[2]; b_tile b; };
    struct finish_block   { c_tile c[2]; };
    struct common_state   { int2 coord; };
    struct consumer_state {
        rt_fl<16, c_tile::cols> tc_accum;
        rt_fl<16, c_tile::cols> fp32_accum;
    };
};

template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 128 : M*N/(2*layout::c_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        if (task_id < super_rows * Cblocks) {
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        } else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A, {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B, {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            warp::zero(args.state.tc_accum);
            warp::zero(args.state.fp32_accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.tc_accum,
                args.input.a[warpgroup::groupid()],
                args.input.b
            );
            warpgroup::mma_async_wait();
            warp::add(args.state.fp32_accum, args.state.fp32_accum, args.state.tc_accum);
            warp::zero(args.state.tc_accum);
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.fp32_accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()], {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            warp::zero(args.state.fp32_accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

} // namespace fp8_bf16out

namespace fp8_bf16out_scaled {

struct matmul_layout {
    using  a_tile         = st_fp8e4m3<64,  128>;
    using  b_tile         = st_fp8e4m3<128, 128>;
    using  c_tile         = st_bf<64,  128>;
    using  a_layout       = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout       = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout       = gl<bf16, 1, 1, -1, -1, c_tile>;
    struct globals        { a_layout A; b_layout B; c_layout C; float dequant_scale; };
    struct input_block    { a_tile a[2]; b_tile b; };
    struct finish_block   { c_tile c[2]; };
    struct common_state   { int2 coord; };
    struct consumer_state {
        rt_fl<16, c_tile::cols> tc_accum;
        rt_fl<16, c_tile::cols> fp32_accum;
    };
};

template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 128 : M*N/(2*layout::c_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks) {
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        } else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A, {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B, {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            warp::zero(args.state.tc_accum);
            warp::zero(args.state.fp32_accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.tc_accum,
                args.input.a[warpgroup::groupid()],
                args.input.b
            );
            warpgroup::mma_async_wait();
            warp::add(args.state.fp32_accum, args.state.fp32_accum, args.state.tc_accum);
            warp::zero(args.state.tc_accum);
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warp::mul(args.state.fp32_accum, args.state.fp32_accum, args.globals.dequant_scale);
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.fp32_accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()], {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            warp::zero(args.state.fp32_accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, float dequant_scale, dim3 grid, dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg, dequant_scale};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

} // namespace fp8_bf16out_scaled

namespace fp8_bf16out_wide_scaled {

struct matmul_layout {
    using  a_tile         = st_fp8e4m3<64,  128>;
    using  b_tile         = st_fp8e4m3<256, 128>;
    using  c_tile         = st_bf<64,  256>;
    using  a_layout       = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout       = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout       = gl<bf16, 1, 1, -1, -1, c_tile>;
    struct globals        { a_layout A; b_layout B; c_layout C; float dequant_scale; };
    struct input_block    { a_tile a[2]; b_tile b; };
    struct finish_block   { c_tile c[2]; };
    struct common_state   { int2 coord; };
    struct consumer_state {
        rt_fl<16, c_tile::cols> accum;
    };
};

template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 128 : M*N/(2*layout::c_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks) {
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        } else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A, {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B, {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                args.input.b
            );
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warp::mul(args.state.accum, args.state.accum, args.globals.dequant_scale);
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()], {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, float dequant_scale, dim3 grid, dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg, dequant_scale};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

} // namespace fp8_bf16out_wide_scaled

namespace fp8_bf16out_deepaccum {

struct matmul_layout {
    using  a_tile         = st_fp8e4m3<64,  128>;
    using  b_tile         = st_fp8e4m3<128, 128>;
    using  c_tile         = st_bf<64,  128>;
    using  a_layout       = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout       = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout       = gl<bf16, 1, 1, -1, -1, c_tile>;
    struct globals        { a_layout A; b_layout B; c_layout C; float dequant_scale; };
    struct input_block    { a_tile a[2]; b_tile b; };
    struct finish_block   { c_tile c[2]; };
    struct common_state   { int2 coord; };
    struct consumer_state {
        float fp32_accum[64];
    };
};

template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 128 : M*N/(2*layout::c_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks) {
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        } else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A, {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B, {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            zero_accum(args.state.fp32_accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            kittens::st_descriptor<typename layout::a_tile, 0> a_desc(args.input.a[warpgroup::groupid()]);
            kittens::st_descriptor<layout::b_tile, 0> b_desc(args.input.b);

            float tc_accum[64];
            zero_accum(tc_accum);
            fp8_wgmma_fence(tc_accum);
            fp8_wgmma_m64n128k32_ss(tc_accum, a_desc.chunk_descriptor(0), b_desc.chunk_descriptor(0), 0);
            #pragma unroll
            for (int k = 1; k < 4; ++k) {
                fp8_wgmma_m64n128k32_ss(tc_accum, a_desc.chunk_descriptor(k), b_desc.chunk_descriptor(k), 1);
            }
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            promote_accum(args.state.fp32_accum, tc_accum);
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            store_accum_bf16_64x128(args.finish.c[warpgroup::groupid()], args.state.fp32_accum, args.globals.dequant_scale);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()], {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            zero_accum(args.state.fp32_accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, float dequant_scale, dim3 grid, dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg, dequant_scale};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

} // namespace fp8_bf16out_deepaccum

namespace fp8_bf16out_deepaccum_n64 {

struct matmul_layout {
    using  a_tile         = st_fp8e4m3<64, 128>;
    using  b_tile         = st_fp8e4m3<64, 128>;
    using  c_tile         = st_bf<64, 64>;
    using  a_layout       = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout       = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout       = gl<bf16, 1, 1, -1, -1, c_tile>;
    struct globals        { a_layout A; b_layout B; c_layout C; float dequant_scale; };
    struct input_block    { a_tile a[2]; b_tile b; };
    struct finish_block   { c_tile c[2]; };
    struct common_state   { int2 coord; };
    struct consumer_state {
        float fp32_accum[32];
    };
};

template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 128 : M*N/(2*layout::c_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks) {
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        } else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A, {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B, {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            zero_accum(args.state.fp32_accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            kittens::st_descriptor<typename layout::a_tile, 0> a_desc(args.input.a[warpgroup::groupid()]);
            kittens::st_descriptor<layout::b_tile, 0> b_desc(args.input.b);

            float tc_accum[32];
            zero_accum(tc_accum);
            fp8_wgmma_fence(tc_accum);
            fp8_wgmma_m64n64k32_ss(tc_accum, a_desc.chunk_descriptor(0), b_desc.chunk_descriptor(0), 0);
            #pragma unroll
            for (int k = 1; k < 4; ++k) {
                fp8_wgmma_m64n64k32_ss(tc_accum, a_desc.chunk_descriptor(k), b_desc.chunk_descriptor(k), 1);
            }
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            promote_accum(args.state.fp32_accum, tc_accum);
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            store_accum_bf16_64x64(args.finish.c[warpgroup::groupid()], args.state.fp32_accum, args.globals.dequant_scale);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()], {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            zero_accum(args.state.fp32_accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, float dequant_scale, dim3 grid, dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg, dequant_scale};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

} // namespace fp8_bf16out_deepaccum_n64

namespace fp8_bf16out_pipe {

struct matmul_layout {
    using  a_tile         = st_fp8e4m3<64,  128>;
    using  b_tile         = st_fp8e4m3<128, 128>;
    using  c_tile         = st_bf<64,  128>;
    using  a_layout       = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout       = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout       = gl<bf16, 1, 1, -1, -1, c_tile>;
    struct globals        { a_layout A; b_layout B; c_layout C; };
    struct input_block    { a_tile a[2]; b_tile b; };
    struct finish_block   { c_tile c[2]; };
    struct common_state   { int2 coord; int num_iters; };
    struct consumer_state {
        rt_fl<16, c_tile::cols> tc_accum_even;
        rt_fl<16, c_tile::cols> tc_accum_odd;
        rt_fl<16, c_tile::cols> fp32_accum;
    };
};

template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    static constexpr int CONSUMER_WGMMA_DEPTH=1;

    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 128 : M*N/(2*layout::c_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks) {
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        } else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        args.common.num_iters = args.num_iters;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A, {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B, {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            warp::zero(args.state.tc_accum_even);
            warp::zero(args.state.tc_accum_odd);
            warp::zero(args.state.fp32_accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            if ((args.iter & 1) == 0) {
                warp::add(args.state.fp32_accum, args.state.fp32_accum, args.state.tc_accum_even);
                warp::zero(args.state.tc_accum_even);
                warpgroup::mma_ABt(
                    args.state.tc_accum_even,
                    args.input.a[warpgroup::groupid()],
                    args.input.b
                );
            } else {
                warp::add(args.state.fp32_accum, args.state.fp32_accum, args.state.tc_accum_odd);
                warp::zero(args.state.tc_accum_odd);
                warpgroup::mma_ABt(
                    args.state.tc_accum_odd,
                    args.input.a[warpgroup::groupid()],
                    args.input.b
                );
            }
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warp::add(args.state.fp32_accum, args.state.fp32_accum, args.state.tc_accum_even);
            warp::add(args.state.fp32_accum, args.state.fp32_accum, args.state.tc_accum_odd);
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.fp32_accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()], {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            warp::zero(args.state.tc_accum_even);
            warp::zero(args.state.tc_accum_odd);
            warp::zero(args.state.fp32_accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

} // namespace fp8_bf16out_pipe

namespace fp8_bf16out_pipe64 {

struct matmul_layout {
    using  a_tile         = st_fp8e4m3<64, 128>;
    using  b_tile         = st_fp8e4m3<64, 128>;
    using  c_tile         = st_bf<64, 64>;
    using  a_layout       = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout       = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout       = gl<bf16, 1, 1, -1, -1, c_tile>;
    struct globals        { a_layout A; b_layout B; c_layout C; };
    struct input_block    { a_tile a[2]; b_tile b; };
    struct finish_block   { c_tile c[2]; };
    struct common_state   { int2 coord; };
    struct consumer_state {
        rt_fl<16, c_tile::cols> tc_accum_even;
        rt_fl<16, c_tile::cols> tc_accum_odd;
        rt_fl<16, c_tile::cols> fp32_accum;
    };
};

template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    static constexpr int CONSUMER_WGMMA_DEPTH=1;

    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 128 : M*N/(2*layout::c_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks) {
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        } else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A, {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B, {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            warp::zero(args.state.tc_accum_even);
            warp::zero(args.state.tc_accum_odd);
            warp::zero(args.state.fp32_accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            if ((args.iter & 1) == 0) {
                warp::add(args.state.fp32_accum, args.state.fp32_accum, args.state.tc_accum_even);
                warp::zero(args.state.tc_accum_even);
                warpgroup::mma_ABt(args.state.tc_accum_even, args.input.a[warpgroup::groupid()], args.input.b);
            } else {
                warp::add(args.state.fp32_accum, args.state.fp32_accum, args.state.tc_accum_odd);
                warp::zero(args.state.tc_accum_odd);
                warpgroup::mma_ABt(args.state.tc_accum_odd, args.input.a[warpgroup::groupid()], args.input.b);
            }
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warp::add(args.state.fp32_accum, args.state.fp32_accum, args.state.tc_accum_even);
            warp::add(args.state.fp32_accum, args.state.fp32_accum, args.state.tc_accum_odd);
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.fp32_accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()], {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            warp::zero(args.state.tc_accum_even);
            warp::zero(args.state.tc_accum_odd);
            warp::zero(args.state.fp32_accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}


} // namespace fp8_bf16out_pipe64

namespace fp8_bf16out_bias {

struct matmul_layout {
    using  a_tile         = st_fp8e4m3<64, 128>;
    using  b_tile         = st_fp8e4m3<128, 128>;
    using  c_tile         = st_bf<64, 128>;
    using  bias_vec       = sv_bf<c_tile::cols>;
    using  a_layout       = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout       = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout       = gl<bf16, 1, 1, -1, -1, c_tile>;
    using  bias_layout    = gl<bf16, 1, 1, 1, -1, bias_vec>;
    struct globals        { a_layout A; b_layout B; c_layout C; bias_layout bias; };
    struct input_block    { a_tile a[2]; b_tile b; };
    struct scratch_block  { bias_vec bias; };
    struct finish_block   { c_tile c[2]; };
    struct common_state   { int2 coord; };
    struct consumer_state {
        rt_fl<16, c_tile::cols> fp32_accum;
    };
};

template<int _SUPER_M=12, cache_policy A_LOAD_POLICY=cache_policy::NORMAL, cache_policy B_LOAD_POLICY=cache_policy::NORMAL, cache_policy STORE_POLICY=cache_policy::NORMAL>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 128 : M*N/(2*layout::c_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        if (task_id < super_rows * Cblocks) {
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        } else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < 2; i++) {
                    tma::load_async<dim::ROW, A_LOAD_POLICY>(args.input.a[i], args.globals.A, {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async<dim::ROW, B_LOAD_POLICY>(args.input.b, args.globals.B, {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            group<NUM_CONSUMER_WARPS>::load(args.scratch.bias, args.globals.bias, {args.common.coord.y});
            group<NUM_CONSUMER_WARPS>::sync(0);
            init_bias(args.state.fp32_accum, args.scratch.bias);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.fp32_accum,
                args.input.a[warpgroup::groupid()],
                args.input.b
            );
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.fp32_accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) {
                tma::store_async<dim::ROW, STORE_POLICY>(args.globals.C, args.finish.c[warpgroup::groupid()], {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            warp::zero(args.state.fp32_accum);
            if (warpgroup::laneid() == 0) arrive(args.finish_finished);
        }
    };
};


template<int _SUPER_M=12, cache_policy A_LOAD_POLICY=cache_policy::NORMAL, cache_policy B_LOAD_POLICY=cache_policy::NORMAL, cache_policy STORE_POLICY=cache_policy::NORMAL>
struct matmul_cluster2_template : matmul_template<_SUPER_M, A_LOAD_POLICY, B_LOAD_POLICY, STORE_POLICY> {
    using base = matmul_template<_SUPER_M, A_LOAD_POLICY, B_LOAD_POLICY, STORE_POLICY>;
    using layout = typename base::layout;
    static constexpr int SUPER_M = _SUPER_M;
    static constexpr int CLUSTER_BLOCKS = 2;
    static constexpr int NUM_CONSUMER_WARPS = base::NUM_CONSUMER_WARPS;
    static constexpr int INPUT_PIPE_STAGES = base::INPUT_PIPE_STAGES;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = base::PRODUCER_BARRIER_ARRIVALS;

    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        int blocks = PERISISTENT_GRID ? 128 : M*N/(2*layout::c_tile::num_elements);
        return dim3((blocks + CLUSTER_BLOCKS - 1) / CLUSTER_BLOCKS * CLUSTER_BLOCKS);
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int cta_rank = cluster_ctarank();
        int cluster_blocks = gridDim.x / CLUSTER_BLOCKS;
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows);
        int Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int Rclusters = Rblocks / CLUSTER_BLOCKS;
        int super_rows = (Rclusters/SUPER_M)*SUPER_M,
            final_rows = Rclusters - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*cluster_blocks + blockIdx.x / CLUSTER_BLOCKS;
        int2 cluster_coord;
        if (task_id < super_rows * Cblocks) {
            cluster_coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        } else if (task_id < Rclusters*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            cluster_coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { (cluster_coord.x*CLUSTER_BLOCKS + cta_rank)*2 + id, cluster_coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::elect_leader()) {
                tma::expect(args.inputs_arrived, args.input);
                int cta_rank = cluster_ctarank();
                for (int i = 0; i < 2; i++) {
                    tma::load_async<dim::ROW, A_LOAD_POLICY>(args.input.a[i], args.globals.A, {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                if (cta_rank == 0) {
                    tma::cluster::load_async<dim::ROW, B_LOAD_POLICY>(args.input.b, args.globals.B, {args.common.coord.y, args.iter}, args.inputs_arrived, static_cast<uint16_t>(0x3));
                }
            }
        }
    };
    using consumer = typename base::consumer;
};

template<typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, bf16 *d_C, bf16 *d_bias, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using bias_layout = typename mmt::layout::bias_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    bias_layout Biasg{d_bias, nullptr, nullptr, nullptr, N};
    globals G{Ag, Bg, Cg, Biasg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

template<typename mmt>
void inner_run_cluster(fp8e4m3 *d_A, fp8e4m3 *d_B, bf16 *d_C, bf16 *d_bias, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using bias_layout = typename mmt::layout::bias_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    bias_layout Biasg{d_bias, nullptr, nullptr, nullptr, N};
    globals G{Ag, Bg, Cg, Biasg};
    LaunchConfig<true> launch_config(grid, block, MAX_SHARED_MEMORY-1024, 0, dim3(mmt::CLUSTER_BLOCKS, 1, 1));
    cudaLaunchKernelEx(launch_config, prototype::lcf::kernel<mmt>, G);
}

} // namespace fp8_bf16out_bias

std::vector<at::Tensor> ln_adaln_quantize_k1024(
    const at::Tensor &x,
    const at::Tensor &shift,
    const at::Tensor &scale,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    int64_t tokens_per_sample,
    double inv_quant_scale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(shift);
    CHECK_INPUT(scale);
    CHECK_INPUT(mean);
    CHECK_INPUT(rstd);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(shift.scalar_type() == at::ScalarType::BFloat16, "shift must be bf16");
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::BFloat16, "scale must be bf16");
    TORCH_CHECK(mean.scalar_type() == at::ScalarType::Float, "mean must be fp32");
    TORCH_CHECK(rstd.scalar_type() == at::ScalarType::Float, "rstd must be fp32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K1024, "x must have shape [M, 1024]");
    TORCH_CHECK(shift.dim() == 2 && shift.size(1) == K1024, "shift must have shape [B, 1024]");
    TORCH_CHECK(scale.sizes() == shift.sizes(), "scale shape must match shift");
    TORCH_CHECK(mean.dim() == 1 && mean.size(0) == x.size(0), "mean must have shape [M]");
    TORCH_CHECK(rstd.dim() == 1 && rstd.size(0) == x.size(0), "rstd must have shape [M]");
    TORCH_CHECK(tokens_per_sample > 0, "tokens_per_sample must be positive");
    TORCH_CHECK(x.size(0) == shift.size(0) * tokens_per_sample, "M must equal B * tokens_per_sample");

    auto out = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto global_amax = at::empty({1}, x.options().dtype(at::ScalarType::Float));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA_ERROR(cudaMemsetAsync(global_amax.data_ptr<float>(), 0, sizeof(float), stream));
    ln_adaln_quantize_k1024_kernel<<<x.size(0), 256, 0, stream>>>(
        reinterpret_cast<fp8e4m3 *>(out.data_ptr<c10::Float8_e4m3fn>()),
        global_amax.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(shift.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(scale.data_ptr<at::BFloat16>()),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        static_cast<float>(inv_quant_scale),
        static_cast<int>(x.size(0)),
        static_cast<int>(tokens_per_sample)
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {out, global_amax};
}

std::vector<at::Tensor> ln_adaln_quantize_stats_k1024(
    const at::Tensor &x,
    const at::Tensor &shift,
    const at::Tensor &scale,
    int64_t tokens_per_sample,
    double inv_quant_scale,
    double eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(shift);
    CHECK_INPUT(scale);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(shift.scalar_type() == at::ScalarType::BFloat16, "shift must be bf16");
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::BFloat16, "scale must be bf16");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K1024, "x must have shape [M, 1024]");
    TORCH_CHECK(shift.dim() == 2 && shift.size(1) == K1024, "shift must have shape [B, 1024]");
    TORCH_CHECK(scale.sizes() == shift.sizes(), "scale shape must match shift");
    TORCH_CHECK(tokens_per_sample > 0, "tokens_per_sample must be positive");
    TORCH_CHECK(x.size(0) == shift.size(0) * tokens_per_sample, "M must equal B * tokens_per_sample");

    auto out = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto mean = at::empty({x.size(0)}, x.options().dtype(at::ScalarType::Float));
    auto rstd = at::empty_like(mean);
    auto global_amax = at::empty({1}, x.options().dtype(at::ScalarType::Float));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA_ERROR(cudaMemsetAsync(global_amax.data_ptr<float>(), 0, sizeof(float), stream));
    ln_adaln_quantize_stats_k1024_kernel<<<x.size(0), 256, 0, stream>>>(
        reinterpret_cast<fp8e4m3 *>(out.data_ptr<c10::Float8_e4m3fn>()),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        global_amax.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(shift.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(scale.data_ptr<at::BFloat16>()),
        static_cast<float>(inv_quant_scale),
        static_cast<int>(x.size(0)),
        static_cast<int>(tokens_per_sample),
        static_cast<float>(eps)
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {out, global_amax, mean, rstd};
}

std::vector<at::Tensor> ln_adaln_quantize_stats_delayed_k1024(
    const at::Tensor &x,
    const at::Tensor &shift,
    const at::Tensor &scale,
    const at::Tensor &quant_scale,
    int64_t tokens_per_sample,
    double eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(shift);
    CHECK_INPUT(scale);
    CHECK_INPUT(quant_scale);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(shift.scalar_type() == at::ScalarType::BFloat16, "shift must be bf16");
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::BFloat16, "scale must be bf16");
    TORCH_CHECK(quant_scale.scalar_type() == at::ScalarType::Float, "quant_scale must be fp32");
    TORCH_CHECK(quant_scale.numel() == 1, "quant_scale must contain one scalar");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K1024, "x must have shape [M, 1024]");
    TORCH_CHECK(shift.dim() == 2 && shift.size(1) == K1024, "shift must have shape [B, 1024]");
    TORCH_CHECK(scale.sizes() == shift.sizes(), "scale shape must match shift");
    TORCH_CHECK(tokens_per_sample > 0, "tokens_per_sample must be positive");
    TORCH_CHECK(x.size(0) == shift.size(0) * tokens_per_sample, "M must equal B * tokens_per_sample");

    auto out = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto row_amax = at::empty({x.size(0)}, x.options().dtype(at::ScalarType::Float));
    auto mean = at::empty({x.size(0)}, x.options().dtype(at::ScalarType::Float));
    auto rstd = at::empty_like(mean);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ln_adaln_quantize_stats_delayed_k1024_kernel<<<x.size(0), 256, 0, stream>>>(
        reinterpret_cast<fp8e4m3 *>(out.data_ptr<c10::Float8_e4m3fn>()),
        row_amax.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(shift.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(scale.data_ptr<at::BFloat16>()),
        quant_scale.data_ptr<float>(),
        static_cast<int>(x.size(0)),
        static_cast<int>(tokens_per_sample),
        static_cast<float>(eps)
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {out, row_amax, mean, rstd};
}

std::vector<at::Tensor> ln_adaln_quantize_stats_vec_delayed_k1024(
    const at::Tensor &x,
    const at::Tensor &shift,
    const at::Tensor &scale,
    const at::Tensor &quant_scale,
    int64_t tokens_per_sample,
    double eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(shift);
    CHECK_INPUT(scale);
    CHECK_INPUT(quant_scale);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(shift.scalar_type() == at::ScalarType::BFloat16, "shift must be bf16");
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::BFloat16, "scale must be bf16");
    TORCH_CHECK(quant_scale.scalar_type() == at::ScalarType::Float, "quant_scale must be fp32");
    TORCH_CHECK(quant_scale.numel() == 1, "quant_scale must contain one scalar");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K1024, "x must have shape [M, 1024]");
    TORCH_CHECK(shift.dim() == 2 && shift.size(1) == K1024, "shift must have shape [B, 1024]");
    TORCH_CHECK(scale.sizes() == shift.sizes(), "scale shape must match shift");
    TORCH_CHECK(tokens_per_sample > 0, "tokens_per_sample must be positive");
    TORCH_CHECK(x.size(0) == shift.size(0) * tokens_per_sample, "M must equal B * tokens_per_sample");

    auto out = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto row_amax = at::empty({x.size(0)}, x.options().dtype(at::ScalarType::Float));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ln_adaln_quantize_stats_vec_delayed_k1024_kernel<<<x.size(0), 256, 0, stream>>>(
        reinterpret_cast<fp8e4m3 *>(out.data_ptr<c10::Float8_e4m3fn>()),
        row_amax.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(shift.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(scale.data_ptr<at::BFloat16>()),
        quant_scale.data_ptr<float>(),
        static_cast<int>(x.size(0)),
        static_cast<int>(tokens_per_sample),
        static_cast<float>(eps)
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {out, row_amax};
}

std::vector<at::Tensor> ln_adaln_quantize_precomputed_vec_k1024(
    const at::Tensor &x,
    const at::Tensor &shift,
    const at::Tensor &scale,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    const at::Tensor &quant_scale,
    int64_t tokens_per_sample
) {
    CHECK_INPUT(x);
    CHECK_INPUT(shift);
    CHECK_INPUT(scale);
    CHECK_INPUT(mean);
    CHECK_INPUT(rstd);
    CHECK_INPUT(quant_scale);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(shift.scalar_type() == at::ScalarType::BFloat16, "shift must be bf16");
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::BFloat16, "scale must be bf16");
    TORCH_CHECK(mean.scalar_type() == at::ScalarType::Float, "mean must be fp32");
    TORCH_CHECK(rstd.scalar_type() == at::ScalarType::Float, "rstd must be fp32");
    TORCH_CHECK(quant_scale.scalar_type() == at::ScalarType::Float, "quant_scale must be fp32");
    TORCH_CHECK(quant_scale.numel() == 1, "quant_scale must contain one scalar");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K1024, "x must have shape [M, 1024]");
    TORCH_CHECK(shift.dim() == 2 && shift.size(1) == K1024, "shift must have shape [B, 1024]");
    TORCH_CHECK(scale.sizes() == shift.sizes(), "scale shape must match shift");
    TORCH_CHECK(mean.dim() == 1 && mean.size(0) == x.size(0), "mean must have shape [M]");
    TORCH_CHECK(rstd.dim() == 1 && rstd.size(0) == x.size(0), "rstd must have shape [M]");
    TORCH_CHECK(tokens_per_sample > 0, "tokens_per_sample must be positive");
    TORCH_CHECK(x.size(0) == shift.size(0) * tokens_per_sample, "M must equal B * tokens_per_sample");

    auto out = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto row_amax = at::empty({x.size(0)}, x.options().dtype(at::ScalarType::Float));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ln_adaln_quantize_precomputed_vec_k1024_kernel<<<x.size(0), 256, 0, stream>>>(
        reinterpret_cast<fp8e4m3 *>(out.data_ptr<c10::Float8_e4m3fn>()),
        row_amax.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(shift.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(scale.data_ptr<at::BFloat16>()),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        quant_scale.data_ptr<float>(),
        static_cast<int>(x.size(0)),
        static_cast<int>(tokens_per_sample)
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {out, row_amax};
}

std::vector<at::Tensor> bias_gelu_quantize_k4096(
    const at::Tensor &x,
    const at::Tensor &bias,
    const at::Tensor &quant_scale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(bias);
    CHECK_INPUT(quant_scale);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(bias.scalar_type() == at::ScalarType::BFloat16, "bias must be bf16");
    TORCH_CHECK(quant_scale.scalar_type() == at::ScalarType::Float, "quant_scale must be fp32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K4096, "x must have shape [M, 4096]");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == K4096, "bias must have shape [4096]");
    TORCH_CHECK(quant_scale.numel() == 1, "quant_scale must contain one scalar");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    auto out = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto row_amax = at::empty({x.size(0)}, x.options().dtype(at::ScalarType::Float));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bias_gelu_quantize_k4096_kernel<<<x.size(0), 256, 0, stream>>>(
        reinterpret_cast<fp8e4m3 *>(out.data_ptr<c10::Float8_e4m3fn>()),
        row_amax.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(bias.data_ptr<at::BFloat16>()),
        quant_scale.data_ptr<float>(),
        static_cast<int>(x.size(0))
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {out, row_amax};
}


std::vector<at::Tensor> bf16_quantize_delayed(
    const at::Tensor &x,
    const at::Tensor &quant_scale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(quant_scale);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(quant_scale.scalar_type() == at::ScalarType::Float, "quant_scale must be fp32");
    TORCH_CHECK(quant_scale.numel() == 1, "quant_scale must contain one scalar");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max() && x.size(1) <= std::numeric_limits<int>::max(),
                "x dimensions must fit int32");

    auto out = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto row_amax = at::empty({x.size(0)}, x.options().dtype(at::ScalarType::Float));
    dim3 block(16, 16);
    dim3 grid((x.size(1) + 15) / 16, (x.size(0) + 15) / 16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA_ERROR(cudaMemsetAsync(row_amax.data_ptr<float>(), 0, row_amax.numel() * sizeof(float), stream));
    bf16_quantize_transpose_delayed_kernel<true, false><<<grid, block, 0, stream>>>(
        reinterpret_cast<fp8e4m3 *>(out.data_ptr<c10::Float8_e4m3fn>()),
        nullptr,
        row_amax.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        quant_scale.data_ptr<float>(),
        static_cast<int>(x.size(0)),
        static_cast<int>(x.size(1))
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {out, row_amax};
}

std::vector<at::Tensor> bf16_quantize_transpose_delayed(
    const at::Tensor &x,
    const at::Tensor &quant_scale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(quant_scale);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(quant_scale.scalar_type() == at::ScalarType::Float, "quant_scale must be fp32");
    TORCH_CHECK(quant_scale.numel() == 1, "quant_scale must contain one scalar");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max() && x.size(1) <= std::numeric_limits<int>::max(),
                "x dimensions must fit int32");

    auto out_t = at::empty({x.size(1), x.size(0)}, x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto row_amax = at::empty({x.size(0)}, x.options().dtype(at::ScalarType::Float));
    dim3 block(16, 16);
    dim3 grid((x.size(1) + 15) / 16, (x.size(0) + 15) / 16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA_ERROR(cudaMemsetAsync(row_amax.data_ptr<float>(), 0, row_amax.numel() * sizeof(float), stream));
    bf16_quantize_transpose_delayed_kernel<false, true><<<grid, block, 0, stream>>>(
        nullptr,
        reinterpret_cast<fp8e4m3 *>(out_t.data_ptr<c10::Float8_e4m3fn>()),
        row_amax.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        quant_scale.data_ptr<float>(),
        static_cast<int>(x.size(0)),
        static_cast<int>(x.size(1))
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {out_t, row_amax};
}

std::vector<at::Tensor> bf16_quantize_rowwise_transpose_delayed(
    const at::Tensor &x,
    const at::Tensor &quant_scale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(quant_scale);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(quant_scale.scalar_type() == at::ScalarType::Float, "quant_scale must be fp32");
    TORCH_CHECK(quant_scale.numel() == 1, "quant_scale must contain one scalar");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max() && x.size(1) <= std::numeric_limits<int>::max(),
                "x dimensions must fit int32");

    auto out = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto out_t = at::empty({x.size(1), x.size(0)}, x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto row_amax = at::empty({x.size(0)}, x.options().dtype(at::ScalarType::Float));
    dim3 block(16, 16);
    dim3 grid((x.size(1) + 15) / 16, (x.size(0) + 15) / 16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA_ERROR(cudaMemsetAsync(row_amax.data_ptr<float>(), 0, row_amax.numel() * sizeof(float), stream));
    bf16_quantize_transpose_delayed_kernel<true, true><<<grid, block, 0, stream>>>(
        reinterpret_cast<fp8e4m3 *>(out.data_ptr<c10::Float8_e4m3fn>()),
        reinterpret_cast<fp8e4m3 *>(out_t.data_ptr<c10::Float8_e4m3fn>()),
        row_amax.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        quant_scale.data_ptr<float>(),
        static_cast<int>(x.size(0)),
        static_cast<int>(x.size(1))
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {out, out_t, row_amax};
}



std::vector<at::Tensor> bf16_quantize_rowwise_transpose_db_delayed(
    const at::Tensor &x,
    const at::Tensor &quant_scale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(quant_scale);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(quant_scale.scalar_type() == at::ScalarType::Float, "quant_scale must be fp32");
    TORCH_CHECK(quant_scale.numel() == 1, "quant_scale must contain one scalar");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max() && x.size(1) <= std::numeric_limits<int>::max(),
                "x dimensions must fit int32");

    auto out = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto out_t = at::empty({x.size(1), x.size(0)}, x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto row_amax = at::empty({x.size(0)}, x.options().dtype(at::ScalarType::Float));
    auto db = at::empty({x.size(1)}, x.options().dtype(at::ScalarType::Float));
    dim3 block(16, 16);
    dim3 grid((x.size(1) + 15) / 16, (x.size(0) + 15) / 16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA_ERROR(cudaMemsetAsync(row_amax.data_ptr<float>(), 0, row_amax.numel() * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(db.data_ptr<float>(), 0, db.numel() * sizeof(float), stream));
    bf16_quantize_transpose_db_delayed_kernel<true, true><<<grid, block, 0, stream>>>(
        reinterpret_cast<fp8e4m3 *>(out.data_ptr<c10::Float8_e4m3fn>()),
        reinterpret_cast<fp8e4m3 *>(out_t.data_ptr<c10::Float8_e4m3fn>()),
        row_amax.data_ptr<float>(),
        db.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        quant_scale.data_ptr<float>(),
        static_cast<int>(x.size(0)),
        static_cast<int>(x.size(1))
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {out, out_t, row_amax, db};
}

std::vector<at::Tensor> gate_bwd_quantize_rowwise_transpose_delayed(
    const at::Tensor &grad_out,
    const at::Tensor &branch_out,
    const at::Tensor &gate,
    const at::Tensor &quant_scale,
    int64_t tokens_per_sample
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(branch_out);
    CHECK_INPUT(gate);
    CHECK_INPUT(quant_scale);
    TORCH_CHECK(grad_out.scalar_type() == at::ScalarType::BFloat16, "grad_out must be bf16");
    TORCH_CHECK(branch_out.scalar_type() == at::ScalarType::BFloat16, "branch_out must be bf16");
    TORCH_CHECK(gate.scalar_type() == at::ScalarType::BFloat16, "gate must be bf16");
    TORCH_CHECK(quant_scale.scalar_type() == at::ScalarType::Float, "quant_scale must be fp32");
    TORCH_CHECK(quant_scale.numel() == 1, "quant_scale must contain one scalar");
    TORCH_CHECK(grad_out.dim() == 2 && branch_out.dim() == 2, "grad_out and branch_out must be 2D");
    TORCH_CHECK(grad_out.sizes() == branch_out.sizes(), "grad_out and branch_out shapes must match");
    TORCH_CHECK(gate.dim() == 2 && gate.size(1) == grad_out.size(1), "gate must have shape [B, H]");
    TORCH_CHECK(tokens_per_sample > 0, "tokens_per_sample must be positive");
    TORCH_CHECK((tokens_per_sample % 16) == 0, "tokens_per_sample must be a multiple of 16");
    TORCH_CHECK(grad_out.size(0) == gate.size(0) * tokens_per_sample, "rows must equal B * tokens_per_sample");
    TORCH_CHECK(grad_out.is_contiguous() && branch_out.is_contiguous() && gate.is_contiguous(),
                "grad_out, branch_out, and gate must be contiguous");
    TORCH_CHECK(grad_out.size(0) <= std::numeric_limits<int>::max() && grad_out.size(1) <= std::numeric_limits<int>::max(),
                "grad_out dimensions must fit int32");

    auto out = at::empty(grad_out.sizes(), grad_out.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto out_t = at::empty({grad_out.size(1), grad_out.size(0)}, grad_out.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto row_amax = at::empty({grad_out.size(0)}, grad_out.options().dtype(at::ScalarType::Float));
    auto dgate = at::empty(gate.sizes(), grad_out.options().dtype(at::ScalarType::Float));

    dim3 block(16, 16);
    dim3 grid((grad_out.size(1) + 15) / 16, (grad_out.size(0) + 15) / 16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA_ERROR(cudaMemsetAsync(row_amax.data_ptr<float>(), 0, row_amax.numel() * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(dgate.data_ptr<float>(), 0, dgate.numel() * sizeof(float), stream));
    gate_bwd_quantize_transpose_delayed_kernel<true, true><<<grid, block, 0, stream>>>(
        reinterpret_cast<fp8e4m3 *>(out.data_ptr<c10::Float8_e4m3fn>()),
        reinterpret_cast<fp8e4m3 *>(out_t.data_ptr<c10::Float8_e4m3fn>()),
        row_amax.data_ptr<float>(),
        dgate.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(grad_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(branch_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(gate.data_ptr<at::BFloat16>()),
        quant_scale.data_ptr<float>(),
        static_cast<int>(grad_out.size(0)),
        static_cast<int>(grad_out.size(1)),
        static_cast<int>(tokens_per_sample)
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {out, out_t, row_amax, dgate};
}

std::vector<at::Tensor> delayed_scaling_update(
    const at::Tensor &row_amax,
    const at::Tensor &scale,
    const at::Tensor &scale_inv,
    const at::Tensor &amax_history,
    const at::Tensor &hist_idx,
    double eps
) {
    CHECK_INPUT(row_amax);
    CHECK_INPUT(scale);
    CHECK_INPUT(scale_inv);
    CHECK_INPUT(amax_history);
    CHECK_INPUT(hist_idx);
    TORCH_CHECK(row_amax.scalar_type() == at::ScalarType::Float, "row_amax must be fp32");
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "scale must be fp32");
    TORCH_CHECK(scale_inv.scalar_type() == at::ScalarType::Float, "scale_inv must be fp32");
    TORCH_CHECK(amax_history.scalar_type() == at::ScalarType::Float, "amax_history must be fp32");
    TORCH_CHECK(hist_idx.scalar_type() == at::ScalarType::Int, "hist_idx must be int32");
    TORCH_CHECK(row_amax.dim() == 1, "row_amax must be 1D");
    TORCH_CHECK(scale.numel() == 1, "scale must contain one scalar");
    TORCH_CHECK(scale_inv.numel() == 1, "scale_inv must contain one scalar");
    TORCH_CHECK(amax_history.dim() == 1 && amax_history.numel() > 0, "amax_history must be non-empty 1D");
    TORCH_CHECK(hist_idx.numel() == 1, "hist_idx must contain one scalar");

    auto global_amax = at::empty({1}, row_amax.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    delayed_scaling_update_kernel<<<1, 256, 0, stream>>>(
        const_cast<float *>(scale.data_ptr<float>()),
        const_cast<float *>(scale_inv.data_ptr<float>()),
        const_cast<float *>(amax_history.data_ptr<float>()),
        const_cast<int *>(hist_idx.data_ptr<int>()),
        global_amax.data_ptr<float>(),
        row_amax.data_ptr<float>(),
        static_cast<int>(row_amax.numel()),
        static_cast<int>(amax_history.numel()),
        static_cast<float>(eps)
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {global_amax, scale, scale_inv, amax_history, hist_idx};
}

at::Tensor reduce_amax(const at::Tensor &partial_amax) {
    CHECK_INPUT(partial_amax);
    TORCH_CHECK(partial_amax.scalar_type() == at::ScalarType::Float, "partial_amax must be fp32");
    TORCH_CHECK(partial_amax.dim() == 1, "partial_amax must be 1D");
    int rows = static_cast<int>(partial_amax.numel());
    int blocks = std::min(1024, (rows + 255) / 256);
    auto block_amax = at::empty({blocks}, partial_amax.options());
    auto global_amax = at::empty({1}, partial_amax.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    reduce_amax_kernel<<<blocks, 256, 0, stream>>>(block_amax.data_ptr<float>(), partial_amax.data_ptr<float>(), rows);
    CHECK_CUDA_ERROR(cudaGetLastError());
    finalize_amax_kernel<<<1, 256, 0, stream>>>(global_amax.data_ptr<float>(), block_amax.data_ptr<float>(), blocks);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return global_amax;
}

at::Tensor fp8_gemm_k1024(const at::Tensor &A, const at::Tensor &B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == K1024 && B.size(1) == K1024, "A and B must have K=1024");
    TORCH_CHECK((A.size(0) % 128) == 0 && (B.size(0) % 256) == 0, "M must be multiple of 128 and N multiple of 256");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options());

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_C = reinterpret_cast<fp8e4m3 *>(C.data_ptr<c10::Float8_e4m3fn>());

    using mmt = matmul_template<8>;
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    inner_run<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}

at::Tensor fp8_gemm_k1024_fp32_out(const at::Tensor &A, const at::Tensor &B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == K1024 && B.size(1) == K1024, "A and B must have K=1024");
    TORCH_CHECK((A.size(0) % 128) == 0 && (B.size(0) % 256) == 0, "M must be multiple of 128 and N multiple of 256");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options().dtype(at::ScalarType::Float));

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    float *d_C = C.data_ptr<float>();

    using mmt = fp8_fp32out::matmul_template<8>;
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_fp32out::inner_run<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}

at::Tensor fp8_gemm_k1024_bf16_out(const at::Tensor &A, const at::Tensor &B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == K1024 && B.size(1) == K1024, "A and B must have K=1024");
    TORCH_CHECK((A.size(0) % 128) == 0 && (B.size(0) % 256) == 0, "M must be multiple of 128 and N multiple of 256");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_C = reinterpret_cast<bf16 *>(C.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out::matmul_template<8>;
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out::inner_run<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}

at::Tensor fp8_gemm_k1024_bf16_out_scaled(const at::Tensor &A, const at::Tensor &B, double a_dequant_scale, double b_dequant_scale) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == K1024 && B.size(1) == K1024, "A and B must have K=1024");
    TORCH_CHECK((A.size(0) % 128) == 0 && (B.size(0) % 256) == 0, "M must be multiple of 128 and N multiple of 256");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_C = reinterpret_cast<bf16 *>(C.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out_scaled::matmul_template<8>;
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out_scaled::inner_run<mmt>(
        d_A, d_B, d_C, M, N, K, static_cast<float>(a_dequant_scale * b_dequant_scale), grid, block
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}

at::Tensor fp8_gemm_k1024_bf16_out_wide_scaled(const at::Tensor &A, const at::Tensor &B, double a_dequant_scale, double b_dequant_scale) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == K1024 && B.size(1) == K1024, "A and B must have K=1024");
    TORCH_CHECK((A.size(0) % 128) == 0 && (B.size(0) % 256) == 0, "M must be multiple of 128 and N multiple of 256");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_C = reinterpret_cast<bf16 *>(C.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out_wide_scaled::matmul_template<8>;
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out_wide_scaled::inner_run<mmt>(
        d_A, d_B, d_C, M, N, K, static_cast<float>(a_dequant_scale * b_dequant_scale), grid, block
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}

at::Tensor fp8_gemm_k1024_bf16_out_deepaccum_scaled(const at::Tensor &A, const at::Tensor &B, double a_dequant_scale, double b_dequant_scale) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == K1024 && B.size(1) == K1024, "A and B must have K=1024");
    TORCH_CHECK((A.size(0) % 128) == 0 && (B.size(0) % 256) == 0, "M must be multiple of 128 and N multiple of 256");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_C = reinterpret_cast<bf16 *>(C.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out_deepaccum::matmul_template<8>;
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out_deepaccum::inner_run<mmt>(
        d_A, d_B, d_C, M, N, K, static_cast<float>(a_dequant_scale * b_dequant_scale), grid, block
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}

at::Tensor fp8_gemm_k1024_bf16_out_deepaccum(const at::Tensor &A, const at::Tensor &B) {
    return fp8_gemm_k1024_bf16_out_deepaccum_scaled(A, B, 1.0, 1.0);
}

at::Tensor fp8_gemm_k1024_bf16_out_deepaccum_n64_scaled(
    const at::Tensor &A,
    const at::Tensor &B,
    double a_dequant_scale,
    double b_dequant_scale
) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == K1024 && B.size(1) == K1024, "A and B must have K=1024");
    TORCH_CHECK((A.size(0) % 128) == 0 && (B.size(0) % 64) == 0, "M must be multiple of 128 and N multiple of 64");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_C = reinterpret_cast<bf16 *>(C.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out_deepaccum_n64::matmul_template<8>;
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out_deepaccum_n64::inner_run<mmt>(
        d_A, d_B, d_C, M, N, K, static_cast<float>(a_dequant_scale * b_dequant_scale), grid, block
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}

at::Tensor fp8_gemm_k1024_bf16_out_deepaccum_n64(const at::Tensor &A, const at::Tensor &B) {
    return fp8_gemm_k1024_bf16_out_deepaccum_n64_scaled(A, B, 1.0, 1.0);
}

at::Tensor fp8_gemm_k1024_bf16_out_pipe(const at::Tensor &A, const at::Tensor &B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == K1024 && B.size(1) == K1024, "A and B must have K=1024");
    TORCH_CHECK((A.size(0) % 128) == 0 && (B.size(0) % 256) == 0, "M must be multiple of 128 and N multiple of 256");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_C = reinterpret_cast<bf16 *>(C.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out_pipe::matmul_template<8>;
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out_pipe::inner_run<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}

at::Tensor fp8_gemm_k1024_bf16_out_pipe64(const at::Tensor &A, const at::Tensor &B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == K1024 && B.size(1) == K1024, "A and B must have K=1024");
    TORCH_CHECK((A.size(0) % 128) == 0 && (B.size(0) % 64) == 0, "M must be multiple of 128 and N multiple of 64");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_C = reinterpret_cast<bf16 *>(C.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out_pipe64::matmul_template<8>;
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out_pipe64::inner_run<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}

template<int SUPER_M, cache_policy A_LOAD_POLICY=cache_policy::NORMAL, cache_policy B_LOAD_POLICY=cache_policy::NORMAL, cache_policy STORE_POLICY=cache_policy::NORMAL>
at::Tensor fp8_gemm_k1024_bf16_out_bias_super(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(bias);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(bias.scalar_type() == at::ScalarType::BFloat16, "bias must be bf16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(A.size(1) == K1024 && B.size(1) == K1024, "A and B must have K=1024");
    TORCH_CHECK(bias.size(0) == B.size(0), "bias must have shape (N,)");
    TORCH_CHECK((A.size(0) % 128) == 0 && (B.size(0) % 128) == 0, "M must be multiple of 128 and N multiple of 128");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_C = reinterpret_cast<bf16 *>(C.data_ptr<at::BFloat16>());
    bf16 *d_bias = reinterpret_cast<bf16 *>(bias.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out_bias::matmul_template<SUPER_M, A_LOAD_POLICY, B_LOAD_POLICY, STORE_POLICY>;
    dim3 grid(mmt::template grid<false>(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out_bias::inner_run<mmt>(d_A, d_B, d_C, d_bias, M, N, K, grid, block);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}

at::Tensor fp8_gemm_k1024_bf16_out_bias_sm4(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    return fp8_gemm_k1024_bf16_out_bias_super<4>(A, B, bias);
}

at::Tensor fp8_gemm_k1024_bf16_out_bias_sm8(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    return fp8_gemm_k1024_bf16_out_bias_super<8>(A, B, bias);
}

at::Tensor fp8_gemm_k1024_bf16_out_bias(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    const auto M = A.size(0);
    const auto N = B.size(0);

    if (N == 4096) {
        return M <= 24576 ? fp8_gemm_k1024_bf16_out_bias_super<4>(A, B, bias)
                          : fp8_gemm_k1024_bf16_out_bias_super<12, cache_policy::NORMAL, cache_policy::NORMAL, cache_policy::EVICT_FIRST>(A, B, bias);
    }
    if (N == 3072) {
        return fp8_gemm_k1024_bf16_out_bias_super<12, cache_policy::NORMAL, cache_policy::NORMAL, cache_policy::EVICT_FIRST>(A, B, bias);
    }
    if (N == 1024) {
        return (M <= 24576 || M >= 262144) ? fp8_gemm_k1024_bf16_out_bias_super<16, cache_policy::NORMAL, cache_policy::NORMAL, cache_policy::EVICT_FIRST>(A, B, bias)
                                           : fp8_gemm_k1024_bf16_out_bias_super<8, cache_policy::NORMAL, cache_policy::NORMAL, cache_policy::EVICT_FIRST>(A, B, bias);
    }
    return fp8_gemm_k1024_bf16_out_bias_super<8>(A, B, bias);
}

at::Tensor fp8_gemm_k1024_bf16_out_bias_sm12(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    return fp8_gemm_k1024_bf16_out_bias_super<12>(A, B, bias);
}

at::Tensor fp8_gemm_k1024_bf16_out_bias_sm16(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    return fp8_gemm_k1024_bf16_out_bias_super<16>(A, B, bias);
}


template<cache_policy A_LOAD_POLICY, cache_policy B_LOAD_POLICY, cache_policy STORE_POLICY>
at::Tensor fp8_gemm_k1024_bf16_out_bias_policy(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    const auto M = A.size(0);
    const auto N = B.size(0);

    if (N == 4096) {
        return M <= 24576 ? fp8_gemm_k1024_bf16_out_bias_super<4, A_LOAD_POLICY, B_LOAD_POLICY, STORE_POLICY>(A, B, bias)
                          : fp8_gemm_k1024_bf16_out_bias_super<12, A_LOAD_POLICY, B_LOAD_POLICY, STORE_POLICY>(A, B, bias);
    }
    if (N == 3072) {
        return fp8_gemm_k1024_bf16_out_bias_super<12, A_LOAD_POLICY, B_LOAD_POLICY, STORE_POLICY>(A, B, bias);
    }
    if (N == 1024) {
        return (M <= 24576 || M >= 262144) ? fp8_gemm_k1024_bf16_out_bias_super<16, A_LOAD_POLICY, B_LOAD_POLICY, STORE_POLICY>(A, B, bias)
                                           : fp8_gemm_k1024_bf16_out_bias_super<8, A_LOAD_POLICY, B_LOAD_POLICY, STORE_POLICY>(A, B, bias);
    }
    return fp8_gemm_k1024_bf16_out_bias_super<8, A_LOAD_POLICY, B_LOAD_POLICY, STORE_POLICY>(A, B, bias);
}

at::Tensor fp8_gemm_k1024_bf16_out_bias_b_evict_last(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    return fp8_gemm_k1024_bf16_out_bias_policy<cache_policy::NORMAL, cache_policy::EVICT_LAST, cache_policy::NORMAL>(A, B, bias);
}

at::Tensor fp8_gemm_k1024_bf16_out_bias_a_first_b_last(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    return fp8_gemm_k1024_bf16_out_bias_policy<cache_policy::EVICT_FIRST, cache_policy::EVICT_LAST, cache_policy::NORMAL>(A, B, bias);
}

at::Tensor fp8_gemm_k1024_bf16_out_bias_store_evict_first(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    return fp8_gemm_k1024_bf16_out_bias_policy<cache_policy::NORMAL, cache_policy::NORMAL, cache_policy::EVICT_FIRST>(A, B, bias);
}

at::Tensor fp8_gemm_k1024_bf16_out_bias_all_cache_hints(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    return fp8_gemm_k1024_bf16_out_bias_policy<cache_policy::EVICT_FIRST, cache_policy::EVICT_LAST, cache_policy::EVICT_FIRST>(A, B, bias);
}



at::Tensor fp8_gemm_k1024_bf16_out_bias_cluster2(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(bias);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(bias.scalar_type() == at::ScalarType::BFloat16, "bias must be bf16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(A.size(1) == K1024 && B.size(1) == K1024, "A and B must have K=1024");
    TORCH_CHECK(bias.size(0) == B.size(0), "bias must have shape (N,)");
    TORCH_CHECK((A.size(0) % 256) == 0 && (B.size(0) % 128) == 0, "cluster2 requires M multiple of 256 and N multiple of 128");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_C = reinterpret_cast<bf16 *>(C.data_ptr<at::BFloat16>());
    bf16 *d_bias = reinterpret_cast<bf16 *>(bias.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out_bias::matmul_cluster2_template<12, cache_policy::NORMAL, cache_policy::NORMAL, cache_policy::EVICT_FIRST>;
    dim3 grid(mmt::template grid<false>(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out_bias::inner_run_cluster<mmt>(d_A, d_B, d_C, d_bias, M, N, K, grid, block);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}

at::Tensor fp8_gemm_k4096_bf16_out_bias(const at::Tensor &A, const at::Tensor &B, const at::Tensor &bias) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(bias);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be fp8 e4m3");
    TORCH_CHECK(bias.scalar_type() == at::ScalarType::BFloat16, "bias must be bf16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(A.size(1) == K4096 && B.size(1) == K4096, "A and B must have K=4096");
    TORCH_CHECK(bias.size(0) == B.size(0), "bias must have shape (N,)");
    TORCH_CHECK((A.size(0) % 128) == 0 && (B.size(0) % 128) == 0, "M must be multiple of 128 and N multiple of 128");

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    at::Tensor C = at::empty({M, N}, A.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3 *>(A.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3 *>(B.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_C = reinterpret_cast<bf16 *>(C.data_ptr<at::BFloat16>());
    bf16 *d_bias = reinterpret_cast<bf16 *>(bias.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out_bias::matmul_template<8>;
    dim3 grid(mmt::template grid<false>(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out_bias::inner_run<mmt>(d_A, d_B, d_C, d_bias, M, N, K, grid, block);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}


at::Tensor fp8_wgrad_2xacc_scaled(const at::Tensor &X_T, const at::Tensor &DY_T, double x_dequant_scale, double dy_dequant_scale) {
    CHECK_INPUT(X_T);
    CHECK_INPUT(DY_T);
    TORCH_CHECK(X_T.scalar_type() == at::ScalarType::Float8_e4m3fn, "X_T must be fp8 e4m3");
    TORCH_CHECK(DY_T.scalar_type() == at::ScalarType::Float8_e4m3fn, "DY_T must be fp8 e4m3");
    TORCH_CHECK(X_T.dim() == 2 && DY_T.dim() == 2, "X_T and DY_T must be 2D");
    TORCH_CHECK(X_T.size(1) == DY_T.size(1), "X_T and DY_T must have matching reduction dimension");
    TORCH_CHECK((X_T.size(0) % 128) == 0 && (DY_T.size(0) % 128) == 0 && (X_T.size(1) % 128) == 0,
                "K, N, and reduction M must be multiples of 128");

    auto K = X_T.size(0);
    auto M = X_T.size(1);
    auto N = DY_T.size(0);
    at::Tensor DW = at::empty({K, N}, X_T.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_X = reinterpret_cast<fp8e4m3 *>(X_T.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_DY = reinterpret_cast<fp8e4m3 *>(DY_T.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_DW = reinterpret_cast<bf16 *>(DW.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out_deepaccum::matmul_template<8>;
    dim3 grid(mmt::grid(K, N, M));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out_deepaccum::inner_run<mmt>(
        d_X, d_DY, d_DW, K, N, M, static_cast<float>(x_dequant_scale * dy_dequant_scale), grid, block
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return DW;
}

at::Tensor fp8_wgrad_2xacc(const at::Tensor &X_T, const at::Tensor &DY_T) {
    return fp8_wgrad_2xacc_scaled(X_T, DY_T, 1.0, 1.0);
}

at::Tensor fp8_dgrad_2xacc_scaled(const at::Tensor &DY, const at::Tensor &W_T, double dy_dequant_scale, double w_dequant_scale) {
    CHECK_INPUT(DY);
    CHECK_INPUT(W_T);
    TORCH_CHECK(DY.scalar_type() == at::ScalarType::Float8_e4m3fn, "DY must be fp8 e4m3");
    TORCH_CHECK(W_T.scalar_type() == at::ScalarType::Float8_e4m3fn, "W_T must be fp8 e4m3");
    TORCH_CHECK(DY.dim() == 2 && W_T.dim() == 2, "DY and W_T must be 2D");
    TORCH_CHECK(DY.size(1) == W_T.size(1), "DY columns must match W_T columns");
    TORCH_CHECK((DY.size(0) % 128) == 0 && (W_T.size(0) % 128) == 0 && (DY.size(1) % 128) == 0,
                "M, K, and reduction N must be multiples of 128");

    auto M = DY.size(0);
    auto N = DY.size(1);
    auto K = W_T.size(0);
    at::Tensor DX = at::empty({M, K}, DY.options().dtype(at::ScalarType::BFloat16));

    fp8e4m3 *d_DY = reinterpret_cast<fp8e4m3 *>(DY.data_ptr<c10::Float8_e4m3fn>());
    fp8e4m3 *d_W = reinterpret_cast<fp8e4m3 *>(W_T.data_ptr<c10::Float8_e4m3fn>());
    bf16 *d_DX = reinterpret_cast<bf16 *>(DX.data_ptr<at::BFloat16>());

    using mmt = fp8_bf16out_deepaccum::matmul_template<8>;
    dim3 grid(mmt::grid(M, K, N));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fp8_bf16out_deepaccum::inner_run<mmt>(
        d_DY, d_W, d_DX, M, K, N, static_cast<float>(dy_dequant_scale * w_dequant_scale), grid, block
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return DX;
}

at::Tensor fp8_dgrad_2xacc(const at::Tensor &DY, const at::Tensor &W_T) {
    return fp8_dgrad_2xacc_scaled(DY, W_T, 1.0, 1.0);
}

PYBIND11_MODULE(_C, m) {
    m.def("ln_adaln_quantize_k1024", &ln_adaln_quantize_k1024);
    m.def("ln_adaln_quantize_stats_k1024", &ln_adaln_quantize_stats_k1024);
    m.def("ln_adaln_quantize_stats_delayed_k1024", &ln_adaln_quantize_stats_delayed_k1024);
    m.def("ln_adaln_quantize_stats_vec_delayed_k1024", &ln_adaln_quantize_stats_vec_delayed_k1024);
    m.def("ln_adaln_quantize_precomputed_vec_k1024", &ln_adaln_quantize_precomputed_vec_k1024);
    m.def("bias_gelu_quantize_k4096", &bias_gelu_quantize_k4096);
    m.def("bf16_quantize_delayed", &bf16_quantize_delayed);
    m.def("bf16_quantize_transpose_delayed", &bf16_quantize_transpose_delayed);
    m.def("bf16_quantize_rowwise_transpose_delayed", &bf16_quantize_rowwise_transpose_delayed);
    m.def("bf16_quantize_rowwise_transpose_db_delayed", &bf16_quantize_rowwise_transpose_db_delayed);
    m.def("gate_bwd_quantize_rowwise_transpose_delayed", &gate_bwd_quantize_rowwise_transpose_delayed);
    m.def("delayed_scaling_update", &delayed_scaling_update);
    m.def("reduce_amax", &reduce_amax);
    m.def("fp8_gemm_k1024", &fp8_gemm_k1024);
    m.def("fp8_gemm_k1024_fp32_out", &fp8_gemm_k1024_fp32_out);
    m.def("fp8_gemm_k1024_bf16_out", &fp8_gemm_k1024_bf16_out);
    m.def("fp8_gemm_k1024_bf16_out_scaled", &fp8_gemm_k1024_bf16_out_scaled);
    m.def("fp8_gemm_k1024_bf16_out_wide_scaled", &fp8_gemm_k1024_bf16_out_wide_scaled);
    m.def("fp8_gemm_k1024_bf16_out_deepaccum", &fp8_gemm_k1024_bf16_out_deepaccum);
    m.def("fp8_gemm_k1024_bf16_out_deepaccum_scaled", &fp8_gemm_k1024_bf16_out_deepaccum_scaled);
    m.def("fp8_gemm_k1024_bf16_out_deepaccum_n64", &fp8_gemm_k1024_bf16_out_deepaccum_n64);
    m.def("fp8_gemm_k1024_bf16_out_deepaccum_n64_scaled", &fp8_gemm_k1024_bf16_out_deepaccum_n64_scaled);
    m.def("fp8_gemm_k1024_bf16_out_pipe", &fp8_gemm_k1024_bf16_out_pipe);
    m.def("fp8_gemm_k1024_bf16_out_pipe64", &fp8_gemm_k1024_bf16_out_pipe64);
    m.def("fp8_gemm_k1024_bf16_out_bias_sm4", &fp8_gemm_k1024_bf16_out_bias_sm4);
    m.def("fp8_gemm_k1024_bf16_out_bias_sm8", &fp8_gemm_k1024_bf16_out_bias_sm8);
    m.def("fp8_gemm_k1024_bf16_out_bias", &fp8_gemm_k1024_bf16_out_bias);
    m.def("fp8_gemm_k1024_bf16_out_bias_sm12", &fp8_gemm_k1024_bf16_out_bias_sm12);
    m.def("fp8_gemm_k1024_bf16_out_bias_sm16", &fp8_gemm_k1024_bf16_out_bias_sm16);
    m.def("fp8_gemm_k1024_bf16_out_bias_b_evict_last", &fp8_gemm_k1024_bf16_out_bias_b_evict_last);
    m.def("fp8_gemm_k1024_bf16_out_bias_a_first_b_last", &fp8_gemm_k1024_bf16_out_bias_a_first_b_last);
    m.def("fp8_gemm_k1024_bf16_out_bias_store_evict_first", &fp8_gemm_k1024_bf16_out_bias_store_evict_first);
    m.def("fp8_gemm_k1024_bf16_out_bias_all_cache_hints", &fp8_gemm_k1024_bf16_out_bias_all_cache_hints);
    m.def("fp8_gemm_k4096_bf16_out_bias", &fp8_gemm_k4096_bf16_out_bias);
    m.def("fp8_wgrad_2xacc", &fp8_wgrad_2xacc);
    m.def("fp8_wgrad_2xacc_scaled", &fp8_wgrad_2xacc_scaled);
    m.def("fp8_dgrad_2xacc", &fp8_dgrad_2xacc);
    m.def("fp8_dgrad_2xacc_scaled", &fp8_dgrad_2xacc_scaled);
}
