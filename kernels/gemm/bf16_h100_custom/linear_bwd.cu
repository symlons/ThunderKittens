#include "kittens.cuh"
#include "prototype.cuh"
#include <cstdint>
#include <math.h>

// ============================================================
// Kernel 1: Fused GELU backward + bias gradient (elementwise)
//   dz = dy * GELU'(preact)
//   db = sum_rows(dz)
// ============================================================

__device__ __forceinline__ float fast_tanh(float x) {
    float y;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Pass 1: Vectorized GELU' elementwise (coalesced linear access, ~26µs)
__global__ void gelu_bwd_kernel(
    __nv_bfloat16 * __restrict__ dz,
    const __nv_bfloat16 * __restrict__ dy,
    const __nv_bfloat16 * __restrict__ preact,
    size_t n
) {
    size_t idx = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 >= n) {
        for (size_t i = idx; i < n; i++) {
            float x = __bfloat162float(preact[i]), g = __bfloat162float(dy[i]);
            float x2 = x*x, a = 0.79788456f*x*(1.f+0.044715f*x2), t;
            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(a));
            float s2 = 1.f-t*t;
            dz[i] = __float2bfloat16(g*(0.5f*(1.f+t)+0.5f*x*s2*0.79788456f*(1.f+3.f*0.044715f*x2)));
        }
        return;
    }
    __nv_bfloat162 go[4], pa[4], res[4];
    *reinterpret_cast<int4*>(go) = *reinterpret_cast<const int4*>(dy + idx);
    *reinterpret_cast<int4*>(pa) = *reinterpret_cast<const int4*>(preact + idx);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 g = __bfloat1622float2(go[i]), p = __bfloat1622float2(pa[i]);
        float x, d, x2, a, t, s2, gp;
        x=p.x; d=g.x; x2=x*x; a=0.79788456f*x*(1.f+0.044715f*x2);
        asm volatile("tanh.approx.f32 %0, %1;":"=f"(t):"f"(a));
        s2=1.f-t*t; gp=0.5f*(1.f+t)+0.5f*x*s2*0.79788456f*(1.f+3.f*0.044715f*x2);
        float r0=d*gp;
        x=p.y; d=g.y; x2=x*x; a=0.79788456f*x*(1.f+0.044715f*x2);
        asm volatile("tanh.approx.f32 %0, %1;":"=f"(t):"f"(a));
        s2=1.f-t*t; gp=0.5f*(1.f+t)+0.5f*x*s2*0.79788456f*(1.f+3.f*0.044715f*x2);
        res[i] = __floats2bfloat162_rn(r0, d*gp);
    }
    *reinterpret_cast<int4*>(dz + idx) = *reinterpret_cast<int4*>(res);
}

// Pass 2: Bias column reduction.
// Grid: blockIdx.x = column block, blockIdx.y = row block
// Each thread handles 1 column across ROWS_PER_BLOCK rows (coalesced: adjacent threads = adjacent cols)
// Partial sums atomicAdd'd to global — only (M/ROWS_PER_BLOCK) atomics per column
// 1 column per thread, coalesced 2-byte reads within warps.
// High rows-per-block to minimize atomics. Grid.y = M/RPB = 8 atomics per column.
constexpr int BIAS_THREADS = 256;
constexpr int BIAS_ROWS_PER_BLOCK = 512;

__global__ void bias_reduce_kernel(
    float * __restrict__ dbias,
    const __nv_bfloat16 * __restrict__ dz,
    int M, int N
) {
    int col = blockIdx.x * BIAS_THREADS + threadIdx.x;
    if (col >= N) return;
    int row_start = blockIdx.y * BIAS_ROWS_PER_BLOCK;
    int row_end = min(row_start + BIAS_ROWS_PER_BLOCK, M);

    float acc = 0.f;
    for (int r = row_start; r < row_end; r++)
        acc += __bfloat162float(dz[r * N + col]);

    atomicAdd(&dbias[col], acc);
}

void launch_gelu_bwd_bias(
    __nv_bfloat16 *dz, float *dbias,
    const __nv_bfloat16 *dy, const __nv_bfloat16 *preact,
    int M, int N, cudaStream_t stream = 0
) {
    // Pass 1: GELU' elementwise
    size_t count = (size_t)M * N;
    int blocks1 = (count + 256*8 - 1) / (256*8);
    gelu_bwd_kernel<<<blocks1, 256, 0, stream>>>(dz, dy, preact, count);

    // Pass 2: bias reduction
    cudaMemsetAsync(dbias, 0, N * sizeof(float), stream);
    dim3 grid2((N + BIAS_THREADS - 1) / BIAS_THREADS, (M + BIAS_ROWS_PER_BLOCK - 1) / BIAS_ROWS_PER_BLOCK);
    bias_reduce_kernel<<<grid2, BIAS_THREADS, 0, stream>>>(dbias, dz, M, N);
}

// ============================================================
// Kernel 2: dW = x^T @ dz  (mma_AtB, no transpose needed)
//   x[M,K], dz[M,N] -> dW[K,N]
//   Reduction dim: M. Uses wide-tile mma_AtB.
// ============================================================

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

template<int M_BLOCK, int N_BLOCK>
struct dw_gemm_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout x, dz, dW; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct scratch_block  {};
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12, int _PIPE_STAGES=4>
struct dw_gemm_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = dw_gemm_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=_PIPE_STAGES, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERSISTENT_GRID=true> __host__ static inline dim3 grid(int K, int N, int M) {
        return dim3(PERSISTENT_GRID ? 132 : K*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if (threadIdx.x == 0) {
            args.globals.x.template prefetch_tma<typename layout::base_tile>();
            args.globals.dz.template prefetch_tma<typename layout::base_tile>();
            args.globals.dW.template prefetch_tma<typename layout::base_tile>();
        }
        int K = args.globals.dW.rows(), N = args.globals.dW.cols();
        int Rblocks = K / (M_BLOCK*64), Cblocks = N / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.x.rows()/64;  // M/64 (reduction dimension)
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
                // x[M,K]: load at {iter(M-dim), coord.x(K-dim)}
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.x,
                                    {args.iter, args.common.coord.x+i}, args.inputs_arrived);
                // dz[M,N]: load at {iter(M-dim), coord.y(N-dim)}
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.dz,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            args.state.accum = 0.f;
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AtB(args.state.accum, args.input.a[warpgroup::groupid()],
                               reinterpret_cast<wide_tile&>(args.input.b));
            warpgroup::mma_async_wait();
            if (warp::elect_leader()) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.dW, args.finish.c[warpgroup::groupid()][i],
                                     {args.common.coord.x, args.common.coord.y + i});
                tma::store_async_read_wait();
            }
            args.state.accum = 0.f;
            if (warp::elect_leader()) arrive(args.finish_finished);
        }
    };
};

// ============================================================
// Kernel 3: dx = dz @ W^T  (mma_ABt per-tile, no transpose needed)
//   dz[M,N], W[K,N] -> dx[M,K]
//   Reduction dim: N.
// ============================================================

template<int M_BLOCK, int N_BLOCK>
struct dx_gemm_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout dz, W, dx; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct scratch_block  {};
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12, int _PIPE_STAGES=4>
struct dx_gemm_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = dx_gemm_layout<M_BLOCK, N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=_PIPE_STAGES, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERSISTENT_GRID=true> __host__ static inline dim3 grid(int M, int K, int N) {
        return dim3(PERSISTENT_GRID ? 132 : M*K/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if (threadIdx.x == 0) {
            args.globals.dz.template prefetch_tma<typename layout::base_tile>();
            args.globals.W.template prefetch_tma<typename layout::base_tile>();
            args.globals.dx.template prefetch_tma<typename layout::base_tile>();
        }
        int M = args.globals.dx.rows(), K = args.globals.dx.cols();
        int Rblocks = M / (M_BLOCK*64), Cblocks = K / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.dz.cols()/64;  // N/64 (reduction dimension)
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
                    tma::load_async(args.input.a[i], args.globals.dz,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.W,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            args.state.accum = 0.f;
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            // Single mma_ABt with tall W tile (N_BLOCK*64 × 64) — single fence+commit
            using tall_tile = st_bf<64*N_BLOCK, 64>;
            warpgroup::mma_ABt(args.state.accum, args.input.a[warpgroup::groupid()],
                               reinterpret_cast<tall_tile&>(args.input.b));
            warpgroup::mma_async_wait();
            if (warp::elect_leader()) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            using wide_tile = st_bf<64, 64*N_BLOCK>;
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::elect_leader()) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.dx, args.finish.c[warpgroup::groupid()][i],
                                     {args.common.coord.x, args.common.coord.y + i});
                tma::store_async_read_wait();
            }
            args.state.accum = 0.f;
            if (warp::elect_leader()) arrive(args.finish_finished);
        }
    };
};

// ============================================================
// Standalone benchmark / correctness harness
// ============================================================
#ifndef TORCH_COMPILE
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include "../common.cuh"

// --- Reference: D = A^T @ B where A[M,K] row-major, B[M,N] row-major, D[K,N] row-major ---
// D[k,n] = sum_m A[m,k] * B[m,n]
template <typename T>
__global__ void reference_AtB_kernel(T* D, const T* A, const T* B, int M, int K, int N) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K && n < N) {
        float acc = 0.0f;
        for (int m = 0; m < M; m++) {
            float a = kittens::base_types::convertor<float, T>::convert(A[m * K + k]);
            float b = kittens::base_types::convertor<float, T>::convert(B[m * N + n]);
            acc += a * b;
        }
        D[k * N + n] = kittens::base_types::convertor<T, float>::convert(acc);
    }
}

template <typename T>
static inline void reference_AtB(T* D, const T* A, const T* B, int M, int K, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (K + 15) / 16);
    reference_AtB_kernel<T><<<grid, block>>>(D, A, B, M, K, N);
}

// --- Reference: GELU backward + bias grad ---
__global__ void ref_gelu_bwd_bias_kernel(
    __nv_bfloat16 *dz, float *dbias,
    const __nv_bfloat16 *dy, const __nv_bfloat16 *preact,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    float x = __bfloat162float(preact[idx]);
    float g = __bfloat162float(dy[idx]);
    float x2 = x * x;
    float a = 0.79788456f * x * (1.0f + 0.044715f * x2);
    float t = tanhf(a);
    float sech2 = 1.0f - t * t;
    float dx = 0.5f * (1.0f + t) + 0.5f * x * sech2 * 0.79788456f * (1.0f + 3.0f * 0.044715f * x2);
    float val = g * dx;
    dz[idx] = __float2bfloat16(val);
    int col = idx % N;
    atomicAdd(&dbias[col], val);
}

// Simple GPU matrix transpose: out[j,i] = in[i,j], in is rows×cols
template<typename T>
__global__ void transpose_kernel(T* out, const T* in, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols, c = idx % cols;
    out[c * rows + r] = in[r * cols + c];
}
template<typename T>
void transpose_matrix(T* out, const T* in, int rows, int cols) {
    int n = rows * cols;
    transpose_kernel<T><<<(n+255)/256, 256>>>(out, in, rows, cols);
}

template<typename T>
void check(const char* name, T const* d_out, T const* d_ref, size_t count) {
    std::vector<T> h_out(count), h_ref(count);
    cudaMemcpy(h_out.data(), d_out, count * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), d_ref, count * sizeof(T), cudaMemcpyDeviceToHost);
    double err_max = 0, err_sum = 0;
    for (size_t i = 0; i < count; i++) {
        float v = kittens::base_types::convertor<float, T>::convert(h_out[i]);
        float r = kittens::base_types::convertor<float, T>::convert(h_ref[i]);
        double e = fabs(v - r);
        err_max = std::max(err_max, e);
        err_sum += e;
    }
    std::cout << name << ": err_max=" << err_max << " err_mean=" << err_sum/count
              << " " << (err_max < 5.0 ? "PASS" : "FAIL") << std::endl;
}

void check_float(const char* name, float const* d_out, float const* d_ref, size_t count) {
    std::vector<float> h_out(count), h_ref(count);
    cudaMemcpy(h_out.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), d_ref, count * sizeof(float), cudaMemcpyDeviceToHost);
    double err_max = 0, err_sum = 0;
    for (size_t i = 0; i < count; i++) {
        double e = fabs(h_out[i] - h_ref[i]);
        err_max = std::max(err_max, e);
        err_sum += e;
    }
    std::cout << name << ": err_max=" << err_max << " err_mean=" << err_sum/count
              << " " << (err_max < 2.0 ? "PASS" : "FAIL") << std::endl;
}

template<typename mmt>
void launch_dw_gemm(bf16 *x, bf16 *dz, bf16 *dW, int M, int K, int N) {
    using global_layout = typename mmt::layout::global_layout;
    using globals = typename mmt::layout::globals;
    // x[M,K], dz[M,N], dW[K,N]
    global_layout Xg{x,  nullptr, nullptr, M, K};
    global_layout DZg{dz, nullptr, nullptr, M, N};
    global_layout DWg{dW, nullptr, nullptr, K, N};
    globals G{Xg, DZg, DWg};
    dim3 grid = mmt::grid(K, N, M);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY-1024);
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

template<typename mmt>
void launch_dx_gemm(bf16 *dz, bf16 *W, bf16 *dx, int M, int N, int K) {
    using global_layout = typename mmt::layout::global_layout;
    using globals = typename mmt::layout::globals;
    // dz[M,N], W[K,N], dx[M,K]
    global_layout DZg{dz, nullptr, nullptr, M, N};
    global_layout Wg{W,  nullptr, nullptr, K, N};
    global_layout DXg{dx, nullptr, nullptr, M, K};
    globals G{DZg, Wg, DXg};
    dim3 grid = mmt::grid(M, K, N);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY-1024);
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

// ============================================================
// TK-convention benchmark helper
// L2-defeating groups, 500 warmup, 100 iters, thermal cooldown
// ============================================================

template<typename mmt, bool IS_DW>
double bench_gemm_tk(const char* label, int M, int K, int N) {
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);

    // Input bytes: A + B + C, all bf16
    size_t input_bytes;
    if constexpr (IS_DW) // x[M,K] + dz[M,N] + dW[K,N]
        input_bytes = (size_t(M)*K + size_t(M)*N + size_t(K)*N) * sizeof(bf16);
    else // dz[M,N] + W[K,N] + dx[M,K]
        input_bytes = (size_t(M)*N + size_t(K)*N + size_t(M)*K) * sizeof(bf16);

    int num_groups = (input_bytes >= size_t(l2_cache_size) * 3) ? 1
                   : int(size_t(l2_cache_size) * 3 / input_bytes) + 1;

    struct Bufs { bf16 *a, *b, *c; };
    std::vector<Bufs> groups(num_groups);
    for (int i = 0; i < num_groups; i++) {
        if constexpr (IS_DW) {
            // dW = Xt @ dz: a=Xt[K,M], b=dz[M,N], c=dW[K,N]
            bf16 *tmp_x;
            cudaMalloc(&groups[i].a, K*M*sizeof(bf16));
            cudaMalloc(&groups[i].b, M*N*sizeof(bf16));
            cudaMalloc(&groups[i].c, K*N*sizeof(bf16));
            cudaMalloc(&tmp_x, M*K*sizeof(bf16));
            fill<bf16, FillMode::RANDOM>(tmp_x, M*K, 42+i*100, -1.f, 1.f);
            transpose_matrix<bf16>(groups[i].a, tmp_x, M, K); // Xt[K,M] = x[M,K]^T
            cudaFree(tmp_x);
            fill<bf16, FillMode::RANDOM>(groups[i].b, M*N, 43+i*100, -1.f, 1.f);
        } else {
            // dx = dz @ Wt: a=dz[M,N], b=Wt[N,K], c=dx[M,K]
            bf16 *tmp_W;
            cudaMalloc(&groups[i].a, M*N*sizeof(bf16));
            cudaMalloc(&groups[i].b, N*K*sizeof(bf16));
            cudaMalloc(&groups[i].c, M*K*sizeof(bf16));
            cudaMalloc(&tmp_W, K*N*sizeof(bf16));
            fill<bf16, FillMode::RANDOM>(groups[i].a, M*N, 42+i*100, -1.f, 1.f);
            fill<bf16, FillMode::RANDOM>(tmp_W, K*N, 43+i*100, -1.f, 1.f);
            transpose_matrix<bf16>(groups[i].b, tmp_W, K, N); // Wt[N,K] = W[K,N]^T
            cudaFree(tmp_W);
        }
    }
    cudaDeviceSynchronize();

    // Setup kernel attributes once
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY-1024);

    auto launch = [&](Bufs &g) {
        if constexpr (IS_DW)
            launch_dw_gemm<mmt>(g.a, g.b, g.c, M, K, N);
        else
            launch_dx_gemm<mmt>(g.a, g.b, g.c, M, N, K);
    };

    // 500 warmup, no sync after
    for (int i = 0; i < 500; i++)
        launch(groups[i % num_groups]);

    // 2 events around 100 iters
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        launch(groups[i % num_groups]);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double us = ms * 1000.0 / 100;
    double flops = 2.0 * M * N * K;
    double tflops = (flops / us) / 1e6;

    std::cout << label << ": " << us << " us  (" << tflops << " TFLOPS)  [groups=" << num_groups << "]\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (auto &g : groups) { cudaFree(g.a); cudaFree(g.b); cudaFree(g.c); }

    sleep_ms(500); // thermal cooldown
    return us;
}

int main() {
    int M = 4096, K = 4096, N = 4096;
    std::cout << "===== Linear Backward Tuning: M=" << M << " K=" << K << " N=" << N << " =====\n";
    std::cout << "TK convention: 500 warmup, 100 iters, L2-defeating groups, 500ms cooldown\n\n";

    // ---- Quick correctness check with default config ----
    {
        bf16 *d_x, *d_Xt, *d_dz, *d_W, *d_Wt, *d_dW, *d_dx, *d_dW_ref, *d_dx_ref;
        cudaMalloc(&d_x,  M*K*sizeof(bf16)); cudaMalloc(&d_Xt, K*M*sizeof(bf16));
        cudaMalloc(&d_dz, M*N*sizeof(bf16));
        cudaMalloc(&d_W,  K*N*sizeof(bf16)); cudaMalloc(&d_Wt, N*K*sizeof(bf16));
        cudaMalloc(&d_dW, K*N*sizeof(bf16));
        cudaMalloc(&d_dx, M*K*sizeof(bf16));
        cudaMalloc(&d_dW_ref, K*N*sizeof(bf16)); cudaMalloc(&d_dx_ref, M*K*sizeof(bf16));
        fill<bf16, FillMode::RANDOM>(d_x,  M*K, 42, -1.f, 1.f);
        fill<bf16, FillMode::RANDOM>(d_dz, M*N, 43, -1.f, 1.f);
        fill<bf16, FillMode::RANDOM>(d_W,  K*N, 44, -1.f, 1.f);
        transpose_matrix<bf16>(d_Xt, d_x, M, K);  // Xt[K,M] = x[M,K]^T
        transpose_matrix<bf16>(d_Wt, d_W, K, N);  // Wt[N,K] = W[K,N]^T

        // dW_ref = x^T @ dz = Xt @ dz (standard AB)
        reference_gemm<bf16, bf16, false>(d_dW_ref, d_Xt, d_dz, K, N, M);
        // dx_ref = dz @ Wt (standard AB)
        reference_gemm<bf16, bf16, false>(d_dx_ref, d_dz, d_Wt, M, K, N);
        cudaDeviceSynchronize();

        launch_dw_gemm<dw_gemm_template<2,4,8>>(d_Xt, d_dz, d_dW, M, K, N);
        launch_dx_gemm<dx_gemm_template<2,4,8>>(d_dz, d_Wt, d_dx, M, N, K);
        cudaDeviceSynchronize();
        check("dW<2,4,8>", d_dW, d_dW_ref, (size_t)K*N);
        check("dx<2,4,8>", d_dx, d_dx_ref, (size_t)M*K);

        cudaFree(d_x); cudaFree(d_Xt); cudaFree(d_dz); cudaFree(d_W); cudaFree(d_Wt);
        cudaFree(d_dW); cudaFree(d_dx); cudaFree(d_dW_ref); cudaFree(d_dx_ref);
    }

    // ---- Sweep dW = x^T @ dz (AtB) configs ----
    // Format: <M_BLOCK, N_BLOCK, SUPER_M, PIPE_STAGES>
    // input_block = (M_BLOCK+N_BLOCK)*64*64*2 bytes, padded to 1024
    // <2,4>: 48KB/stage → max 4 stages in 228KB
    // <2,2>: 32KB/stage → max 7 stages
    // <2,3>: 40KB/stage → max 5 stages
    std::cout << "\n===== dW = x^T @ dz (AtB) sweep =====\n";
    std::cout << "--- <2,4> SUPER_M sweep ---\n";
    bench_gemm_tk<dw_gemm_template<2,4, 4,4>, true>("dW<2,4, 4,4>", M, K, N);
    bench_gemm_tk<dw_gemm_template<2,4, 8,4>, true>("dW<2,4, 8,4>", M, K, N);
    bench_gemm_tk<dw_gemm_template<2,4,16,4>, true>("dW<2,4,16,4>", M, K, N);
    bench_gemm_tk<dw_gemm_template<2,4,32,4>, true>("dW<2,4,32,4>", M, K, N);
    std::cout << "--- <2,4> pipe sweep (best SUPER_M) ---\n";
    bench_gemm_tk<dw_gemm_template<2,4,16,3>, true>("dW<2,4,16,3>", M, K, N);
    std::cout << "--- <2,2> configs (more pipe stages) ---\n";
    bench_gemm_tk<dw_gemm_template<2,2, 8,4>, true>("dW<2,2, 8,4>", M, K, N);
    bench_gemm_tk<dw_gemm_template<2,2, 8,5>, true>("dW<2,2, 8,5>", M, K, N);
    bench_gemm_tk<dw_gemm_template<2,2, 8,6>, true>("dW<2,2, 8,6>", M, K, N);
    bench_gemm_tk<dw_gemm_template<2,2,16,4>, true>("dW<2,2,16,4>", M, K, N);
    bench_gemm_tk<dw_gemm_template<2,2,16,5>, true>("dW<2,2,16,5>", M, K, N);
    bench_gemm_tk<dw_gemm_template<2,2,16,6>, true>("dW<2,2,16,6>", M, K, N);

    // ---- Sweep dx = dz @ W^T (ABt) configs ----
    std::cout << "\n===== dx = dz @ W^T (ABt) sweep =====\n";
    std::cout << "--- <2,4> SUPER_M sweep ---\n";
    bench_gemm_tk<dx_gemm_template<2,4, 4,4>, false>("dx<2,4, 4,4>", M, K, N);
    bench_gemm_tk<dx_gemm_template<2,4, 8,4>, false>("dx<2,4, 8,4>", M, K, N);
    bench_gemm_tk<dx_gemm_template<2,4,16,4>, false>("dx<2,4,16,4>", M, K, N);
    bench_gemm_tk<dx_gemm_template<2,4,32,4>, false>("dx<2,4,32,4>", M, K, N);
    std::cout << "--- <2,4> pipe sweep ---\n";
    bench_gemm_tk<dx_gemm_template<2,4,16,3>, false>("dx<2,4,16,3>", M, K, N);
    std::cout << "--- <2,2> configs (more pipe stages) ---\n";
    bench_gemm_tk<dx_gemm_template<2,2, 8,4>, false>("dx<2,2, 8,4>", M, K, N);
    bench_gemm_tk<dx_gemm_template<2,2, 8,5>, false>("dx<2,2, 8,5>", M, K, N);
    bench_gemm_tk<dx_gemm_template<2,2, 8,6>, false>("dx<2,2, 8,6>", M, K, N);
    bench_gemm_tk<dx_gemm_template<2,2,16,4>, false>("dx<2,2,16,4>", M, K, N);
    bench_gemm_tk<dx_gemm_template<2,2,16,5>, false>("dx<2,2,16,5>", M, K, N);
    bench_gemm_tk<dx_gemm_template<2,2,16,6>, false>("dx<2,2,16,6>", M, K, N);

    return 0;
}

#else
// ============================================================
// PyTorch binding
// ============================================================
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "pyutils/torchutils.cuh"

void gelu_bwd_bias_entrypoint(
    const at::Tensor &dy,
    const at::Tensor &preact,
    const at::Tensor &dz,
    const at::Tensor &dbias
) {
    TORCH_CHECK(dy.is_cuda() && preact.is_cuda() && dz.is_cuda() && dbias.is_cuda());
    int M = dy.size(0), N = dy.size(1);
    launch_gelu_bwd_bias(
        reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()),
        reinterpret_cast<float*>(dbias.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(preact.data_ptr()),
        M, N, at::cuda::getCurrentCUDAStream()
    );
}

void dw_gemm_entrypoint(
    const at::Tensor &x,
    const at::Tensor &dz,
    const at::Tensor &dW
) {
    using mmt = dw_gemm_template<2, 4, 16>;
    using global_layout = typename mmt::layout::global_layout;
    using globals = typename mmt::layout::globals;

    int M = x.size(0), K = x.size(1), N = dz.size(1);

    global_layout Xg  = kittens::py::tensor_to_gl<global_layout>(x);
    global_layout DZg = kittens::py::tensor_to_gl<global_layout>(dz);
    global_layout DWg = kittens::py::tensor_to_gl<global_layout>(dW);
    globals G{Xg, DZg, DWg};

    dim3 grid = mmt::grid(K, N, M);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    prototype::lcf::kernel<mmt><<<grid, block, smem, stream>>>(G);
}

void dx_gemm_entrypoint(
    const at::Tensor &dz,
    const at::Tensor &W,
    const at::Tensor &dx
) {
    using mmt = dx_gemm_template<2, 4, 16>;
    using global_layout = typename mmt::layout::global_layout;
    using globals = typename mmt::layout::globals;

    int M = dz.size(0), N = dz.size(1), K = W.size(0);

    global_layout DZg = kittens::py::tensor_to_gl<global_layout>(dz);
    global_layout Wg  = kittens::py::tensor_to_gl<global_layout>(W);
    global_layout DXg = kittens::py::tensor_to_gl<global_layout>(dx);
    globals G{DZg, Wg, DXg};

    dim3 grid = mmt::grid(M, K, N);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    prototype::lcf::kernel<mmt><<<grid, block, smem, stream>>>(G);
}

PYBIND11_MODULE(_linear_bwd, m) {
    m.def("gelu_bwd_bias", &gelu_bwd_bias_entrypoint);
    m.def("dw_gemm", &dw_gemm_entrypoint);
    m.def("dx_gemm", &dx_gemm_entrypoint);
}
#endif
