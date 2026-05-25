#include "kittens.cuh"
#include "prototype.cuh"
#include <cstdint>
#include <math.h>

// ============================================================
// Fused Linear Backward:
//   1. gelu_bwd_kernel: dz = dy * gelu'(preact)  [standalone, ~26µs]
//   2. dw_gemm: dW = x^T @ dz                    [unchanged AtB GEMM]
//   3. dx_gemm_bias: dx = dz @ W^T + fused dbias  [ABt GEMM with bias]
//
// The bias gradient is fused into the dx GEMM to eliminate
// a separate reduction kernel and one extra read of dz.
// ============================================================

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// ============================================================
// Standalone GELU backward kernel (vectorized, ~26µs for 4096x4096)
// ============================================================
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

void launch_gelu_bwd(
    __nv_bfloat16 *dz,
    const __nv_bfloat16 *dy, const __nv_bfloat16 *preact,
    int M, int N, cudaStream_t stream = 0
) {
    size_t count = (size_t)M * N;
    int blocks = (count + 256*8 - 1) / (256*8);
    gelu_bwd_kernel<<<blocks, 256, 0, stream>>>(dz, dy, preact, count);
}

constexpr int GELU_BWD_BIAS_COLS = 16;
constexpr int GELU_BWD_BIAS_ROW_THREADS = 16;
constexpr int GELU_BWD_BIAS_ROWS_PER_BLOCK = 256;

__device__ static inline float gelu_tanh_grad(float x) {
    float x2 = x * x;
    float a = 0.79788456f * x * (1.f + 0.044715f * x2);
    float t;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(a));
    float s2 = 1.f - t * t;
    return 0.5f * (1.f + t) + 0.5f * x * s2 * 0.79788456f * (1.f + 3.f * 0.044715f * x2);
}

__global__ void gelu_bwd_bias_fused_kernel(
    __nv_bfloat16 *__restrict__ dz,
    float *__restrict__ dbias,
    const __nv_bfloat16 *__restrict__ dy,
    const __nv_bfloat16 *__restrict__ preact,
    int M,
    int N
) {
    __shared__ float sums[GELU_BWD_BIAS_ROW_THREADS][GELU_BWD_BIAS_COLS];
    int col = blockIdx.x * GELU_BWD_BIAS_COLS + threadIdx.x;
    int row_start = blockIdx.y * GELU_BWD_BIAS_ROWS_PER_BLOCK;
    int row_end = min(row_start + GELU_BWD_BIAS_ROWS_PER_BLOCK, M);
    int ty = threadIdx.y;

    float acc = 0.f;
    if (col < N) {
        for (int row = row_start + ty; row < row_end; row += GELU_BWD_BIAS_ROW_THREADS) {
            size_t idx = size_t(row) * N + col;
            float x = __bfloat162float(preact[idx]);
            float g = __bfloat162float(dy[idx]);
            float out = g * gelu_tanh_grad(x);
            dz[idx] = __float2bfloat16(out);
            acc += out;
        }
    }
    sums[ty][threadIdx.x] = acc;
    __syncthreads();

    if (ty == 0 && col < N) {
        float total = 0.f;
        #pragma unroll
        for (int i = 0; i < GELU_BWD_BIAS_ROW_THREADS; i++) {
            total += sums[i][threadIdx.x];
        }
        atomicAdd(&dbias[col], total);
    }
}

// ============================================================
// dW = x^T @ dz  (mma_AtB) — unchanged from linear_bwd.cu
// ============================================================

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
        args.num_iters = args.globals.x.rows()/64;
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
                    tma::load_async(args.input.a[i], args.globals.x,
                                    {args.iter, args.common.coord.x+i}, args.inputs_arrived);
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
// dx = dz @ W^T  (mma_ABt) with FUSED bias gradient
//   dz[M,N], W[K,N] -> dx[M,K]
//   Also: dbias[N] = sum_rows(dz)
//
// The reduction dimension is N. The grid tiles over M×K.
// For bias: only the K-block owner (coord.y==0) accumulates.
// Each M-row tile visits every N-column exactly once during
// the reduction, so each warpgroup sums its 64-row chunk of dz
// column-by-column and atomicAdds to dbias.
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
        args.num_iters = args.globals.dz.cols()/64;  // N/64
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
// dx = dz @ W for native PyTorch weight layout
//   dz[M,N], W[N,K] -> dx[M,K], where W is [out_features, in_features]
// ============================================================

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12, int _PIPE_STAGES=4>
struct dx_native_gemm_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = dx_gemm_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
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
        args.num_iters = args.globals.dz.cols()/64;  // N/64
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
            warpgroup::mma_AB(args.state.accum, args.input.a[warpgroup::groupid()],
                              reinterpret_cast<wide_tile&>(args.input.b));
            warpgroup::mma_async_wait();
            if (warp::elect_leader()) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
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
// Standalone bias reduction kernel
// ============================================================
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

// ============================================================
// PyTorch binding
// ============================================================
#ifdef TORCH_COMPILE
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaMemsetAsync(dbias.data_ptr(), 0, N * sizeof(float), stream);
    dim3 block(GELU_BWD_BIAS_COLS, GELU_BWD_BIAS_ROW_THREADS);
    dim3 grid((N + GELU_BWD_BIAS_COLS - 1) / GELU_BWD_BIAS_COLS,
              (M + GELU_BWD_BIAS_ROWS_PER_BLOCK - 1) / GELU_BWD_BIAS_ROWS_PER_BLOCK);
    gelu_bwd_bias_fused_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()),
        reinterpret_cast<float*>(dbias.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(preact.data_ptr()),
        M,
        N
    );
}

void bias_reduce_entrypoint(
    const at::Tensor &dz,
    const at::Tensor &dbias
) {
    TORCH_CHECK(dz.is_cuda() && dbias.is_cuda());
    TORCH_CHECK(dz.scalar_type() == at::kBFloat16);
    TORCH_CHECK(dbias.scalar_type() == at::kFloat);
    TORCH_CHECK(dz.dim() == 2 && dbias.dim() == 1);
    int M = dz.size(0), N = dz.size(1);
    TORCH_CHECK(dbias.size(0) == N);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(dbias.data_ptr(), 0, N * sizeof(float), stream);
    dim3 grid((N + BIAS_THREADS - 1) / BIAS_THREADS, (M + BIAS_ROWS_PER_BLOCK - 1) / BIAS_ROWS_PER_BLOCK);
    bias_reduce_kernel<<<grid, BIAS_THREADS, 0, stream>>>(
        reinterpret_cast<float*>(dbias.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dz.data_ptr()),
        M, N
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

void dx_gemm_native_entrypoint(
    const at::Tensor &dz,
    const at::Tensor &W,
    const at::Tensor &dx
) {
    using mmt = dx_native_gemm_template<2, 4, 16>;
    using global_layout = typename mmt::layout::global_layout;
    using globals = typename mmt::layout::globals;

    int M = dz.size(0), N = dz.size(1), K = W.size(1);

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

PYBIND11_MODULE(_linear_bwd_fused, m) {
    m.def("gelu_bwd_bias", &gelu_bwd_bias_entrypoint);
    m.def("bias_reduce", &bias_reduce_entrypoint);
    m.def("dw_gemm", &dw_gemm_entrypoint);
    m.def("dx_gemm", &dx_gemm_entrypoint);
    m.def("dx_gemm_native", &dx_gemm_native_entrypoint);
}
#endif
