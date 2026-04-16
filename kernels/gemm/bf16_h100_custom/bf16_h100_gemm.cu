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

__device__ static inline void apply_gelu(rt_fl<16, 256> &acc) {
    // rt_fl<16, 256>: height=1 (16/16), width=16 (256/16), 4 float2 per tile
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
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
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
            if (warpgroup::laneid() == 0) {
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
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            // Store pre-activation (before GELU) for backward pass
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::laneid() == 0) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.preact, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y + i});
            }
            // Apply GELU in registers while TMA reads preact from smem
            apply_gelu(args.state.accum);
            if (warpgroup::laneid() == 0)
                tma::store_async_read_wait();
            // Store post-activation
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::laneid() == 0) {
                for (int i = 0; i < N_BLOCK; i++)
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y + i});
            }
            // Overlap: reinit accum while TMA reads C from smem
            init_bias(args.state.accum, args.scratch.bias);
            if (warpgroup::laneid() == 0)
                tma::store_async_read_wait();
            if (warp::laneid() == 0) arrive(args.finish_finished);
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
    // int Cblocks = 22, Rblocks = 24;
    // int Cblocks192 = 20, Rblocks192 = 16;
    // run_benchmark<matmul_template<4>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
    // run_benchmark<matmul_template<8>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
    // run_benchmark<matmul_template<12>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
    int N;
    N = 4096;
    run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // N = 3072;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 4096;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // N = 6144;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 8192;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // N = 12288;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 16384;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<2,4,12>>(N, N, N);
    // run_benchmark<matmul_template<3,3,12>>(192*12, 192*11, 8192);
    // run_benchmark<matmul_template<2,4,11>>(128*22, 256* 6, 8192);
    // run_benchmark<matmul_template<2,4,1>>(128 * 132, 256, 256);
    // run_benchmark<matmul_template<2,4,1>>(128 * 133, 256, 256);
    // run_benchmark<matmul_template<2,4,1>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,8>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,12>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,128>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<3,3,12>>(192*22, 192*6*2, 8192);
    // run_benchmark<matmul_template<3,3,12>>(192*22, 192*6*2, 16384);
    // run_benchmark<matmul_template<2,4,11>>(128*22*2, 256* 6*2, 8192);
    // run_benchmark<matmul_template<3,3,12>>(192*12*2, 192*11*2, 8192*2);
    // run_benchmark<matmul_template<2,4,11>>(128*22*2, 256* 6*2, 8192*2);
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

    cudaFuncSetAttribute(
        prototype::lcf::kernel<mmt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem
    );

    prototype::lcf::kernel<mmt>
        <<<grid, block, smem, stream>>>(G);
}

PYBIND11_MODULE(_C, m) {
    m.def("gemm_custom", &gemm_custom_entrypoint);
}
#endif
