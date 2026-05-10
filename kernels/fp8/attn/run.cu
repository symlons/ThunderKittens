#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "kittens.cuh"
#include "prototype.cuh"
#include "../common.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

using o_dtype = float;

template<typename mmt>
void inner_run(
    fp8e4m3 *d_A, fp8e4m3 *d_B, c_dtype *d_C, 
    c_dtype *d_scale_a, c_dtype *d_scale_b,
    size_t M, size_t N, size_t K, 
    dim3 grid, dim3 block
) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    q_layout Ag{d_A, nullptr, nullptr, M, K};
    k_layout Bg{d_B, nullptr, nullptr, N, K};
    v_layout Cg{d_C, nullptr, nullptr, M, N};
    o_layout Cg{d_C, nullptr, nullptr, M, N};

    // scales
    using scale_a_layout = typename mmt::layout::scale_a_layout;
    using scale_b_layout = typename mmt::layout::scale_b_layout;
    scale_a_layout scale_a{d_scale_a, nullptr, nullptr, nullptr, M};
    scale_b_layout scale_b{d_scale_b, nullptr, nullptr, nullptr, N};

    globals G{Ag, Bg, Cg, scale_a, scale_b};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

void write_matrix_to_csv(const std::string& filename, float* matrix, int rows, int cols) {
    std::ofstream file(filename);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << matrix[i * cols + j];
            if (j < cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}


template<typename mmt>
int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";

    // Allocate host memory
    float *h_q = new float[M * K];
    float *h_k = new float[K * N];
    float *h_v = new float[M * N];

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution dis(0.0f, 1.0f);

    // Initialize matrices with random values
    // for (int i = 0; i < M * K; ++i) h_A[i] = i / 100000.0f;  // dis(gen) * 0.2f; 
    // for (int i = 0; i < K * N; ++i) h_B[i] = i / 100000.0f;   // dis(gen) * 0.2f; 
    for (int i = 0; i < M * K; ++i) h_q[i] = dis(gen) * 0.2f; 
    for (int i = 0; i < K * N; ++i) h_k[i] = dis(gen) * 0.2f; 
    for (int i = 0; i < K * N; ++i) h_v[i] = dis(gen) * 0.2f; 

    std::cout << "Initialized matrices" << std::endl;

    // Allocate device memory
    fp8e4m3 *d_q, *d_k, *d_v;
    o_dtype *d_o;
    cudaMalloc(&d_q, M*K*sizeof(fp8e4m3));
    cudaMalloc(&d_k, K*N*sizeof(fp8e4m3));
    cudaMalloc(&d_v, M*N*sizeof(fp8e4m3));
    cudaMalloc(&d_o, M*N*sizeof(o_dtype));
    // scales
    o_dtype *d_scale_q, *d_scale_k, *d_scales_v;
    cudaMalloc(&d_scale_q, M*sizeof(float));
    cudaMalloc(&d_scale_k, N*sizeof(float));
    cudaMalloc(&d_scale_v, N*sizeof(float));
    // float buffers for reference GEMM
    float *d_A_float, *d_B_float, *d_C_ref;
    cudaMalloc(&d_q_float, M*K*sizeof(float));
    cudaMalloc(&d_k_float, K*N*sizeof(float));
    cudaMalloc(&d_v_float, K*N*sizeof(float));
    cudaMalloc(&d_o_ref, M*N*sizeof(float));

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Copy float matrices to device and compute reference GEMM on GPU
    cudaMemcpy(d_q_float, h_q, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_float, h_k, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_float, h_v, K*N*sizeof(float), cudaMemcpyHostToDevice);
    reference_gemm<float, float, true>(d_C_ref, d_A_float, d_B_float, M, N, K);
    cudaDeviceSynchronize();
    std::cout << "Computed reference GEMM on device" << std::endl;

    //  Obtain inputs on GPU device
    const float FP8_E4M3_MAX = 448.0f;
    // const float FP8_E4M3_MIN = -448.0f;
    c_dtype *h_scale_q = new o_dtype[M];
    o_dtype *h_scale_k = new o_dtype[N];
    o_dtype *h_scale_v = new o_dtype[N];
    __nv_fp8_e4m3 *h_A_fp8_scaled = new __nv_fp8_e4m3[M * K];
    __nv_fp8_e4m3 *h_B_fp8_scaled = new __nv_fp8_e4m3[K * N];
    
    // row-wise scaling
    for(int row = 0; row < M; row++) {
        float max_val = 0.0f;
        for(int col = 0; col < K; col++) {
            float abs_val = std::abs(h_A[row * K + col]);
            max_val = std::max(max_val, abs_val);
        }
        h_scale_a[row] = c_dtype(max_val / FP8_E4M3_MAX); 
        if ( row < 10 ) {
            std::cout << "h_scale_a[" << row << "] = " << float(h_scale_a[row]) << ", max_val: " << max_val << std::endl;
        }
    }

    // fill h_A_fp8_scaled by following to_float8_e4m3fn. 
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            h_A_fp8_scaled[i * K + j] = __nv_fp8_e4m3(h_A[i * K + j] / float(h_scale_a[i]));
        }
    }

    // column-wise scaling
    for(int col = 0; col < N; col++) {
        float max_val = 0.0f;
        for(int row = 0; row < K; row++) {
            float abs_val = std::abs(h_B[row + col*K]);
            max_val = std::max(max_val, abs_val);
        }
        h_scale_b[col] = c_dtype(max_val / FP8_E4M3_MAX);

        if ( col < 10 ) {
            std::cout << "h_scale_b[" << col << "] = " << float(h_scale_b[col]) << ", max_val: " << max_val << std::endl;
        }
    }

    // fill h_B_fp8_scaled by following to_float8_e4m3fn
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < K; j++) {
            h_B_fp8_scaled[j + i * K] = __nv_fp8_e4m3(h_B[j + i * K] / float(h_scale_b[i]));
        }
    }
    
    cudaMemcpy(d_A, h_A_fp8_scaled, M*K*sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_fp8_scaled, K*N*sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_a, h_scale_a, M*sizeof(c_dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_b, h_scale_b, N*sizeof(c_dtype), cudaMemcpyHostToDevice);

    /* 
    Launch kernel
    */
    std::cout << "Copied matrices to device" << std::endl;
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch kernel
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < ( 2 ); i++) { // warmup
        inner_run<mmt>(d_A, d_B, d_C, d_scale_a, d_scale_b, M, N, K, grid, block); 
    }

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = ( 10 );
    for(int i = 0; i < ITERS; i++) {
        inner_run<mmt>(d_A, d_B, d_C, d_scale_a, d_scale_b, M, N, K, grid, block); 
    }
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;

    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy result back to host
    c_dtype *h_C_out = new c_dtype[M * N];
    float *h_C_ref = new float[M * N];
    cudaMemcpy(h_C_out, d_C, M*N*sizeof(c_dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) {
        h_C[i] = float(h_C_out[i]);
    }

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f, total_error = 0.0f, total_ref = 0.0f, total_ours=0.0f;
    float input_a = 0.0f, input_b = 0.0f;
    int error_count = 0;
    printf("Num rows: %zu, Num cols: %zu\n", M, N);
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if( error > 0.7f ) { // large because of fp8 vs fp32 numerics # error > 0.10
            if(error_count < 10) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 700) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
        total_ref += std::abs(h_C_ref[i]);
        total_error += error;
        total_ours += std::abs(h_C[i]);
    }

    for (int i = 0; i < M * K; i++) {
        input_a += std::abs(h_A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        input_b += std::abs(h_B[i]);
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Average error: " << total_error / M / N << std::endl;
    std::cout << "Average ref: " << total_ref / (M * N) << std::endl;
    std::cout << "Average ours: " << total_ours / M / N << std::endl;
    std::cout << "Average input_a: " << input_a / M / K << std::endl;
    std::cout << "Average input_b: " << input_b / K / N << std::endl;
    std::cout << "Error count: " << error_count << std::endl;
 
    // write_matrix_to_csv("h_C_ref.csv", h_C_ref, M, N);
    // write_matrix_to_csv("h_C.csv", h_C, M, N);

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_C_out;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_float);
    cudaFree(d_B_float);
    cudaFree(d_C_ref);

    return 0;
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    run_benchmark<matmul_template<8>>(M, N, K);
    return 0;
}

