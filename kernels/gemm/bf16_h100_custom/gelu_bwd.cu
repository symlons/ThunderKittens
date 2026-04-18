#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// GELU backward: grad_input = grad_output * gelu'(preact)
// where gelu(x) = 0.5 * x * (1 + tanh(a)), a = 0.79788456 * x * (1 + 0.044715 * x^2)
// gelu'(x) = 0.5 * (1 + tanh(a)) + 0.5 * x * sech^2(a) * 0.79788456 * (1 + 3 * 0.044715 * x^2)

__device__ __forceinline__ float fast_tanh(float x) {
    float y;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Vectorized kernel: process 8 bf16 values per thread (128 bits = 4 x bf16_2)
__global__ void gelu_backward_kernel(
    __nv_bfloat16 * __restrict__ grad_input,
    const __nv_bfloat16 * __restrict__ grad_output,
    const __nv_bfloat16 * __restrict__ preact,
    size_t n
) {
    // Each thread processes 8 elements
    size_t idx = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 >= n) {
        // Scalar fallback for tail
        for (size_t i = idx; i < n; i++) {
            float x = __bfloat162float(preact[i]);
            float dy = __bfloat162float(grad_output[i]);
            float x2 = x * x;
            float a = 0.79788456f * x * (1.0f + 0.044715f * x2);
            float t = fast_tanh(a);
            float sech2 = 1.0f - t * t;
            float dx = 0.5f * (1.0f + t) + 0.5f * x * sech2 * 0.79788456f * (1.0f + 3.0f * 0.044715f * x2);
            grad_input[i] = __float2bfloat16(dy * dx);
        }
        return;
    }

    // Load 8 bf16 as 4 x bf16_2 (128 bits) per input
    __nv_bfloat162 go[4], pa[4];
    *reinterpret_cast<int4*>(go) = *reinterpret_cast<const int4*>(grad_output + idx);
    *reinterpret_cast<int4*>(pa) = *reinterpret_cast<const int4*>(preact + idx);

    __nv_bfloat162 result[4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 g = __bfloat1622float2(go[i]);
        float2 p = __bfloat1622float2(pa[i]);

        // First element
        float x = p.x;
        float dy = g.x;
        float x2 = x * x;
        float a = 0.79788456f * x * (1.0f + 0.044715f * x2);
        float t = fast_tanh(a);
        float sech2 = 1.0f - t * t;
        float dx = 0.5f * (1.0f + t) + 0.5f * x * sech2 * 0.79788456f * (1.0f + 3.0f * 0.044715f * x2);
        float r0 = dy * dx;

        // Second element
        x = p.y;
        dy = g.y;
        x2 = x * x;
        a = 0.79788456f * x * (1.0f + 0.044715f * x2);
        t = fast_tanh(a);
        sech2 = 1.0f - t * t;
        dx = 0.5f * (1.0f + t) + 0.5f * x * sech2 * 0.79788456f * (1.0f + 3.0f * 0.044715f * x2);
        float r1 = dy * dx;

        result[i] = __floats2bfloat162_rn(r0, r1);
    }

    *reinterpret_cast<int4*>(grad_input + idx) = *reinterpret_cast<int4*>(result);
}

void launch_gelu_backward(
    __nv_bfloat16 *grad_input,
    const __nv_bfloat16 *grad_output,
    const __nv_bfloat16 *preact,
    size_t n,
    cudaStream_t stream = 0
) {
    const int threads = 256;
    const int elems_per_thread = 8;
    int blocks = (n + threads * elems_per_thread - 1) / (threads * elems_per_thread);
    gelu_backward_kernel<<<blocks, threads, 0, stream>>>(grad_input, grad_output, preact, n);
}

// ============================================================
// Reference CPU-style kernel for correctness checking
// ============================================================
__global__ void gelu_backward_reference_kernel(
    __nv_bfloat16 *grad_input,
    const __nv_bfloat16 *grad_output,
    const __nv_bfloat16 *preact,
    size_t n
) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = __bfloat162float(preact[idx]);
    float dy = __bfloat162float(grad_output[idx]);
    float x2 = x * x;
    float a = 0.79788456f * x * (1.0f + 0.044715f * x2);
    float t = tanhf(a);
    float sech2 = 1.0f - t * t;
    float dx = 0.5f * (1.0f + t) + 0.5f * x * sech2 * 0.79788456f * (1.0f + 3.0f * 0.044715f * x2);
    grad_input[idx] = __float2bfloat16(dy * dx);
}

// ============================================================
// Standalone benchmark / correctness test
// ============================================================
#ifndef TORCH_COMPILE

#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

void fill_random_bf16(__nv_bfloat16 *d_ptr, size_t count, uint64_t seed, float lo, float hi) {
    std::vector<__nv_bfloat16> h(count);
    uint64_t x = seed;
    for (size_t i = 0; i < count; i++) {
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        float u = float(x >> 40) * (1.0f / 16777216.0f);
        h[i] = __float2bfloat16(u * (hi - lo) + lo);
    }
    cudaMemcpy(d_ptr, h.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
}

int main() {
    size_t sizes[][2] = {{4096, 4096}, {8192, 8192}, {16384, 16384}};
    bool all_correct = true;

    for (auto &sz : sizes) {
        size_t M = sz[0], N = sz[1];
        size_t count = M * N;

        __nv_bfloat16 *d_grad_out, *d_preact, *d_grad_in, *d_grad_in_ref;
        cudaMalloc(&d_grad_out, count * sizeof(__nv_bfloat16));
        cudaMalloc(&d_preact,   count * sizeof(__nv_bfloat16));
        cudaMalloc(&d_grad_in,  count * sizeof(__nv_bfloat16));
        cudaMalloc(&d_grad_in_ref, count * sizeof(__nv_bfloat16));

        fill_random_bf16(d_grad_out, count, 42, -1.0f, 1.0f);
        fill_random_bf16(d_preact,   count, 99, -2.0f, 2.0f);

        // Reference
        {
            int threads = 256;
            int blocks = (count + threads - 1) / threads;
            gelu_backward_reference_kernel<<<blocks, threads>>>(d_grad_in_ref, d_grad_out, d_preact, count);
        }

        // Optimized
        launch_gelu_backward(d_grad_in, d_grad_out, d_preact, count);
        cudaDeviceSynchronize();

        // Correctness
        std::vector<__nv_bfloat16> h_out(count), h_ref(count);
        cudaMemcpy(h_out.data(), d_grad_in,     count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ref.data(), d_grad_in_ref, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

        double err_max = 0, err_sum = 0, abs_sum = 0;
        for (size_t i = 0; i < count; i++) {
            float v = __bfloat162float(h_out[i]);
            float r = __bfloat162float(h_ref[i]);
            double e = fabs(v - r);
            err_max = std::max(err_max, e);
            err_sum += e;
            abs_sum += fabs(r);
        }
        bool correct = err_max < 0.02;
        all_correct &= correct;

        // Benchmark
        int warmup = 500, iters = 1000;
        for (int i = 0; i < warmup; i++)
            launch_gelu_backward(d_grad_in, d_grad_out, d_preact, count);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++)
            launch_gelu_backward(d_grad_in, d_grad_out, d_preact, count);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        double us = ms * 1000.0 / iters;
        double bytes = 3.0 * count * 2.0;
        double bw_tb = (bytes / (us * 1e-6)) / 1e12;

        std::cout << "===== M=" << M << " N=" << N << " =====" << std::endl;
        std::cout << "  correctness: " << (correct ? "PASS" : "FAIL")
                  << "  (err_max=" << err_max << " err_mean=" << err_sum/count << ")" << std::endl;
        std::cout << "  time: " << us << " us  BW: " << bw_tb << " TB/s  eff: "
                  << (bw_tb / 3.35) * 100.0 << " %" << std::endl;

        cudaFree(d_grad_out);
        cudaFree(d_preact);
        cudaFree(d_grad_in);
        cudaFree(d_grad_in_ref);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return all_correct ? 0 : 1;
}

#else
// ============================================================
// PyTorch binding
// ============================================================
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void gelu_backward_entrypoint(
    const at::Tensor &grad_output,
    const at::Tensor &preact,
    const at::Tensor &grad_input
) {
    TORCH_CHECK(grad_output.is_cuda() && preact.is_cuda() && grad_input.is_cuda());
    TORCH_CHECK(grad_output.is_contiguous() && preact.is_contiguous() && grad_input.is_contiguous());
    TORCH_CHECK(grad_output.dtype() == torch::kBFloat16);

    size_t n = grad_output.numel();
    launch_gelu_backward(
        reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(preact.data_ptr()),
        n,
        at::cuda::getCurrentCUDAStream()
    );
}

PYBIND11_MODULE(_gelu_bwd, m) {
    m.def("gelu_backward", &gelu_backward_entrypoint);
}
#endif
