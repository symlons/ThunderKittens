#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define DIM     "\033[2m"
#define CYAN    "\033[36m"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess)                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " -> " << cudaGetErrorString(err) << std::endl;        \
    } while (0)

int main() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp props{};
        CHECK_CUDA(cudaGetDeviceProperties(&props, dev));

        int cc_major, cc_minor;
        CHECK_CUDA(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev));
        CHECK_CUDA(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev));

        std::cout << "\n" << BOLD << "=== Device " << dev << " ===" << RESET << "\n";

        std::cout << "\n" << BOLD << "[RAW DEVICE]" << RESET << "\n";
        std::cout << DIM;
        std::cout << "Name                  : " << props.name << "\n";
        std::cout << "Compute Capability    : " << cc_major << "." << cc_minor << "\n";
        std::cout << "Multiprocessors       : " << props.multiProcessorCount << "\n";
        std::cout << "Global Memory         : " << props.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "L2 Cache              : " << props.l2CacheSize / (1024 * 1024) << " MB\n";
        std::cout << "Warp Size             : " << props.warpSize << "\n";
        std::cout << "Max Threads / Block   : " << props.maxThreadsPerBlock << "\n";
        std::cout << "Max Threads / SM      : " << props.maxThreadsPerMultiProcessor << "\n";
        std::cout << "Shared Mem / SM       : " << props.sharedMemPerMultiprocessor / 1024 << " KB\n";
        std::cout << "Shared Mem / Block    : " << props.sharedMemPerBlock / 1024 << " KB\n";
        std::cout << RESET;

        int sm_clock, mem_clock, bus_width;
        CHECK_CUDA(cudaDeviceGetAttribute(&sm_clock, cudaDevAttrClockRate, dev));
        CHECK_CUDA(cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, dev));
        CHECK_CUDA(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, dev));

        std::cout << "\n" << BOLD << "[RAW CLOCKS]" << RESET << "\n";
        std::cout << DIM;
        std::cout << "SM Clock              : " << sm_clock * 1e-6 << " GHz\n";
        std::cout << "Memory Clock          : " << mem_clock * 1e-6 << " GHz\n";
        std::cout << "Memory Bus Width      : " << bus_width << " bits\n";
        std::cout << RESET;

        int regs_sm, regs_block;
        CHECK_CUDA(cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, dev));
        CHECK_CUDA(cudaDeviceGetAttribute(&regs_block, cudaDevAttrMaxRegistersPerBlock, dev));

        std::cout << "\n" << BOLD << "[RAW LIMITS]" << RESET << "\n";
        std::cout << DIM;
        std::cout << "Registers / SM        : " << regs_sm << "\n";
        std::cout << "Registers / Block     : " << regs_block << "\n";
        std::cout << RESET;

        int val;
        CHECK_CUDA(cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        std::cout << DIM << "Max SMEM / Block Opt  : " << val / 1024 << " KB\n" << RESET;

        int async_engines;
        CHECK_CUDA(cudaDeviceGetAttribute(&async_engines, cudaDevAttrAsyncEngineCount, dev));

        std::cout << DIM << "Async Engines         : " << async_engines;
        if (async_engines == 0)
            std::cout << " (no overlap)";
        else if (async_engines == 1)
            std::cout << " (1 copy engine)";
        else if (async_engines == 2)
            std::cout << " (H2D + D2H overlap)";
        else
            std::cout << " (full duplex + compute overlap)";
        std::cout << "\n" << RESET;

        CHECK_CUDA(cudaDeviceGetAttribute(&val, cudaDevAttrConcurrentKernels, dev));
        std::cout << DIM << "Concurrent Kernels    : " << val << "\n" << RESET;

        std::cout << "\n" << CYAN << "[DERIVED]" << RESET << "\n";

        double smClockGHz = sm_clock * 1e-6;

        double bandwidth =
            2.0 * mem_clock * 1e3 * (bus_width / 8.0) / 1e9;

        std::cout << CYAN << "Memory Bandwidth      : " << bandwidth << " GB/s\n";

        double regs_per_thread = (double)regs_sm / props.maxThreadsPerMultiProcessor;

        std::cout << "Registers / Thread SM : " << regs_per_thread << "\n";

        double fp32_flops_per_sm_per_cycle;
        double tensor_flops_per_sm_per_cycle;

        if (cc_major == 9) {
            fp32_flops_per_sm_per_cycle = 128.0 * 2.0;
            tensor_flops_per_sm_per_cycle = 4096.0;
        } else if (cc_major == 8) {
            fp32_flops_per_sm_per_cycle = 64.0 * 2.0;
            tensor_flops_per_sm_per_cycle = 2048.0;
        } else {
            fp32_flops_per_sm_per_cycle = 64.0 * 2.0;
            tensor_flops_per_sm_per_cycle = 1024.0;
        }

        double fp32_tflops = props.multiProcessorCount * fp32_flops_per_sm_per_cycle * smClockGHz / 1000.0;
        double tensor_tflops = props.multiProcessorCount * tensor_flops_per_sm_per_cycle * smClockGHz / 1000.0;
        std::cout << "FP32 TFLOPS (peak)    : " << fp32_tflops << "\n";
        std::cout << "Tensor TFLOPS (peak)  : " << tensor_tflops << "\n" << RESET;

        CHECK_CUDA(cudaDeviceGetAttribute(&val, cudaDevAttrUnifiedAddressing, dev));
        std::cout << DIM << "\nUnified Addressing    : " << val << RESET << "\n";
    }

    return 0;
}
