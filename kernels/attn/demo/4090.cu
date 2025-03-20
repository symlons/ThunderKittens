#include "kittens.cuh"

constexpr int ATTN_B = 16; // batch_size
constexpr int ATTN_H = 16; // attention heads
constexpr int ATTN_N = 1024; // sequence length 
constexpr int ATTN_D = 64; // Q, K, V dimension / gets passed as template parameter to the kernel (4090.impl)
constexpr int ITER   = 10; // number of iterations (used in 4090_harness.impl)

using namespace kittens;

constexpr int NUM_WORKERS = 4; // This kernel uses 4 worker warps (number of warps) per block.
constexpr int PIPE_STAGES = 3; // number of stages in the pipeline

template<int D> constexpr size_t ROWS = 16*(128/D); // height of each worker tile (rows)
template <int D, typename T = bf16, typename L = row_l>
using qkvo_tile = rt<T, ROWS<D>, D, L>; // Defines the tile layout for Query, Key, Value, and Output (QKVO) matrices in the registers.
template<int D, typename T=float> using attn_tile = rt<T, ROWS<D>, ROWS<D>>;
template<int D> using shared_tile = st_bf<ROWS<D>, D>; // defines shared memory layout
template<int D> using global_layout = gl<bf16, -1, -1, -1, D>; // B, N, H, specified at runtime, D known at compile time for this kernel.
template<int D> struct globals { global_layout<D> Qg, Kg, Vg, Og; };

template <int D> // here is the D defined as template parameter (defined in 4090.impl)
__launch_bounds__(NUM_WORKERS *WARP_THREADS, 1) // WARP_THREADS is 32 and defined by CUDA, NUM_WORKERS * WARP_THREADS is the number of threads in a block
                                                // 1 represents the minimum number of blocks per multiprocessor
    __global__ void attend_ker(const __grid_constant__ globals<D> g)
    // Declares a CUDA kernel function with a reference to the struct 'globals' in global memory,
    // which holds the global memory for the attention tiles (Q, K, V, O) used in the attention mechanism.
    // __grid_constant__ indicates that the global memory reference 'g' is the same across all blocks in the grid.
    // This allows the compiler to optimize memory access, improving performance.
{

    using load_group = kittens::group<2>; // pairs of workers collaboratively load k, v tiles
    // A "warpgroup" is a special group of 4 consecutive warps defined by NVIDIA for certain SM_90+ operations.
    int loadid = load_group::groupid(), workerid = kittens::warpid(); // which worker am I?
    // gets the warpid of the current thread and the groupid of the current thread. Where groupid represents the **warp group** id.
    // warpid: Gets the ID of the current warp (a group of 32 threads) within a block.
    // groupid: Gets the ID of the current warp group (a group of N warps) within a block.
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS; // determines how many cooperative loading groups are formed.
    // This value is then used in the kernel to control how the loading of key and value data is distributed among the warps.const int batch = blockIdx.z, head = blockIdx.y, q_seq = blockIdx.x * NUM_WORKERS + workerid;
    // load_group::GROUP_WARPS: alias to number of warps in a group for abstraction, can be changed ...

    extern __shared__ alignment_dummy __shm[];  // extern __shared__ is used to allocate shared memory dynamically.
    // Declare an external, dynamically sized shared memory array named '__shm'.
    // 'extern __shared__' indicates that the array resides in shared memory and its size will be determined at kernel launch,
    // based on the 'sharedMemPerBlock' parameter in the kernel's launch configuration.
    // 'alignment_dummy' ensures proper memory alignment for optimal performance.
    shared_allocator al((int*)&__shm[0]);
    // Initialize a 'shared_allocator' object named 'al'.
    // kittens handles the alignement and compile time verification etc.
    // The allocator manages the allocation of memory within the shared memory region pointed to by '__shm'.
    // '(int*)&__shm[0]' provides the allocator with the base address of the shared memory region as an integer pointer,
    // for integer sized memory allocations. This is done to abstract away the complexity of manual shared memory management.

    shared_tile<D> (&k_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    // This is a type representing a tile of data stored in shared memory, where D is a template parameter representing the dimension of the tile. (D represents the height of each worker tile, it is defined at the top.)
    // al manages the allocation of shared memory as defined above.
    // k_smem represents the shared memory tile for the key data, with dimensions [LOAD_BLOCKS][PIPE_STAGES].
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    // same as the above just for the values matrix.
    
    shared_tile<D> (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem); // !!
    // Reinterprets the shared memory allocated for 'k_smem' as a 1D array 'qo_smem'.
    // 'qo_smem' becomes a reference to the first 'NUM_WORKERS' elements of 'k_smem', treated as individual shared_tile<D> objects.
    // This allows each worker to access a dedicated portion of the shared memory for query and output data.
    // this helps with: The code is trying to minimize the amount of shared memory used by reusing the memory allocated for k_smem for qo_smem
    // Essentially, the code is using the same memory for the query/output and the key/value data, but at different times.
    // (reinterpret_cast reinterprets the bit pattern of one type as another, without changing the underlying bits)

    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg, k_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile<D, bf16, col_l> v_reg; // V is column layout, as we use mma_AB.
    qkvo_tile<D, float> o_reg; // Output tile.
    attn_tile<D, float> att_block; // attention tile, in float. (We want to use float wherever possible.)
    attn_tile<D, bf16> att_block_mma; // bf16 attention tile for the second mma_AB. We cast right before that op.
    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec; // these are column vectors for the in-place softmax.
    // each warp loads its own Q tile of 16x64
    if (q_seq*ROWS<D> < g.Qg.depth()) { // bounds check
        load<1, false>(qo_smem[workerid], g.Qg, {batch, q_seq, head, 0});  // going through shared memory improves coalescing of dram reads.
        // TODO: find the definition of load<1, false> and what it does.
        __syncwarp()
        load(q_reg, qo_smem[workerid]);
    }
    __syncthreads(); // finished loading of Q into shared memory

    if constexpr(D == 64) q_reg *= __float2bfloat16(0.125f * 1.44269504089f);
    else if constexpr(D == 128) q_reg *= __float2bfloat16(0.08838834764f * 1.44269504089f);

    max_vec = base_types::constants<float>::neg_infty();
    norm_vec = 0.f;
    o_reg = 0.f;
    // launch the load of the first k, v tiles
    int kv_blocks = (g.Kg.depth() + LOAD_BLOCKS*ROWS<D>-1) / (LOAD_BLOCKS*ROWS<D>), tic = 0;
    load_group::load_async<1, false>(k_smem[loadid][0], g.Kg, {batch, loadid, head, 0});
    load_group::load_async<1, false>(v_smem[loadid][0], g.Vg, {batch, loadid, head, 0});
    // loads K and V tiles into shared memory for the first time.


    // iterate over k, v for these q's that have been loaded
    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=(tic+1)%3) {
        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
        if(next_load_idx*ROWS<D> < g.Kg.depth()) {
            int next_tic = (tic+1)%3;
            load_group::load_async<1, false>(k_smem[loadid][next_tic], g.Kg, {batch, next_load_idx, head, 0});
            load_group::load_async<1, false>(v_smem[loadid][next_tic], g.Vg, {batch, next_load_idx, head, 0});
            load_async_wait<1>(); // next k, v can stay in flight.
        }

        // This loop implements a pipelined loading strategy.
        // It calculates the indices of the next key/value blocks to be loaded,
        // initiates asynchronous loads for those blocks, and waits for the previous loads to complete.
        // This allows the kernel to overlap memory transfers with computation, improving performance. // TODO: i don't think that this is computation but rather it overlaps with laoding the next tiles.
        // The tic variable and the PIPE_STAGES shared memory buffers are used to manage the pipelined loading process.
        // the price of the first synchronize load before this load here is paid only once in the beginning

        else load_async_wait();
        __syncthreads();
        // The key is that while some threads are waiting, the memory controllers are still working, and future loads are being started.

        #pragma unroll LOAD_BLOCKS
        for(int subtile = 0; subtile < LOAD_BLOCKS && (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D> < g.Kg.depth(); subtile++) {
            // The loop iterates over each loading group (subtile). This loop is used to process each subtile of the current KV block.
            load(k_reg, k_smem[subtile][tic]); // load k from shared into registers
            att_block = 0.f; // zero 16x16 attention tile
            mma<transpose::N, transpose::T>(att_block, q_reg, k_reg, att_block); // Q@K.T

            // uses masking to handle padding, but it does not implement causal attention.
            int first_index = (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D>; // one past the last KV index of this tile
            int start_fill = g.Kg.depth()-first_index < ROWS<D> ? g.Kg.depth()-first_index : ROWS<D>;
            right_fill(att_block, att_block, start_fill, base_types::constants<float>::neg_infty());

            max_vec_last = max_vec;
            max_vec = max<axis::COL>(att_block, max_vec); 
            att_block = exp2(att_block - max_vec); 
            max_vec_last = exp2(max_vec_last - max_vec); 
            norm_vec *= max_vec_last; 
            norm_vec = sum<axis::COL>(att_block, norm_vec); 
            att_block_mma = att_block; // copy to bf16 tile
            load(v_reg, v_smem[subtile][tic]); 
            o_reg *= max_vec_last; 
            mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg);
        }
    }

    o_reg /= norm_vec;
    __syncthreads();                    // Synchronizes all threads within a CUDA thread block
    if (q_seq*ROWS<D> < g.Og.depth()) { // write out o.
        store(qo_smem[workerid], o_reg); // going through shared memory improves coalescing of dram writes.
        __syncwarp();                    // Synchronizes only threads within a single warp.
        store<1, false>(g.Og, qo_smem[workerid], {batch, q_seq, head, 0});
    }
}

#include "4090_harness.impl"