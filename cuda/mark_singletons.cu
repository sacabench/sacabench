#include <string>
#include <iostream>
#include "cub-1.8.0/cub/cub.cuh"

template <uint32_t BLOCK_THREADS, uint32_t ITEMS_PER_THREAD>
__global__ static void mark_heads(uint32_t* d_in, bool* d_flags) {

    // Specialize BlockDiscontinuity for a 1D block of 128 threads on type int
    typedef cub::BlockDiscontinuity<uint32_t, ITEMS_PER_THREAD> BlockDiscontinuity;
    // Allocate shared memory for BlockDiscontinuity
    __shared__ typename BlockDiscontinuity::TempStorage temp_storage;

    // Specialize BlockLoad type for our thread block
    typedef cub::BlockLoad<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    // Specialize BlockStore type for our thread block
    typedef cub::BlockStore<bool, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;

    // Shared memory
    __shared__ typename BlockLoadT::TempStorage    load;
    __shared__ typename BlockStoreT::TempStorage   store;

    // Obtain a segment of consecutive items that are blocked across threads
    uint32_t thread_data[ITEMS_PER_THREAD];
    BlockLoadT(load).Load(d_in, thread_data);

    // block of flags per thread
    bool flags[4];
    // Collectively compute head flags for discontinuities in the segment
    BlockDiscontinuity(temp_storage).FlagHeads(flags, thread_data, cub::Inequality());

    // Store flags from a blocked arrangement
    BlockStoreT(store).Store(d_flags, flags);
}

int main()
{
    const uint32_t g_grid_size = 1; // WTF ist this???
    const uint32_t BLOCK_THREADS = 1024;
    const uint32_t ITEMS_PER_THREAD = 4;
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    uint32_t* h_sa = new uint32_t[TILE_SIZE];
    bool* h_head_flags = new bool[TILE_SIZE];

    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        h_sa[i] = (i/2) % 10;
    }

    uint32_t* d_sa = NULL;
    bool* d_head_flags = NULL;

    cudaMalloc((void**)&d_sa, sizeof(uint32_t) * TILE_SIZE);
    cudaMalloc((void**)&d_head_flags, sizeof(bool) * TILE_SIZE);

    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        std::cout << h_sa[i] << " ";
        h_head_flags[i] = false;
    }
    std::cout << std::endl;

    // copy problem to device
    cudaMemcpy(d_sa, h_sa, sizeof(uint32_t) * TILE_SIZE, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    mark_heads<BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>>(d_sa, d_head_flags);

    cudaDeviceSynchronize();

    // copy solution to host
    cudaMemcpy(h_head_flags, d_head_flags, sizeof(bool) * TILE_SIZE, cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        std::cout << h_head_flags[i] << " ";
    }
    std::cout << std::endl;

    if(h_sa) delete[] h_sa;
    if(h_head_flags) delete[] h_head_flags;
    if(d_sa) cudaFree(d_sa);
    if(d_head_flags) cudaFree(d_head_flags);

    return 0;
}
