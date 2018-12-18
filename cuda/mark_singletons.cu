#include <string>
#include <iostream>
#include "cub-1.8.0/cub/cub.cuh"
#include "cub-1.8.0/cub/block/block_load.cuh"
#include "cub-1.8.0/cub/block/block_store.cuh"

template <uint32_t BLOCK_THREADS, uint32_t ITEMS_PER_THREAD>
__global__
static void mark_heads(uint32_t* d_in, bool* head_flags) {

    // Specialize BlockDiscontinuity for a 1D block of 128 threads on type int
    typedef cub::BlockDiscontinuity<uint32_t, ITEMS_PER_THREAD> BlockDiscontinuity;
    // Allocate shared memory for BlockDiscontinuity
    __shared__ typename BlockDiscontinuity::TempStorage temp_storage;

    // Specialize BlockLoad type for our thread block
    typedef cub::BlockLoad<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT> BlockLoadT;
    // Specialize BlockStore type for our thread block
    typedef cub::BlockStore<bool, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_DIRECT> BlockStoreT;

    // Shared memory
    __shared__ union TempStorage
    {
        typename BlockLoadT::TempStorage    load;
        typename BlockStoreT::TempStorage   store;
    } storage;

    // Obtain a segment of consecutive items that are blocked across threads
    uint32_t thread_data[ITEMS_PER_THREAD];
    BlockLoadT(storage.load).Load(d_in, thread_data);

    // block of flags per thread
    bool flags[ITEMS_PER_THREAD];
    // Collectively compute head flags for discontinuities in the segment
    BlockDiscontinuity(temp_storage).FlagHeads(flags, thread_data, cub::Inequality());

    // Store flags from a blocked arrangement
    BlockStoreT(storage.store).Store(head_flags, flags);
}

int main()
{
    uint32_t n = 8;
    uint32_t* sa = new uint32_t[8];
    bool* head_flags = new bool[8];
    sa[4] = 1;

    for (uint32_t i = 0; i < n; ++i) {
        std::cout << sa[i];
    }
    std::cout << std::endl;

    //cudaMallocManaged(&sa, n*sizeof(uint32_t));
    //cudaMallocManaged(&head_flags, n*sizeof(bool));

    mark_heads<2,4><<<2,4>>>(sa, head_flags);

    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < n; ++i) {
        std::cout << head_flags[i];
    }
    std::cout << std::endl;

    if(sa) delete[] sa;
    if(head_flags) delete[] head_flags;

    return 0;
}
