#include <string>
#include <bitset>
#include <iostream>
#include "cub-1.8.0/cub/cub.cuh"

/**
 * \brief Bucket inequality functor
 */
struct BucketInequality
{
    __host__ __device__ __forceinline__ BucketInequality(uint32_t* p_isa) : isa(p_isa) {}
    /// Boolean bucket inequality operator, returns <tt>(isa[suf_a] != isa[suf_b])</tt>
    template <typename T>
    __host__ __device__ __forceinline__ bool operator()(const T &suf_a, const T &suf_b) const
    {
        return isa[suf_a] != isa[suf_b];
    }

    private:
    uint32_t* isa;
};

template <uint32_t BLOCK_THREADS, uint32_t ITEMS_PER_THREAD>
__global__ static void mark_heads(uint32_t* d_sa, uint32_t* d_isa, bool* d_flags) {

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
    BlockLoadT(load).Load(d_sa, thread_data);

    // block of flags per thread
    bool flags[4];
    BucketInequality bucket_ineq(d_isa);
    // Collectively compute head flags for discontinuities in the segment
    BlockDiscontinuity(temp_storage).FlagHeads(flags, thread_data, bucket_ineq);
    // Store flags from a blocked arrangement
    BlockStoreT(store).Store(d_flags, flags);
}

/**
 * \brief Maps array of head flags to array of singleton flags
 */
__global__
void reduce_flags(bool* in_flags, bool* out_flags, uint32_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index+1; i < n; i+=stride) {
        out_flags[i] = in_flags[i] & in_flags[i+1];
    }
}

/**
  * \brief Applies a bitmask to all flagged entries in sa
  */
__global__
void mask_entries(uint32_t* sa, bool* flags, uint32_t mask, uint32_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index+1; i < n; i+=stride) {
        sa[i] = sa[i] | (flags[i] * mask);
    }
}

int main()
{
    const uint32_t g_grid_size = 1; // WTF ist this???
    const uint32_t BLOCK_THREADS = 32;
    const uint32_t ITEMS_PER_THREAD = 4;
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    /*
       STEP 0: Example initialization on host
     */

    uint32_t* h_sa = new uint32_t[TILE_SIZE];
    uint32_t* h_isa = new uint32_t[TILE_SIZE];
    bool* h_head_flags = new bool[TILE_SIZE + 1];

    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        h_isa[i] = (i % 3) / 2; // random example, not intended for production
        h_sa[i] = i;
        h_head_flags[i] = false;

    }
    h_head_flags[TILE_SIZE] = true; // pseudo new bucket at the end

    /* debug output
    std::cout << "ISA" << std::endl;
    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        std::cout << h_isa[i] << " ";
    }
    std::cout << std::endl;
     */

    // copy problem to device
    uint32_t* d_sa = NULL;
    uint32_t* d_isa = NULL;
    bool* d_head_flags = NULL;
    bool* d_singleton_flags = NULL;
    cudaMalloc((void**)&d_sa, sizeof(uint32_t) * TILE_SIZE);
    cudaMalloc((void**)&d_isa, sizeof(uint32_t) * TILE_SIZE);
    cudaMalloc((void**)&d_head_flags, sizeof(bool) * (TILE_SIZE + 1));
    cudaMalloc((void**)&d_singleton_flags, sizeof(bool) * (TILE_SIZE + 1));
    cudaMemcpy(d_sa, h_sa, sizeof(uint32_t) * TILE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_isa, h_isa, sizeof(uint32_t) * TILE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_head_flags, h_head_flags, sizeof(bool) * (TILE_SIZE + 1), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    /* 
       STEP 1: Flag heads of groups
     */
    mark_heads<BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>>(d_sa, d_isa, d_head_flags);
    cudaDeviceSynchronize();

    /* debug output
    std::cout << "Group Heads" << std::endl;
    cudaMemcpy(h_head_flags, d_head_flags, sizeof(bool) * (TILE_SIZE + 1), cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < TILE_SIZE + 1; ++i) {
        std::cout << h_head_flags[i] << " ";
    }
    std::cout << std::endl;
     */

    /* 
       STEP 2: Reduce Flags to singleton groups 
     */
    reduce_flags<<<ITEMS_PER_THREAD, BLOCK_THREADS>>>(d_head_flags, d_singleton_flags, TILE_SIZE);

    /*
       STEP 3: Mask SA entries
     */
    static constexpr uint32_t NEGATIVE_MASK = uint32_t(1) << (sizeof(uint32_t) * 8 - 1);
    mask_entries<<<ITEMS_PER_THREAD, BLOCK_THREADS>>>(d_sa, d_singleton_flags, NEGATIVE_MASK, TILE_SIZE);

    /*
       copy solution to host
     */
    cudaMemcpy(h_sa, d_sa, sizeof(uint32_t) * (TILE_SIZE), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_head_flags, d_singleton_flags, sizeof(bool) * (TILE_SIZE + 1), cudaMemcpyDeviceToHost);

    /* debug output
    std::cout << "Singleton Flags and Final SA" << std::endl;
    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        std::cout << h_head_flags[i] << " ";
        std::cout << std::bitset<32>(h_sa[i]) << " ";
        std::cout << std::endl;
    }
     */

    /*
       free memory
     */
    if(h_sa) delete[] h_sa;
    if(h_isa) delete[] h_isa;
    if(h_head_flags) delete[] h_head_flags;
    if(d_sa) cudaFree(d_sa);
    if(d_isa) cudaFree(d_isa);
    if(d_head_flags) cudaFree(d_head_flags);
    if(d_singleton_flags) cudaFree(d_singleton_flags);

    return 0;
}
