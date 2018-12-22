#include <string>
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

struct CustomAnd
{
    template <typename T>
    CUB_RUNTIME_FUNCTION __forceinline__ __device__
    T operator()(const T &a, const T &b) const {
        return a & b;
    }
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
    // Collectively compute head flags for discontinuities in the segment
    BlockDiscontinuity(temp_storage).FlagHeads(flags, thread_data, cub::Inequality());//BucketInequality(d_isa));
    // Store flags from a blocked arrangement
    BlockStoreT(store).Store(d_flags, flags);
}

template <typename OP>
void prefix_sum_inclusive(bool* flags_in, OP op, size_t n)
{
    //Indices
    bool* flags_out;   // e.g., [        ...        ]

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&flags_out, n*sizeof(bool));

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, flags_in, flags_out, op, n);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, flags_in, flags_out, op, n);

    cudaDeviceSynchronize();

    cudaMemcpy(flags_in, flags_out, n*sizeof(bool), cudaMemcpyDeviceToDevice);
}

__global__
void flag_entries(uint32_t* sa, bool* flags, uint32_t mask, uint32_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index+1; i < n; i+=stride) {
        sa[i] = sa[i] & (flags[i] * mask);
    }
}

int main()
{
    const uint32_t g_grid_size = 1; // WTF ist this???
    const uint32_t BLOCK_THREADS = 32;
    const uint32_t ITEMS_PER_THREAD = 4;
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    /*
       STEP 0: Initialization on host
     */

    uint32_t* h_sa = new uint32_t[TILE_SIZE];
    uint32_t* h_isa = new uint32_t[TILE_SIZE];
    bool* h_head_flags = new bool[TILE_SIZE + 1];

    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        h_isa[i] = (i/2) % 10;
        h_sa[i] = (i/2) % 10;
        h_head_flags[i] = false;

        std::cout << h_isa[i] << " ";
    }
    std::cout << std::endl;
    h_head_flags[TILE_SIZE] = true; // pseudo new bucket at the end

    // copy problem to device
    uint32_t* d_sa = NULL;
    uint32_t* d_isa = NULL;
    bool* d_head_flags = NULL;
    cudaMalloc((void**)&d_sa, sizeof(uint32_t) * TILE_SIZE);
    cudaMalloc((void**)&d_isa, sizeof(uint32_t) * TILE_SIZE);
    cudaMalloc((void**)&d_head_flags, sizeof(bool) * (TILE_SIZE + 1));
    cudaMemcpy(d_sa, h_sa, sizeof(uint32_t) * TILE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_isa, h_isa, sizeof(uint32_t) * TILE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_head_flags, h_head_flags, sizeof(bool) * (TILE_SIZE + 1), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    /* 
       STEP 1: Flag heads of groups
     */
    mark_heads<BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>>(d_sa, d_isa, d_head_flags);
    cudaDeviceSynchronize();

    /* 
       STEP 2: Reduce Flags to singleton groups 
     */
    CustomAnd and_op;
    prefix_sum_inclusive(d_head_flags, and_op, TILE_SIZE);  

    /*
       STEP 3: Mask SA entries
     */
    static constexpr uint32_t NEGATIVE_MASK = uint32_t(1) << (sizeof(uint32_t) * 8 - 1);
    flag_entries<<<ITEMS_PER_THREAD, BLOCK_THREADS>>>(d_sa, d_head_flags + sizeof(bool), NEGATIVE_MASK, TILE_SIZE);

    /*
       copy solution to host
     */
    cudaMemcpy(h_sa, d_sa, sizeof(uint32_t) * (TILE_SIZE), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_head_flags, d_head_flags, sizeof(bool) * (TILE_SIZE + 1), cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        std::cout << h_sa[i] << " ";
    }
    std::cout << std::endl;

    for (uint32_t i = 0; i < TILE_SIZE + 1; ++i) {
        std::cout << h_head_flags[i] << " ";
    }
    std::cout << std::endl;

    if(h_sa) delete[] h_sa;
    if(h_isa) delete[] h_isa;
    if(h_head_flags) delete[] h_head_flags;
    if(d_sa) cudaFree(d_sa);
    if(d_isa) cudaFree(d_isa);
    if(d_head_flags) cudaFree(d_head_flags);

    return 0;
}
