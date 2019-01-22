#include <string>
#include <iostream>
#include "cub-1.8.0/cub/cub.cuh"

#include "cuda_wrapper_interface.hpp"
#include "cuda_util.cuh"

#include "prefix_doubler_interface.hpp"

#define NUM_BLOCKS 2048
#define NUM_THREADS_PER_BLOCK 256

template <typename sa_index>
struct utils {
    static constexpr sa_index NEGATIVE_MASK = size_t(1)
                                              << (sizeof(sa_index) * 8 - 1);
};


template <typename sa_index>
struct Compare_four_chars
{
public:
    Compare_four_chars(sa_index* _text) : text(_text) {};
    sa_index* text;
    template <typename index>
    CUB_RUNTIME_FUNCTION __forceinline__ __device__
    bool operator()(const index &x, const index &y) const {
        return text[x]<text[y];
    }
};

//Quick and dirty version, which packs four chars in one sa_index (either
//uint32_t or uint64_t)
template <typename sa_index>
void word_packing_generic(const uint8_t* chars, sa_index* result, size_t n) {

    typedef unsigned char u8;
    if(n>3) {
    for(size_t i = 0; i<n-3 ;++i) {
        result[i] = ((u8)chars[i] << 24) | ((u8)chars[i+1] << 16) | ((u8)chars[i+2] << 8) | (u8)chars[i+3];
    }
    }
    if(n>2) {
    result[n-3] = ((u8)chars[n-3] << 24) | ((u8)chars[n-2] << 16) | ((u8)chars[n-1] << 8);
    }
    if(n>1) {
    result[n-2] = ((u8)chars[n-2] << 24) | ((u8)chars[n-1] << 16);
    }
    if(n>0) {
    result[n-1] = ((u8)chars[n-1] << 24);
    }

}

//Quick and dirty version, which packs four chars in one sa_index (either
// uint64_t
void word_packing_64(const char* chars, uint64_t* result, size_t n) {

    for(size_t i = 0; i<n-7 ;++i) {
        result[i] = ((uint64_t)chars[i] << 56) | ((uint64_t)chars[i+1] << 48) | ((uint64_t)chars[i+2] << 40) | ((uint64_t)chars[i+3] << 32) |
                    ((uint64_t)chars[i] << 24) | ((uint64_t)chars[i+1] << 16) | ((uint64_t)chars[i+2] << 8) | (uint64_t)chars[i+3];
    }

    result[n-7] = ((uint64_t)chars[n-7] << 56) | ((uint64_t)chars[n-6] << 48) | ((uint64_t)chars[n-5] << 40) | ((uint64_t)chars[n-4] << 32) |
                    ((uint64_t)chars[n-3] << 24) | ((uint64_t)chars[n-2] << 16) | ((uint64_t)chars[n-1] << 8);

    result[n-6] = ((uint64_t)chars[n-6] << 56) | ((uint64_t)chars[n-5] << 48) | ((uint64_t)chars[n-4] << 40) | ((uint64_t)chars[n-3] << 32) |
                    ((uint64_t)chars[n-2] << 24) | ((uint64_t)chars[n-1] << 16);
    result[n-5] = ((uint64_t)chars[n-5] << 56) | ((uint64_t)chars[n-4] << 48) | ((uint64_t)chars[n-3] << 40) | ((uint64_t)chars[n-2] << 32) |
                    ((uint64_t)chars[n-3] << 1);
    result[n-4] = ((uint64_t)chars[n-4] << 56) | ((uint64_t)chars[n-3] << 48) | ((uint64_t)chars[n-2] << 40) | ((uint64_t)chars[n-1] << 32);


    result[n-3] = ((uint64_t)chars[n-3] << 56) | ((uint64_t)chars[n-2] << 48) | ((uint64_t)chars[n-1] << 40);
    result[n-2] = ((uint64_t)chars[n-2] << 56) | ((uint64_t)chars[n-1] << 48);
    result[n-1] = ((uint64_t)chars[n-1] << 56);

}

void word_packing(const uint8_t* chars, uint32_t* result, size_t n) {
    word_packing_generic(chars, result, n);
}

void word_packing(const uint8_t* chars, uint64_t* result, size_t n) {
    word_packing_generic(chars, result, n);
}

/*
    \brief Kernel function for setting diff flags.
    DEPRECATED
*/
template <typename sa_index>
__global__
static void set_flags_kernel(size_t size, sa_index* sa, sa_index* isa,
            sa_index* aux) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    /*
    // Set values in aux to 0 (except first value)
    for(size_t i=index+10; i < size; i+=stride) {
        aux[i] = 0;
    }*/

    if(index == 0) {
        // First index has no predecessor -> mark as true by default
        aux[0] = 1;
    }
    // Avoid iteration for index 0
    // Set flags if different rank to predecessor
    for(size_t i=index+1; i < size; i+=stride) {
        aux[i] = (isa[sa[i-1]] != isa[sa[i]]);
    }
}

/*
    \brief Kernel function (uint32_t) for setting diff flags.
*/
__global__
static void set_flags_kernel_32(size_t size, uint32_t* sa, uint32_t* isa,
            uint32_t* aux) {
    extern __shared__ uint32_t smem_32[];
    uint32_t* s_isa = smem_32;
    uint32_t* s_sa = &s_isa[NUM_THREADS_PER_BLOCK+1];

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if(index == 0) {
        // First index has no predecessor -> mark as true by default
        aux[0] = 1;
    }
    // Avoid iteration for index 0
    // Set flags if different rank to predecessor
    for(size_t i=index+1; i < size; i+=stride) {
        // Load into shared memory according to stride offset
        // Load first element with current stride offset
        if(threadIdx.x == 0) {
            s_sa[0] = sa[i-1];
            s_isa[0] = isa[s_sa[0]];
        }
        // Load second isa value for following computation
        s_sa[threadIdx.x+1] = sa[i];
        s_isa[threadIdx.x+1] = isa[s_sa[threadIdx.x+1]];

        __syncthreads();

        aux[i] = (s_isa[threadIdx.x] != s_isa[threadIdx.x+1]);
        // aux[i] = (isa[sa[i-1]] != isa[sa[i]]);
    }
}
/*
    \brief Kernel function (uint64_t) for setting diff flags.

    Shared memory needs different name to 32-bit Version because cuda sometimes
    behaves strangely. This also caused the split of the kernel for different
    types (shared memory doesn't support template parameters).
*/
__global__
static void set_flags_kernel_64(size_t size, uint64_t* sa, uint64_t* isa,
            uint64_t* aux) {
    extern __shared__ uint64_t smem_64[];
    uint64_t* s_isa = smem_64;
    uint64_t* s_sa = &s_isa[NUM_THREADS_PER_BLOCK+1];

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if(index == 0) {
        // First index has no predecessor -> mark as true by default
        aux[0] = 1;
    }


    // Avoid iteration for index 0
    // Set flags if different rank to predecessor
    for(size_t i=index+1; i < size; i+=stride) {
        // Load into shared memory according to stride offset
        // Load first element with current stride offset
        if(threadIdx.x == 0) {
            s_sa[0] = sa[i-1];
            s_isa[0] = isa[s_sa[0]];
        }
        // Load second isa value (isa[sa[i+1]] for thread i) for following
        // computation
        s_sa[threadIdx.x+1] = sa[i];
        s_isa[threadIdx.x+1] = isa[s_sa[threadIdx.x+1]];

        __syncthreads();

        aux[i] = (s_isa[threadIdx.x] != s_isa[threadIdx.x+1]);
        // aux[i] = (isa[sa[i-1]] != isa[sa[i]]);
    }
}

void set_flags(size_t size, uint32_t* sa, uint32_t* isa, uint32_t* aux) {
    //std::cout << "Calling 32 Version of set_flags." << std::endl;
    set_flags_kernel_32<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK,
            2*(NUM_THREADS_PER_BLOCK+1)*sizeof(uint32_t)>>>(size, sa, isa, aux);
    cudaDeviceSynchronize();
}

void set_flags(size_t size, uint64_t* sa, uint64_t* isa, uint64_t* aux) {
    //std::cout << "Calling 64 Version of set_flags." << std::endl;
    set_flags_kernel_64<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK,
            2*(NUM_THREADS_PER_BLOCK+1)*sizeof(uint64_t)>>>(size, sa, isa, aux);
    cudaDeviceSynchronize();
}

/*
    \brief Kernel function for checking wether a group should be marked,
    i.e. inverting its rank.
*/
template <typename sa_index>
__global__
static void mark_groups_kernel(size_t size, sa_index* sa, sa_index* isa,
            sa_index* aux) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    // Check if group is singleton by checking wether flag has been set for
    // element and successor
    for(size_t i=index; i < size-1; i+=stride) {
        if(aux[i] + aux[i+1] > 1) {
            // Condition met -> invert rank of suffix sa[i]
            isa[sa[i]] = isa[sa[i]] | utils<sa_index>::NEGATIVE_MASK;
        }
    }
    // Separate check for last suffix because it has no successor
    if(aux[size-1] > 0) {
        isa[sa[size-1]] = isa[sa[size-1]] | utils<sa_index>::NEGATIVE_MASK;
    }
}

__global__
static void mark_groups_kernel_32(size_t size, uint32_t* sa, uint32_t* isa,
            uint32_t* aux) {
    extern __shared__ uint32_t s_aux_32[];
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    // Check if group is singleton by checking wether flag has been set for
    // element and successor
    for(size_t i=index; i < size-1; i+=stride) {
        // Load aux values into shared memory.

        if(threadIdx.x == 0) {
            s_aux_32[0] = aux[i];
        }

        s_aux_32[threadIdx.x+1] = aux[i+1];

        __syncthreads();
        if(s_aux_32[threadIdx.x] + s_aux_32[threadIdx.x+1] > 1) {
            // Condition met -> invert rank of suffix sa[i]
            isa[sa[i]] = isa[sa[i]] | utils<uint32_t>::NEGATIVE_MASK;
        }
    }
    // Separate check for last suffix because it has no successor
    if(index == 0 && aux[size-1] > 0) {
        isa[sa[size-1]] = isa[sa[size-1]] | utils<uint32_t>::NEGATIVE_MASK;
    }
}

__global__
static void mark_groups_kernel_64(size_t size, uint64_t* sa, uint64_t* isa,
            uint64_t* aux) {
    extern __shared__ uint64_t s_aux_64[];
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    // Check if group is singleton by checking wether flag has been set for
    // element and successor
    for(size_t i=index; i < size-1; i+=stride) {
        // Load aux values into shared memory.

        if(threadIdx.x == 0) {
            s_aux_64[0] = aux[i];
        }

        s_aux_64[threadIdx.x+1] = aux[i+1];

        __syncthreads();
        if(s_aux_64[threadIdx.x] + s_aux_64[threadIdx.x+1] > 1) {
            // Condition met -> invert rank of suffix sa[i]
            isa[sa[i]] = isa[sa[i]] | utils<uint64_t>::NEGATIVE_MASK;
        }
    }
    // Separate check for last suffix because it has no successor
    if(index == 0 && aux[size-1] > 0) {
        isa[sa[size-1]] = isa[sa[size-1]] | utils<uint64_t>::NEGATIVE_MASK;
    }
}

void mark_groups(size_t size, uint32_t* sa, uint32_t* isa, uint32_t* aux) {
    mark_groups_kernel_32<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK,
            (NUM_THREADS_PER_BLOCK+1)*sizeof(uint32_t)>>>(size, sa, isa,
                aux);
    cudaDeviceSynchronize();
}

void mark_groups(size_t size, uint64_t* sa, uint64_t* isa, uint64_t* aux) {
    mark_groups_kernel_64<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK,
            (NUM_THREADS_PER_BLOCK+1)*sizeof(uint64_t)>>>(size, sa, isa,
                aux);
    cudaDeviceSynchronize();
}

/*
    \brief Kernel for initializing the sa with the suffix positions (in text
    order).
*/
template <typename sa_index>
__global__
static void initialize_sa_gpu_kernel(size_t n, sa_index* sa) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i+=stride) {
        sa[i] = i;
    }

}

void initialize_sa_gpu(size_t size, uint32_t* sa) {
    initialize_sa_gpu_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, sa);
    cudaDeviceSynchronize();
}

void initialize_sa_gpu(size_t size, uint64_t* sa) {
    initialize_sa_gpu_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, sa);
    cudaDeviceSynchronize();
}
/*
    Calculates inclusive prefix sum on GPU using the provided CUB Method
*/
template <typename OP, typename sa_index>
void prefix_sum_cub_inclusive_kernel(sa_index* array, OP op, size_t n)
{
    //TODO: submit allocated memory instead of allocating new array
    //Indices
    //sa_index  *values_out;   // e.g., [        ...        ]

    // Allocate Unified Memory – accessible from CPU or GPU
    //cudaMallocManaged(&values_out, n*sizeof(sa_index));

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, array, array,op, n);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, array, array,op, n);

    cudaDeviceSynchronize();

    //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(array,values_out,n);

    //cudaMemcpy(array, values_out, n*sizeof(sa_index), cudaMemcpyDeviceToDevice);

    //cudaFree(values_out);

}


/*
    Auxiliary function for initializing ISA
    Computes inital aux array, with index if own value other to predecessor,
    else 0
*/
template <typename Comp, typename sa_index>
__global__
void fill_aux_for_isa_kernel(sa_index* sa, sa_index* aux, size_t n, Comp comp) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if(index == 0) {
        aux[0]=0;
    }

    for (size_t i = index+1; i < n; i+=stride) {
        aux[i] = i * (comp(sa[i - 1], sa[i]) != 0);
    }
}

void fill_aux_for_isa(uint32_t* text, uint32_t* sa, uint32_t* isa,
            size_t size) {
    auto cmp = Compare_four_chars<uint32_t>(text);
    fill_aux_for_isa_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(sa, isa,
                size, cmp);
    cudaDeviceSynchronize();
}

void fill_aux_for_isa(uint64_t* text, uint64_t* sa, uint64_t* isa, size_t size) {
    auto cmp = Compare_four_chars<uint64_t>(text);
    fill_aux_for_isa_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(sa, isa,
                size, cmp);
    cudaDeviceSynchronize();
}

/*
    Auxiliary function for initializing ISA
    writes aux in ISA
*/
template <typename sa_index>
__global__
void scatter_to_isa_kernel(sa_index* isa, sa_index* aux, sa_index* sa,
            size_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n; i+=stride) {
        isa[sa[i]]=aux[i];
    }
}

void scatter_to_isa(uint32_t* isa, uint32_t* aux, uint32_t* sa, size_t size) {
    scatter_to_isa_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(isa, aux, sa,
                size);
    cudaDeviceSynchronize();
}

void scatter_to_isa(uint64_t* isa, uint64_t* aux, uint64_t* sa, size_t size) {
    scatter_to_isa_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(isa, aux, sa,
                size);
    cudaDeviceSynchronize();
}

template <typename sa_index>
__global__
void update_ranks_build_aux_kernel(sa_index* h_ranks, sa_index* aux, size_t n) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if(index == 0) {
        aux[0]=0;
    }

    for (size_t i = index+1; i < n; i+=stride) {
        aux[i] = (h_ranks[i-1]!=h_ranks[i]) * i;
    }
}

void update_ranks_build_aux(uint32_t* h_ranks, uint32_t* aux, size_t size) {
    update_ranks_build_aux_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(
            h_ranks, aux, size);
    cudaDeviceSynchronize();
}

void update_ranks_build_aux(uint64_t* h_ranks, uint64_t* aux, size_t size) {
    update_ranks_build_aux_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(
            h_ranks, aux, size);
    cudaDeviceSynchronize();
}

template <typename sa_index>
__global__
void update_ranks_build_aux_tilde_kernel(sa_index* h_ranks, sa_index* two_h_ranks,
        sa_index* aux, size_t n) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if(index == 0) {
        aux[0]=h_ranks[0];
    }

    for (size_t i = index+1; i < n; i+=stride) {

        bool new_group = (h_ranks[i-1] != h_ranks[i]
            || two_h_ranks[i-1] != two_h_ranks[i]);
        // Werte in aux überschrieben?
        aux[i] = new_group * (h_ranks[i] + i - aux[i]);
    }
}

void update_ranks_build_aux_tilde(uint32_t* h_ranks, uint32_t* two_h_ranks,
        uint32_t* aux, size_t size) {
    update_ranks_build_aux_tilde_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(
            h_ranks, two_h_ranks, aux, size);
    cudaDeviceSynchronize();
}

void update_ranks_build_aux_tilde(uint64_t* h_ranks, uint64_t* two_h_ranks,
        uint64_t* aux, size_t size) {
    update_ranks_build_aux_tilde_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(
            h_ranks, two_h_ranks, aux, size);
    cudaDeviceSynchronize();
}

/*
\brief Sets values in aux array if tuples for suffix indices should be
created.
*/

template <typename sa_index>
__global__
void set_tuple_kernel_shared(size_t size, size_t h, sa_index* sa,
        sa_index* isa, sa_index* aux) {
    extern __shared__ int sa_buffer[];
    int t_index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    sa_index index;
    for(size_t i=t_index; i < size; i+=stride) {
        sa_buffer[threadIdx.x] = sa[i];
        aux[i] = 0;
        if(sa_buffer[threadIdx.x] >= h) {
            index = sa_buffer[threadIdx.x]-h;
            if((isa[index] & utils<sa_index>::NEGATIVE_MASK) ==
                    sa_index(0)) {
                ++aux[i];
            }
            // Second condition cannot be true if sa[i] < h
            index = sa_buffer[threadIdx.x];
            if((isa[index] & utils<sa_index>::NEGATIVE_MASK) > sa_index(0)
                    && index >= 2*h && (isa[index-2*h] &
                    utils<sa_index>::NEGATIVE_MASK) == sa_index(0)) {
                ++aux[i];
            }
        }
    }
}

void set_tuple(size_t size, size_t h, uint32_t* sa, uint32_t* isa,
            uint32_t* aux) {
    set_tuple_kernel_shared<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK*sizeof(uint32_t)>>>(size, h, sa, isa,
                aux);
    cudaDeviceSynchronize();
}

void set_tuple(size_t size, size_t h, uint64_t* sa, uint64_t* isa,
            uint64_t* aux) {
    set_tuple_kernel_shared<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, size*sizeof(uint64_t)>>>(size, h, sa, isa,
                aux);
    cudaDeviceSynchronize();
}
/*
\brief Creates tuples of <suffix index, h_rank> (missing: 2h_rank) by inserting
    them in the corresponding arrays with the help of aux (contains position for
    insertion)
*/
template <typename sa_index>
__global__
void new_tuple_kernel(size_t size, size_t h, sa_index* sa,
        sa_index* isa, sa_index* aux, sa_index* tuple_index,
        sa_index* h_rank) {
    int t_index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    // Using aux, sa_val and isa_val to reduce access to global memory
    sa_index index;
    for(size_t i=t_index; i < size; i+=stride) {
        if(sa[i] >= h) {
            index = sa[i]-h;
            if((isa[index] & utils<sa_index>::NEGATIVE_MASK) ==
                    sa_index(0)) {
                tuple_index[aux[i]] = index;
                // Increment aux[i] incase inducing suffix is also added
                h_rank[aux[i]++] = isa[index];
            }
            // Check if inducing suffix is also added.
            index = sa[i];
            if((isa[index] & utils<sa_index>::NEGATIVE_MASK) > sa_index(0)
                    && index >= 2*h && (isa[index-2*h] &
                    utils<sa_index>::NEGATIVE_MASK) == sa_index(0)) {
                tuple_index[aux[i]] = index;
                h_rank[aux[i]] = isa[index] ^
                    utils<sa_index>::NEGATIVE_MASK;
            }
        }
    }
}

void new_tuple(size_t size, size_t h, uint32_t* sa, uint32_t* isa,
            uint32_t* aux, uint32_t* tuple_index, uint32_t* h_rank) {
    new_tuple_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, h, sa, isa,
            aux, tuple_index, h_rank);
    cudaDeviceSynchronize();
}

void new_tuple(size_t size, size_t h, uint64_t* sa, uint64_t* isa,
            uint64_t* aux, uint64_t* tuple_index, uint64_t* h_rank) {
    new_tuple_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, h, sa, isa,
            aux, tuple_index, h_rank);
    cudaDeviceSynchronize();
}

/*
    Auxiliary function for initializing ISA
    writes aux in ISA
*/
template <typename sa_index>
__global__
void isa_to_sa_kernel(sa_index* isa, sa_index* sa, size_t n) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i+=stride) {
        sa[isa[i]^utils<sa_index>::NEGATIVE_MASK]=i;
    }
}

void isa_to_sa(uint32_t* isa, uint32_t* sa, size_t size) {
    isa_to_sa_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(isa, sa, size);
    cudaDeviceSynchronize();
}

void isa_to_sa(uint64_t* isa, uint64_t* sa, size_t size) {
    isa_to_sa_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(isa, sa, size);
    cudaDeviceSynchronize();
}

/*
    \brief Generates 2h-ranks after tuples have been sorted.
    TODO: Wrapper
*/
template <typename sa_index>
__global__
void generate_two_h_kernel(size_t size, size_t h, sa_index* sa, sa_index* isa,
            sa_index* two_h_rank) {
    int t_index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    for(size_t i=t_index; i < size; i+=stride) {
       // Case can only occur if index + h < max. index (of original sequence)
       if((isa[sa[i]] & utils<sa_index>::NEGATIVE_MASK)
                == sa_index(0)) {
           // Retrieve rank of 2h-suffix
           two_h_rank[i] = isa[sa[i]+h];
       } else {
           two_h_rank[i] = isa[sa[i]];
       }
   }
}

void generate_two_h_rank(size_t size, size_t h, uint32_t* sa,
            uint32_t* isa, uint32_t* two_h_rank) {
    generate_two_h_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, h, sa,
            isa, two_h_rank);
    cudaDeviceSynchronize();
}

void generate_two_h_rank(size_t size, size_t h, uint64_t* sa,
            uint64_t* isa, uint64_t* two_h_rank) {
    generate_two_h_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, h, sa,
            isa, two_h_rank);
    cudaDeviceSynchronize();
}


/*
    Copies one array size_to another by using GPU threads
    Maybe use memcpy? http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g48efa06b81cc031b2aa6fdc2e9930741.html
*/
/*
__global__
static void copy_to_array(size_t* in, size_t* out, size_t n) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i+=stride) {
        in[i] = out[i];
    }

}


*/

/*
template <typename sa_index, class osipov_impl>
static void prefix_doubling_gpu(sa_index* gpu_text, sa_index* out_sa,
            osipov_impl& osipov) {

    size_t size = osipov.get_size();
    size_t n = size;
    //Fill SA
    osipov.initialize_sa();
    cudaDeviceSynchronize();

    //Sort by four characters
    osipov.inital_sort();
    cudaDeviceSynchronize();
    auto sa = osipov.get_sa();
    std::cout<<"SA: ";
    for(size_t i = 0; i<size; ++i) {
        std::cout<<sa[i]<<", ";
    }
    std::cout<<std::endl;

    //Init ISA with group numbers according to initial sorting
    osipov.initialize_isa();
    cudaDeviceSynchronize();

    auto isa = osipov.get_isa();

    std::cout<<std::endl;
    std::cout<<"ISA: ";
    for(size_t i = 0 ; i< size ; ++i) {
        std::cout<<isa[sa[i]]<<", ";
    }
    std::cout<<std::endl;

    osipov.mark_singletons();

    auto aux = osipov.get_aux();

    std::cout << "Flags: ";
    for(size_t i = 0; i<size; ++i) {
        std::cout<<aux[i]<<", ";
    }
    std::cout<<std::endl;

    std::cout<<"ISA (with singletons): ";
    for(size_t i = 0 ; i< size ; ++i) {
        std::cout<<isa[sa[i]]<<", ";
    }
    std::cout<<std::endl;
    size_t h = 4;
    size_t s;

    auto h_rank = osipov.get_rank();
    auto two_h_rank = osipov.get_two_rank();

    while(size > 0) {
        osipov.slice_container(size);

        s = osipov.create_tuples(size, h);

        std::cout << "Tuples: ";
        for(size_t i=0; i < s; ++i) {
            std::cout << "<" << sa[i] << "," << h_rank[i] << ">, ";
        }
        std::cout << std::endl;

        // Continue iteration as usual only if there are elements left
        // (i.e. s>0)
        if(s>0) {
            osipov.slice_container(s);
            osipov.stable_sort();
            cudaDeviceSynchronize();

            std::cout << "Tuples after sorting: ";
            for(size_t i=0; i < s; ++i) {
                std::cout << "<" << sa[i] << "," << h_rank[i] <<">, ";
            }
            std::cout << std::endl;

            osipov.update_ranks(h);

            std::cout << "Updated h_rank: ";
            for(size_t i=0; i < s; ++i) {
                std::cout << "<" << sa[i] << "," << aux[i] << ">, ";
            }
            osipov.update_container(s);
            std::cout << std::endl;
            // TODO: UPDATE SA!
            osipov.mark_singletons();
        }
        // End of iteration; update size and h.
        size = s;
        h = 2*h;
    }
    osipov.finalize(n);
    std::cout<<"Result:"<<std::endl;
    for(int i = 0; i < n; ++i) {
        std::cout<<(sa[i])<<", ";
    }
    std::cout<<std::endl;
}

TODO: Move to osipov_gpu.hpp
int main()
{
    std::string text_str = "trhsrznttstrhrhvsrthsrcadcvsdnvsvoisemvosdinvaofmafvnsodivjasifn";
    const char* text = text_str.c_str();
    size_t n = text_str.size()+1;
    std::cout<<"n: "<<n<<std::endl;


    uint32_t* packed_text;
    packed_text = (uint32_t *) malloc(n*sizeof(uint32_t));
    //Pack text, so you can compare four chars at once
    word_packing(text, packed_text, n);

    //GPU arrays
    uint32_t* gpu_text;
    uint32_t* out_sa;
    gpu_text = allocate_managed_cuda_buffer_of<uint32_t>(n);
    //Copy text to GPU
    memset(gpu_text, 0, n*sizeof(uint32_t));
    cuda_check(cudaMemcpy(gpu_text, packed_text, n*sizeof(uint32_t), cudaMemcpyHostToDevice));
    out_sa = allocate_managed_cuda_buffer_of<uint32_t>(n);

    //additional arrays
    uint32_t* sa;
    uint32_t* isa;
    uint32_t* aux;

    uint32_t* h_rank;
    uint32_t* two_h_rank;
    //allocate additional arrays directly on GPU
    sa = allocate_managed_cuda_buffer_of<uint32_t>(n);
    isa = allocate_managed_cuda_buffer_of<uint32_t>(n);
    aux = allocate_managed_cuda_buffer_of<uint32_t>(n);
    h_rank = allocate_managed_cuda_buffer_of<uint32_t>(n);
    two_h_rank = allocate_managed_cuda_buffer_of<uint32_t>(n);

    cuda_check(cudaDeviceSynchronize());


    auto osipov = osipov_gpu<uint32_t>(n, gpu_text, sa, isa, aux, h_rank, two_h_rank);

    prefix_doubling_gpu<uint32_t, osipov_gpu<uint32_t>>(gpu_text, out_sa, osipov);

    return 0;
}

*/
/*
int main()
{
    std::string text_str = "caabaccaabacaa";
    const char* text = text_str.c_str();
    size_t n = text_str.size()+1;
    std::cout<<"n: "<<n<<std::endl;

    uint64_t* packed_text;
    packed_text = (uint64_t *) malloc(n*sizeof(uint64_t));
    for(int i = 0; i<n; ++i) {
        std::cout<<packed_text[i]<<",";
    }
    std::cout<<std::endl;

    //Pack text, so you can compare four chars at once
    word_packing_64(text, packed_text, n);

    for(int i = 0; i<n; ++i) {
        std::cout<<packed_text[i]<<",";
    }
    std::cout<<std::endl;

    return 0;
}
*/
