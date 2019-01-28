#include <string>
#include <iostream>
#include "../external/cub/cub/cub.cuh"

#include "cuda_wrapper_interface.hpp"
#include "cuda_util.cuh"

#include "prefix_doubler_interface.hpp"
#include <omp.h>


#define NUM_THREADS_PER_BLOCK 1024
#define NUM_BLOCKS(x) ((x + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK)


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
    #pragma omp parallel for
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

    #pragma omp parallel for
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
*/
template <typename sa_index>
__global__
static void set_flags_kernel(size_t size, sa_index* sa, sa_index* isa,
            sa_index* aux) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

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

void set_flags(size_t size, uint32_t* sa, uint32_t* isa, uint32_t* aux) {
    //std::cout << "Calling 32 Version of set_flags." << std::endl;
    set_flags_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(size, sa, isa, aux);
    //cudaDeviceSynchronize();
}

void set_flags(size_t size, uint64_t* sa, uint64_t* isa, uint64_t* aux) {
    //std::cout << "Calling 64 Version of set_flags." << std::endl;
    set_flags_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(size, sa, isa, aux);
    //cudaDeviceSynchronize();
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


void mark_groups(size_t size, uint32_t* sa, uint32_t* isa, uint32_t* aux) {
    mark_groups_kernel<<<NUM_BLOCKS(size),NUM_THREADS_PER_BLOCK>>>(size, sa, isa,
                aux);
    //cudaDeviceSynchronize();
}

void mark_groups(size_t size, uint64_t* sa, uint64_t* isa, uint64_t* aux) {
    mark_groups_kernel<<<NUM_BLOCKS(size),NUM_THREADS_PER_BLOCK>>>(size, sa, isa,
                aux);
    //cudaDeviceSynchronize();
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
    initialize_sa_gpu_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(size, sa);
    //cudaDeviceSynchronize();
}

void initialize_sa_gpu(size_t size, uint64_t* sa) {
    initialize_sa_gpu_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(size, sa);
    //cudaDeviceSynchronize();
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
    fill_aux_for_isa_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(sa, isa,
                size, cmp);
   // cudaDeviceSynchronize();
}

void fill_aux_for_isa(uint64_t* text, uint64_t* sa, uint64_t* isa, size_t size) {
    auto cmp = Compare_four_chars<uint64_t>(text);
    fill_aux_for_isa_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(sa, isa,
                size, cmp);
   // cudaDeviceSynchronize();
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
    scatter_to_isa_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(isa, aux, sa,
                size);
    //cudaDeviceSynchronize();
}

void scatter_to_isa(uint64_t* isa, uint64_t* aux, uint64_t* sa, size_t size) {
    scatter_to_isa_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(isa, aux, sa,
                size);
    //cudaDeviceSynchronize();
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
    update_ranks_build_aux_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(
            h_ranks, aux, size);
    //cudaDeviceSynchronize();
}

void update_ranks_build_aux(uint64_t* h_ranks, uint64_t* aux, size_t size) {
    update_ranks_build_aux_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(
            h_ranks, aux, size);
    //cudaDeviceSynchronize();
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
    update_ranks_build_aux_tilde_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(
            h_ranks, two_h_ranks, aux, size);
    //cudaDeviceSynchronize();
}

void update_ranks_build_aux_tilde(uint64_t* h_ranks, uint64_t* two_h_ranks,
        uint64_t* aux, size_t size) {
    update_ranks_build_aux_tilde_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(
            h_ranks, two_h_ranks, aux, size);
    //cudaDeviceSynchronize();
}

/*
\brief Sets values in aux array if tuples for suffix indices should be
created.
*/
template <typename sa_index>
__global__
void set_tuple_kernel(size_t size, size_t h, sa_index* sa,
        sa_index* isa, sa_index* aux) {
    int t_index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    sa_index index;
    for(size_t i=t_index; i < size; i+=stride) {
        aux[i] = 0;
        if(sa[i] >= h) {
            index = sa[i]-h;
            if((isa[index] & utils<sa_index>::NEGATIVE_MASK) ==
                    sa_index(0)) {
                ++aux[i];
            }
            // Second condition cannot be true if sa[i] < h
            index = sa[i];
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
    set_tuple_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(size, h, sa, isa,
                aux);
    //cudaDeviceSynchronize();
}

void set_tuple(size_t size, size_t h, uint64_t* sa, uint64_t* isa,
            uint64_t* aux) {
    set_tuple_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(size, h, sa, isa,
                aux);
    //cudaDeviceSynchronize();
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
    new_tuple_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(size, h, sa, isa,
            aux, tuple_index, h_rank);
    //cudaDeviceSynchronize();
}

void new_tuple(size_t size, size_t h, uint64_t* sa, uint64_t* isa,
            uint64_t* aux, uint64_t* tuple_index, uint64_t* h_rank) {
    new_tuple_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(size, h, sa, isa,
            aux, tuple_index, h_rank);
    //cudaDeviceSynchronize();
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
    isa_to_sa_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(isa, sa, size);
    //cudaDeviceSynchronize();
}

void isa_to_sa(uint64_t* isa, uint64_t* sa, size_t size) {
    isa_to_sa_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(isa, sa, size);
    //cudaDeviceSynchronize();
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
    generate_two_h_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(size, h, sa,
            isa, two_h_rank);
    //cudaDeviceSynchronize();
}

void generate_two_h_rank(size_t size, size_t h, uint64_t* sa,
            uint64_t* isa, uint64_t* two_h_rank) {
    generate_two_h_kernel<<<NUM_BLOCKS(size), NUM_THREADS_PER_BLOCK>>>(size, h, sa,
            isa, two_h_rank);
    //cudaDeviceSynchronize();
}

