#include <string>
#include <iostream>
#include "cub-1.8.0/cub/cub.cuh"
#include "cuda_wrapper_interface.hpp"


#define NUM_BLOCKS 2
#define NUM_THREADS_PER_BLOCK 4

template <typename sa_index>
struct utils {
    static constexpr sa_index NEGATIVE_MASK = size_t(1)
                                              << (sizeof(sa_index) * 8 - 1);
};

struct Max_without_branching
{
    template <typename T>
    CUB_RUNTIME_FUNCTION __forceinline__ __device__
    T operator()(const T &x, const T &y) const {
        return (x ^ ((x ^ y) & -(x < y)));
    }
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

//Quick and dirty version, which packs four chars in one size_t
template <typename sa_index>
void word_packing(const char* chars, sa_index* result, size_t n) {

    typedef unsigned char u8;
    for(size_t i = 0; i<n-3 ;++i) {
        result[i] = ((u8)chars[i] << 24) | ((u8)chars[i+1] << 16) | ((u8)chars[i+2] << 8) | (u8)chars[i+3];
    }
    result[n-3] = ((u8)chars[n-3] << 24) | ((u8)chars[n-2] << 16) | ((u8)chars[n-1] << 8);
    result[n-2] = ((u8)chars[n-2] << 24) | ((u8)chars[n-1] << 16);
    result[n-1] = ((u8)chars[n-1] << 24);

}

/*
    Init SA on GPU. Every GPU thread writes his index size_to SA,
    then jumps stride size until end is reached
*/
template <typename sa_index>
__global__
static void initialize_sa_gpu(size_t n, sa_index*  sa) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i+=stride) {
        sa[i] = i;
    }

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
    Sorts SA according to text using the CUB Radixsort
*/
template <typename sa_index>
static void inital_sorting(sa_index* text, sa_index* sa, sa_index* aux, size_t n) {

     //Actual values
    sa_index  *keys_out;     // e.g., [        ...        ]


    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&keys_out, n*sizeof(sa_index));


    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs<sa_index, sa_index>(d_temp_storage, temp_storage_bytes,
        text, keys_out, sa, aux, n);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);


    // Run sorting operation
    cub::DeviceRadixSort::SortPairs<sa_index, sa_index>(d_temp_storage, temp_storage_bytes,
        text, keys_out, sa, aux, n);


    cudaDeviceSynchronize();

    //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(sa,aux,n);

    cudaMemcpy(sa, aux, n*sizeof(sa_index), cudaMemcpyDeviceToDevice);
}


/*
    Calculates inclusive prefix sum on GPU using the provided CUB Method
*/
template <typename OP, typename sa_index>
void prefix_sum_cub_inclusive(sa_index* array, OP op, size_t n)
{
    //Indices
    sa_index  *values_out;   // e.g., [        ...        ]

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&values_out, n*sizeof(sa_index));

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, array, values_out,op, n);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, array, values_out,op, n);

    cudaDeviceSynchronize();

    //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(array,values_out,n);

    cudaMemcpy(array, values_out, n*sizeof(sa_index), cudaMemcpyDeviceToDevice);


}
/*
    Auxiliary function for initializing ISA
    Computes inital aux array, with index if own value other to predecessor, else 0
*/
template <typename Comp, typename sa_index>
__global__
void fill_aux_for_isa(sa_index* sa, sa_index* aux, size_t n, Comp comp) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if(index == 0) {
        aux[0]=0;
    }

    for (size_t i = index+1; i < n; i+=stride) {
        aux[i] = i * (comp(sa[i - 1], sa[i]) != 0);
    }
}
/*
    Auxiliary function for initializing ISA
    writes aux in ISA
*/
template <typename sa_index>
__global__
void scatter_to_isa(sa_index* isa, sa_index* aux, sa_index* sa, size_t n) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    //Maybe TODO: Avoid Bank Conflicts
    for (size_t i = index; i < n; i+=stride) {
        isa[sa[i]]=aux[i];
    }
}

template <typename sa_index>
__global__
void update_ranks_build_aux(sa_index* two_h_ranks, sa_index* aux, size_t n) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if(index == 0) {
        aux[0]=0;
    }

    for (size_t i = index+1; i < n; i+=stride) {
        aux[i] = (two_h_ranks[i-1]!=two_h_ranks[i]) * i;
    }
}

template <typename sa_index>
__global__
void update_ranks_build_aux_tilde(sa_index* two_h_ranks, sa_index* h_ranks, sa_index* aux, size_t n) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if(index == 0) {
        aux[0]=two_h_ranks[0];
    }

    for (size_t i = index+1; i < n; i+=stride) {

        bool new_group = (two_h_ranks[i-1] != two_h_ranks[i] || h_ranks[i-1] != h_ranks[i]);

        aux[index] = new_group * (two_h_ranks[i] + i - aux[i]);
    }
}

template <typename sa_index>
void update_ranks(sa_index* two_h_ranks, sa_index* h_ranks, sa_index* aux, size_t n) {

    //Build Aux
    update_ranks_build_aux<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(two_h_ranks, aux, n);
    cudaDeviceSynchronize();

    //prefix sum over aux
    Max_without_branching max;
    prefix_sum_cub_inclusive(aux, max, n);
    cudaDeviceSynchronize();

    //Build aux "tilde"
    update_ranks_build_aux_tilde<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(two_h_ranks, h_ranks, aux, n);
    cudaDeviceSynchronize();

    //prefix sum over aux "tilde"
    prefix_sum_cub_inclusive(aux, max, n);
    cudaDeviceSynchronize();

    //Scatter to ISA TODO IN MAIN FUNCTION!
    //scatter_to_isa<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(isa, aux, sa, n);
}


/*
    Init ISA with prefix sum method
*/
template <typename Comp, typename sa_index>
void initialize_isa(sa_index* isa, sa_index* sa, sa_index* aux, size_t n, Comp comp) {

    fill_aux_for_isa<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(sa,aux,n, comp);


    cudaDeviceSynchronize();

    Max_without_branching max;

    prefix_sum_cub_inclusive(aux,max, n);

    cudaDeviceSynchronize();

    scatter_to_isa<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(isa,aux,sa,n);
}


/*
\brief Sets values in aux array if tuples for suffix indices should be
created.
*/
template <typename sa_index>
__global__
void set_tuple(size_t size, size_t h, sa_index* sa,
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
                aux[i] = 1;
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

/*
\brief Creates tuples of <suffix index, h_rank> (missing: 2h_rank) by inserting
    them in the corresponding arrays with the help of aux (contains position for
    insertion)
*/
template <typename sa_index>
__global__
void new_tuple(size_t size, size_t h, sa_index* sa,
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


/*
\brief Sets values in aux array if tuples for suffix indices should be
created.
template <typename sa_index>
__global__
void set_tuple(size_t size, size_t h, sa_index* sa,
        sa_index* isa, sa_index* aux) {
    int t_index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    // Using sa_val and aux in shared mem to reduce global mem access
    __shared__ sa_index index, sa_val, aux_val;
    for(size_t i=t_index; i < size; i+=stride) {
        //TODO: Save in shared memory
        aux_val=0;
        sa_val = sa[i];
        if(sa_val >= h) {
            index = sa_val-h;
            if((isa[index] & utils<sa_index>::NEGATIVE_MASK) ==
                    sa_index(0)) {
                ++aux_val;
            }
            // Second condition cannot be true if sa[i] < h
            index = sa_val;
            if((isa[index] & utils<sa_index>::NEGATIVE_MASK) > sa_index(0)
                    && index >= 2*h && (isa[index-2*h] &
                    utils<sa_index>::NEGATIVE_MASK) == sa_index(0)) {
                ++aux_val;
            }
        }
        aux[i] = aux_val;
        aux = 0;
    }
}
*/
/*
\brief Creates tuples of <suffix index, h_rank> (missing: 2h_rank) by inserting
    them in the corresponding arrays with the help of aux (contains position for
    insertion)
template <typename sa_index>
__global__
void new_tuple(size_t size, size_t h, sa_index* sa,
        sa_index* isa, sa_index* aux, sa_index* tuple_index,
        sa_index* h_rank) {
    int t_index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    // Using aux, sa_val and isa_val to reduce access to global memory
    __shared__ sa_index index;
    // Optional: remove isa from shared memory
    __shared sa_index* sa_shared, isa_shared, aux_shared;
    for(size_t i=t_index; i < size; i+=stride) {
        aux_val = aux[i];
        sa_val = sa[i];
        if(sa_val >= h) {
            index = sa_val-h;
            isa_val = isa[index];
            if((isa_val & utils<sa_index>::NEGATIVE_MASK) ==
                    sa_index(0)) {
                tuple_index[aux_val] = index;
                // Increment aux[i] incase inducing suffix is also added
                h_rank[aux_val++] = isa_val;
            }
            // Check if inducing suffix is also added.
            index = sa_val;
            isa_val = isa[sa_val];
            if((isa_val & utils<sa_index>::NEGATIVE_MASK) > sa_index(0)
                    && index >= 2*h && (isa[index-2*h] &
                    utils<sa_index>::NEGATIVE_MASK) == sa_index(0)) {
                tuple_index[aux_val] = index;
                h_rank[aux_val] = isa_val ^
                    utils<sa_index>::NEGATIVE_MASK;
            }
        }
    }
}
*/
template <typename sa_index>
size_t create_tuples(size_t size, size_t h, sa_index* sa,
        sa_index* isa, sa_index* aux, sa_index* tuple_index,
        sa_index* h_rank) {
    size_t s=0;
    //TODO: Set block_amount and block_size accordingly
    set_tuple<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, h, sa, isa, aux);
    cudaDeviceSynchronize();
    // Save amount of tuples for last index (gets overwritten by prefix sum)
    s = aux[size-1];
    // Prefix sum
    exclusive_sum(aux, aux, size);
    cudaDeviceSynchronize();
    // Adjust s
    s += aux[size-1];
    new_tuple<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, h, sa, isa, aux,
            tuple_index, h_rank);
    cudaDeviceSynchronize();
    return s;
}

size_t create_tuples(size_t size, size_t h, uint64_t* sa, uint64_t* isa,
        uint64_t* aux, uint64_t* tuple_index, uint64_t* h_rank) {
            return create_tuples<uint64_t>(size, h, sa, isa, aux, tuple_index,
                h_rank);
        }

size_t create_tuples(size_t size, size_t h, uint32_t* sa, uint32_t* isa,
        uint32_t* aux, uint32_t* tuple_index, uint32_t* h_rank) {
            return create_tuples<uint32_t>(size, h, sa, isa, aux, tuple_index,
                h_rank);
        }

/*
size_t create_tuples(size_t size, size_t h, int* sa, int* isa, int* aux,
            int* tuple_index, int* h_rank) {
    return create_tuples<int>(size, h, sa, isa, aux, tuple_index, h_rank);
}*/

template <typename sa_index>
void sort_tuples(sa_index* tuple_index, sa_index* two_h_ranks, sa_index* aux1, sa_index* aux2 ,size_t n) {


     // Determine temporary device storage requirements
     void     *d_temp_storage = NULL;
     size_t   temp_storage_bytes = 0;

     cub::DeviceRadixSort::SortPairs<sa_index, sa_index>(d_temp_storage, temp_storage_bytes,
         two_h_ranks, aux1, tuple_index, aux2, n);
     // Allocate temporary storage
     cudaMalloc(&d_temp_storage, temp_storage_bytes);

     // Run sorting operation
     cub::DeviceRadixSort::SortPairs<sa_index, sa_index>(d_temp_storage, temp_storage_bytes,
        two_h_ranks, aux1, tuple_index, aux2, n);

     cudaDeviceSynchronize();

     //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(tuple_index,aux1,n);
     cudaMemcpy(tuple_index, aux1, n*sizeof(sa_index), cudaMemcpyDeviceToDevice);
     cudaMemcpy(two_h_ranks, aux2, n*sizeof(sa_index), cudaMemcpyDeviceToDevice);

     //cudaDeviceSynchronize();
     //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(two_h_ranks,aux2,n);



}

static void prefix_doubling_gpu(uint32_t* gpu_text, uint32_t* out_sa, size_t n) {

    //additional arrays
    uint32_t* sa;
    uint32_t* isa;
    uint32_t* aux;

    //allocate additional arrays directly on GPU
    cudaMallocManaged(&sa, n*sizeof(uint32_t));
    cudaMallocManaged(&isa, n*sizeof(uint32_t));
    cudaMallocManaged(&aux, n*sizeof(uint32_t));
    cudaDeviceSynchronize();

    //Fill SA
    initialize_sa_gpu<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(n, sa);
    cudaDeviceSynchronize();

    //Sort by four characters
    inital_sorting(gpu_text, sa, aux, n);
    cudaDeviceSynchronize();

    std::cout<<"SA: ";
    for(size_t i = 0; i<n; ++i) {
        std::cout<<sa[i]<<", ";
    }
    std::cout<<std::endl;

    //Init ISA with group numbers according to initial sorting
    Compare_four_chars<uint32_t> comp(gpu_text);
    initialize_isa(isa, sa, aux, n, comp);
    cudaDeviceSynchronize();

    std::cout<<std::endl;
    std::cout<<"ISA: ";
    for(size_t i = 0 ; i< n ; ++i) {
        std::cout<<isa[sa[i]]<<", ";
    }
    std::cout<<std::endl;

    size_t h = 4;

    uint32_t* tuple_index;
    uint32_t* h_rank;
    cudaMallocManaged(&tuple_index, n*sizeof(uint32_t));
    cudaMallocManaged(&h_rank, n*sizeof(uint32_t));

    std::cout << "Creating tuples." << std::endl;
    create_tuples(n, h, sa, isa, aux, tuple_index, h_rank);

    std::cout << "Tuples: ";
    for(size_t i=0; i < n; ++i) {
        std::cout << "<" << tuple_index[i] << "," << h_rank[i] << ">, ";
    }
    std::cout << std::endl;
/*
    phase.split("Mark singletons");
    mark_singletons(sa, isa);
    phase.split("Loop Initialization");

    // std::cout << "isa: " << isa << std::endl;
    size_t size = sa.size();
    size_t s = 0;

    auto tuple_container =
        util::make_container<std::tuple<sa_index, sa_index, sa_index>>(
            size);
    util::span<std::tuple<sa_index, sa_index, sa_index>> tuples;
    compare_tuples<sa_index> cmp;
    while (size > 0) {
        phase.split("Iteration");
        aux = util::span<sa_index>(aux_container).slice(0, size);
        tuples = tuple_container.slice(0, size);

        //s = create_tuples<sa_index>(tuples.slice(0, size), size, h, sa, isa);
        s = create_tuples_parallel<sa_index>(tuples.slice(0, size),
                size, h, sa, isa, aux);
        //std::cout << "Elements left: " << size << std::endl;

        // std::cout << "Next size: " << s << std::endl;
        // Skip all operations till size gets its new size, if this
        // iteration contains no tuples
        if (s > 0) {
            tuples = tuples.slice(0, s);
            aux = util::span<sa_index>(aux).slice(0, s);
            // std::cout << "Sorting tuples." << std::endl;
            cmp = compare_tuples<sa_index>(tuples);
            util::sort::std_par_stable_sort(tuples, cmp);
            sa = sa.slice(0, s);
            update_ranks_prefixsum(tuples, aux);
            // std::cout << "Writing new order to sa." << std::endl;
            for (size_t i = 0; i < s; ++i) {
                sa[i] = std::get<0>(tuples[i]);
            }

            for (size_t i = 0; i < s; ++i) {

                isa[std::get<0>(tuples[i])] =
                    aux[i]; // std::get<1>(tuples[i]);
            }
            mark_singletons(sa, isa);
        }
        size = s;
        h = 2 * h;
    }
    phase.split("Write out SA");
    for (size_t i = 0; i < out_sa.size(); ++i) {
        out_sa[isa[i] ^ utils<sa_index>::NEGATIVE_MASK] = i;
    }
    */
}
int main()
{
    std::string text_str = "caabaccaabacaa";
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
    cudaMallocManaged(&gpu_text, n*sizeof(uint32_t));
    //Copy text to GPU
    memset(gpu_text, 0, n*sizeof(uint32_t));
    cudaMemcpy(gpu_text, packed_text, n*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMallocManaged(&out_sa, n*sizeof(uint32_t));
    cudaDeviceSynchronize();


    prefix_doubling_gpu(gpu_text, out_sa, n);

    return 0;
}
