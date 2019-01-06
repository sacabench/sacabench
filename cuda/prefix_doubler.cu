#include <string>
#include <iostream>
#include "cub-1.8.0/cub/cub.cuh"


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

//Quick and dirty version, which packs four chars in one sa_index (either
//uint32_t or uint64_t)
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
    \brief Kernel function for setting diff flags.
*/
template <typename sa_index>
__global__
static void set_flags(size_t size, sa_index* sa, sa_index* isa,
            sa_index* aux) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    /*
    // Set values in aux to 0 (except first value)
    for(size_t i=index+10; i < size; i+=stride) {
        aux[i] = 0;
    }*/

    // Avoid iteration for index 0
    // Set flags if different rank to predecessor
    for(size_t i=index+1; i < size; i+=stride) {
        aux[i] = isa[sa[i-1]] != isa[sa[i]] ? 1 : 0;
    }
}

/*
    \brief Kernel function for checking wether a group should be marked,
    i.e. inverting its rank.
*/
template <typename sa_index>
__global__
static void mark_groups(size_t size, sa_index* sa, sa_index* isa,
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

/*
    \brief Kernel for initializing the sa with the suffix positions (in text
    order).
*/
template <typename sa_index>
__global__
static void initialize_sa_gpu(size_t n, sa_index* sa) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i+=stride) {
        sa[i] = i;
    }
}

/*
    Calculates inclusive prefix sum on GPU using the provided CUB Method
*/
template <typename OP, typename sa_index>
void prefix_sum_cub_inclusive(sa_index* array, OP op, size_t n)
{
    //TODO: submit allocated memory instead of allocating new array
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
    Computes inital aux array, with index if own value other to predecessor,
    else 0
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

template <typename sa_index>
struct osipov_gpu {
private:
    sa_index* text;

    Compare_four_chars<sa_index> cmp;
    // Three base arrays
    sa_index* sa;
    sa_index* isa;
    sa_index* aux;
    size_t size;

    // Arrays for tuples (use sa for tuple_index)
    // "Inducing reference"
    sa_index* two_h_rank;
    // Rank at depth h
    sa_index* h_rank;



public:
    /*
    sa_index* text;

    Compare_four_chars<sa_index> cmp;
    // Three base arrays
    sa_index* sa;
    sa_index* isa;
    sa_index* aux;
    size_t size;

    // Arrays for tuples (use sa for tuple_index)
    // "Inducing reference"
    sa_index* two_h_rank;
    // Rank at depth h
    sa_index* h_rank;

    */

    osipov_gpu(size_t size, sa_index* text, sa_index* sa, sa_index* isa,
            sa_index* aux, sa_index* two_h_rank, sa_index* h_rank) : size(size),
            text(text), sa(sa), isa(isa), aux(aux), two_h_rank(two_h_rank),
            h_rank(h_rank), cmp(Compare_four_chars<sa_index>(text)) {}

    /*
        Getter for debugging.
    */

    sa_index* get_sa() {return sa;}

    sa_index* get_isa() {return isa;}

    sa_index* get_aux() {return aux;}

    sa_index* get_rank() {return h_rank;}

    sa_index* get_two_rank() {return two_h_rank;}

    size_t get_size() {return size;}

    /*
        \brief slices array, i.e. sets the correct size.
    */
    void slice_container(size_t s) {size = s;}

    /*
        \brief method which is called in osipov.hpp but doesn't actually do
        anything.
        TODO: Check if logic can be moved to slice_container in cpu versions.
    */
    void slice_sa(size_t s) {}

    /*
        \brief PLACEHOLDER FUNCTION: Checks each h-group for singleton groups and
        marks these groups by inverting them.

        TODO: Replace with efficient version (see mark_singletons.cu).
    */
    void mark_singletons() {
        if(size > 0) {
            // First index has no predecessor -> mark as true by default
            aux[0] = 1;

            // Move sa, isa to shared memory

            set_flags<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, sa, isa, aux);
            cudaDeviceSynchronize();
            mark_groups<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, sa, isa, aux);
            cudaDeviceSynchronize();
        }
    }

    /*
        \brief Init SA on GPU. Every GPU thread writes his index size_to SA,
        then jumps stride size until end is reached.

        Wrapper for kernel
    */
    void initialize_sa() {
        initialize_sa_gpu<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(size, sa);
    }

    /*
        Initially sorts SA according to text using the CUB Radixsort
    */
    void inital_sort() {

         //Actual values; use h_rank as temp storage
        auto keys_out = h_rank;     // e.g., [        ...        ]


        // Allocate Unified Memory – accessible from CPU or GPU
        // cudaMallocManaged(&keys_out, size*sizeof(sa_index));


        // Determine temporary device storage requirements
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;

        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            text, keys_out, sa, aux, size);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);


        // Run sorting operation
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            text, keys_out, sa, aux, size);


        cudaDeviceSynchronize();

        //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(sa,aux,n);

        cudaMemcpy(sa, aux, size*sizeof(sa_index), cudaMemcpyDeviceToDevice);

        cudaDeviceSynchronize();
    }

    void update_ranks() {

        //Build Aux
        update_ranks_build_aux<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(h_rank, aux, size);
        cudaDeviceSynchronize();

        //prefix sum over aux
        Max_without_branching max;
        prefix_sum_cub_inclusive(aux, max, size);
        cudaDeviceSynchronize();

        std::cout << "Aux after first pass: ";
        for(size_t i=0; i < size; ++i) {
            std::cout << aux[i] << ", ";
        }
        std::cout << std::endl;

        //Build aux "tilde"
        update_ranks_build_aux_tilde<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(h_rank, two_h_rank, aux, size);
        cudaDeviceSynchronize();

        //prefix sum over aux "tilde"
        prefix_sum_cub_inclusive(aux, max, size);
        cudaDeviceSynchronize();

        std::cout << "Aux after second pass: ";
        for(size_t i=0; i < size; ++i) {
            std::cout << aux[i] << ", ";
        }
        std::cout << std::endl;

        //Scatter to ISA TODO IN MAIN FUNCTION!
        //scatter_to_isa<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(isa, aux, sa, n);
    }


    /*
        Init ISA with prefix sum method
    */
    void initialize_isa() {

        fill_aux_for_isa<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(sa, aux, size, cmp);

        cudaDeviceSynchronize();

        Max_without_branching max;

        prefix_sum_cub_inclusive(aux, max, size);

        cudaDeviceSynchronize();

        scatter_to_isa<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(isa, aux, sa, size);

    }

    /*
        \brief Extracts the suffix index and the h-rank for all considered
        suffixes during this iteration.
    */
    size_t create_tuples(size_t size, size_t h) {
        size_t s=0;
        auto tuple_index = two_h_rank;
        //TODO: Set block_amount and block_size accordingly
        set_tuple<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, h, sa, isa, aux);


        cudaDeviceSynchronize();

        std::cout << "Aux init: ";
        for(size_t i=0; i < size; ++i) {
            std::cout << aux[i] << ", ";
        }
        std::cout << std::endl;

        // Save amount of tuples for last index (gets overwritten by prefix sum)
        s = aux[size-1];
        // Prefix sum
        //exclusive_sum(aux, aux, size);
        // Use h_rank as temporary array as it wasn't needed before

        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        // Receive needed temp storage
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, aux, h_rank, size);
        // Allocate needed temp storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, aux, h_rank, size);
        cudaFree(d_temp_storage);

        cudaDeviceSynchronize();

        cudaMemcpy(aux, h_rank, size*sizeof(sa_index), cudaMemcpyDeviceToDevice);

        std::cout << "Aux after prefix sum: ";
        for(size_t i=0; i < size; ++i) {
            std::cout << aux[i] << ", ";
        }
        std::cout << std::endl;

        // Adjust s by amount of tuples from first 'size-1' suffixes.
        s += aux[size-1];
        new_tuple<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(size, h, sa, isa, aux,
                tuple_index, h_rank);
        cudaDeviceSynchronize();
        /*
            Copy tuple indices from temporary storage in tuple_index/two_h_rank
            to sa.
        */
        cudaMemcpy(tuple_index, sa, size*sizeof(sa_index),
                cudaMemcpyDeviceToDevice);
        return s;
    }
    /*
        \brief Generates the two-h ranks for each tuple using the corresponding
        kernel.
    */
    void generate_two_h_ranks(size_t size, size_t h) {
        generate_two_h_kernel<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(size, h, sa,
            isa, two_h_rank);
    }

    /*
        \brief Sorts generated tuples via radix sort.
    */
    void stable_sort() {
        auto aux1 = aux;
        auto aux2 = two_h_rank;

         // Determine temporary device storage requirements
         void     *d_temp_storage = NULL;
         size_t   temp_storage_bytes = 0;

         cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
             h_rank, aux1, sa, aux2, size);
         // Allocate temporary storage
         cudaMalloc(&d_temp_storage, temp_storage_bytes);

         // Run sorting operation
         cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
             h_rank, aux1, sa, aux2, size);

         cudaDeviceSynchronize();

         //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(tuple_index,aux1,n);
         cudaMemcpy(sa, aux2, size*sizeof(sa_index), cudaMemcpyDeviceToDevice);
         cudaMemcpy(h_rank, aux1, size*sizeof(sa_index), cudaMemcpyDeviceToDevice);



         //cudaDeviceSynchronize();
         //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(two_h_ranks,aux2,n);
    }

    void update_container(size_t s) {
        scatter_to_isa<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(isa, aux, sa, s);
    }



};

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
template <typename sa_index, class osipov_impl>
static void prefix_doubling_gpu(sa_index* gpu_text, sa_index* out_sa,
            osipov_impl& osipov) {
    size_t size = osipov.get_size();
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
            // Use two_h_rank as aux2 because it hasn't been filled for this
            // iteration
            osipov.stable_sort();
            cudaDeviceSynchronize();

            std::cout << "Tuples after sorting: ";
            for(size_t i=0; i < s; ++i) {
                std::cout << "<" << sa[i] << "," << h_rank[i] <<">, ";
            }
            std::cout << std::endl;

            // TODO: Move to either sort_tuples or update_ranks
            // Generate 2h-ranks after sorting
            osipov.generate_two_h_ranks(s, h);

            cudaDeviceSynchronize();
            std::cout << "Tuples with 2h-ranks: ";
            for(size_t i=0; i < s; ++i) {
                std::cout << "<" << sa[i] << "," << h_rank[i] << ","
                    << two_h_rank[i] <<">, ";
            }
            std::cout << std::endl;

            osipov.update_ranks();

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
    packed_text = (uint32_t *) malloc(n*sizeof(size_t));
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

    //additional arrays
    uint32_t* sa;
    uint32_t* isa;
    uint32_t* aux;

    uint32_t* h_rank;
    uint32_t* two_h_rank;
    //allocate additional arrays directly on GPU
    cudaMallocManaged(&sa, n*sizeof(uint32_t));
    cudaMallocManaged(&isa, n*sizeof(uint32_t));
    cudaMallocManaged(&aux, n*sizeof(uint32_t));
    cudaMallocManaged(&h_rank, n*sizeof(uint32_t));
    cudaMallocManaged(&two_h_rank, n*sizeof(uint32_t));

    cudaDeviceSynchronize();

    auto osipov = osipov_gpu<uint32_t>(n, gpu_text, sa, isa, aux, h_rank, two_h_rank);


    prefix_doubling_gpu<uint32_t, osipov_gpu<uint32_t>>(gpu_text, out_sa, osipov);

    return 0;
}
