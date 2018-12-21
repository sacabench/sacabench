#include <string>
#include <iostream>
#include "cub-1.8.0/cub/cub.cuh"


#define NUM_BLOCKS 2
#define NUM_THREADS_PER_BLOCK 4


struct Max_without_branching
{
    template <typename T>
    CUB_RUNTIME_FUNCTION __forceinline__ __device__
    T operator()(const T &x, const T &y) const {
        return (x ^ ((x ^ y) & -(x < y)));
    }
};

struct Compare_four_chars
{
public:  
    Compare_four_chars(size_t* _text) : text(_text) {};
    size_t* text;
    template <typename index>
    CUB_RUNTIME_FUNCTION __forceinline__ __device__
    bool operator()(const index &x, const index &y) const {
        return text[x]<text[y];
    }
};

//Quick and dirty version, which packs four chars in one size_t
void word_packing(const char* chars, size_t* result, size_t n) {

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
__global__
static void initialize_sa_gpu(size_t n, size_t*  sa) {

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
static void inital_sorting(size_t* text, size_t* sa, size_t* aux, size_t n) {

     //Actual values
    size_t  *keys_out;     // e.g., [        ...        ]


    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&keys_out, n*sizeof(size_t));


    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        text, keys_out, sa, aux, n);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);


    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        text, keys_out, sa, aux, n);


    cudaDeviceSynchronize();

    //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(sa,aux,n);

    cudaMemcpy(sa, aux, n*sizeof(size_t), cudaMemcpyDeviceToDevice);
}


/*
    Calculates inclusive prefix sum on GPU using the provided CUB Method
*/
template <typename OP>
void prefix_sum_cub_inclusive(size_t* array, OP op, size_t n)
{
    //Indices
    size_t  *values_out;   // e.g., [        ...        ]

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&values_out, n*sizeof(size_t));

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

    cudaMemcpy(array, values_out, n*sizeof(size_t), cudaMemcpyDeviceToDevice);


}
/*
    Auxiliary function for initializing ISA
    Computes inital aux array, with index if own value other to predecessor, else 0
*/
template <typename Comp>
__global__
void fill_aux_for_isa(size_t* sa, size_t* aux, size_t n, Comp comp) {

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
__global__
void scatter_to_isa(size_t* isa, size_t* aux,size_t* sa, size_t n) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    //Maybe TODO: Avoid Bank Conflicts
    for (size_t i = index; i < n; i+=stride) {
        isa[sa[i]]=aux[i];
    }
}

__global__
void update_ranks_build_aux(size_t* two_h_ranks, size_t* aux, size_t n) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if(index == 0) {
        aux[0]=0;
    }

    for (size_t i = index+1; i < n; i+=stride) {
        aux[i] = (two_h_ranks[i-1]!=two_h_ranks[i]) * i;
    }
}

__global__
void update_ranks_build_aux_tilde(size_t* two_h_ranks, size_t* h_ranks, size_t* aux, size_t n) {

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

void update_ranks(size_t* two_h_ranks, size_t* h_ranks, size_t* aux, size_t n) {

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
template <typename Comp>
void initialize_isa(size_t* isa, size_t* sa, size_t* aux, size_t n, Comp comp) {

    fill_aux_for_isa<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(sa,aux,n, comp);

    cudaDeviceSynchronize();

    Max_without_branching max;

    prefix_sum_cub_inclusive(aux,max, n);

    cudaDeviceSynchronize();

    scatter_to_isa<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(isa,aux,sa,n);

}

void sort_tuples(size_t* tuple_index, size_t* two_h_ranks, size_t* aux1, size_t* aux2 ,size_t n) {


     // Determine temporary device storage requirements
     void     *d_temp_storage = NULL;
     size_t   temp_storage_bytes = 0;
 
     cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
         two_h_ranks, aux1, tuple_index, aux2, n);
     // Allocate temporary storage
     cudaMalloc(&d_temp_storage, temp_storage_bytes);
 
     // Run sorting operation
     cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        two_h_ranks, aux1, tuple_index, aux2, n);
 
     cudaDeviceSynchronize();

     //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(tuple_index,aux1,n);
     cudaMemcpy(tuple_index, aux1, n*sizeof(size_t), cudaMemcpyDeviceToDevice);
     cudaMemcpy(two_h_ranks, aux2, n*sizeof(size_t), cudaMemcpyDeviceToDevice);

     //cudaDeviceSynchronize();
     //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(two_h_ranks,aux2,n);


     
}

static void prefix_doubling_gpu(size_t* gpu_text, size_t* out_sa, size_t n) {
    
    //additional arrays
    size_t* sa;
    size_t* isa_container;
    size_t* aux_container;

    //allocate additional arrays directly on GPU
    cudaMallocManaged(&sa, n*sizeof(size_t));
    cudaMallocManaged(&isa_container, n*sizeof(size_t));
    cudaMallocManaged(&aux_container, n*sizeof(size_t));
    cudaDeviceSynchronize();

    //Fill SA 
    initialize_sa_gpu<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(n, sa);
    cudaDeviceSynchronize();

    //Sort by four characters
    inital_sorting(gpu_text, sa, aux_container, n);
    cudaDeviceSynchronize();

    std::cout<<"SA: ";
    for(size_t i = 0; i<n; ++i) {
        std::cout<<sa[i]<<", ";
    }
    std::cout<<std::endl;

    //Init ISA with group numbers according to initial sorting
    Compare_four_chars comp(gpu_text);
    initialize_isa(out_sa, sa, aux_container, n, comp);
    cudaDeviceSynchronize();

    std::cout<<std::endl;
    std::cout<<"ISA: ";
    for(size_t i = 0 ; i< n ; ++i) {
        std::cout<<out_sa[sa[i]]<<", ";
    }
    std::cout<<std::endl;
    
    //size_t h = 4;

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


    size_t* packed_text;
    packed_text = (size_t *) malloc(n*sizeof(size_t));
    //Pack text, so you can compare four chars at once
    word_packing(text, packed_text, n);

    //GPU arrays
    size_t* gpu_text;
    size_t* out_sa;
    cudaMallocManaged(&gpu_text, n*sizeof(size_t));
    //Copy text to GPU
    memset(gpu_text, 0, n*sizeof(size_t));
    cudaMemcpy(gpu_text, packed_text, n*sizeof(size_t), cudaMemcpyHostToDevice);  
    cudaMallocManaged(&out_sa, n*sizeof(size_t));
    cudaDeviceSynchronize();


    prefix_doubling_gpu(gpu_text, out_sa, n);

    return 0;
}
