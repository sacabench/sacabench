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

struct Compare_first_char
{
public:  
    Compare_first_char(int* _text) : text(_text) {};
    int* text;
    template <typename index>
    CUB_RUNTIME_FUNCTION __forceinline__ __device__
    bool operator()(const index &x, const index &y) const {
        return text[x]<text[y];
    }
};

//Quick and dirty version, which packs four chars in one int
void word_packing(const char* chars, int* result, int n) {

    typedef unsigned char u8; 
    for(int i = 0; i<n-3 ;++i) {
        result[i] = ((u8)chars[i] << 24) | ((u8)chars[i+1] << 16) | ((u8)chars[i+2] << 8) | (u8)chars[i+3];
    }
    result[n-3] = ((u8)chars[n-3] << 24) | ((u8)chars[n-2] << 16) | ((u8)chars[n-1] << 8);
    result[n-2] = ((u8)chars[n-2] << 24) | ((u8)chars[n-1] << 16);
    result[n-1] = ((u8)chars[n-1] << 24);

}

    __global__
    static void initialize_sa_gpu(int n, int*  sa) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i+=stride) {
            sa[i] = i;
        }

    }

    __global__
    static void copy_to_array(int* in, int* out, int n) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i+=stride) {
            in[i] = out[i];
        }

    }


    static void inital_sorting(int* text, int* sa, int* aux, int n) {


     //Actual values
    int  *keys_out;     // e.g., [        ...        ]


    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&keys_out, n*sizeof(int));


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

    copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(sa,aux,n);



    }

    template <typename OP>
void prefix_sum_cub_inclusive(int* array, OP op, int n)
{
        //Indices
        int  *values_out;   // e.g., [        ...        ]

    
    
        // Allocate Unified Memory – accessible from CPU or GPU
        cudaMallocManaged(&values_out, n*sizeof(int));
    
        // Determine temporary device storage requirements
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, array, values_out,op, n);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run exclusive prefix sum
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, array, values_out,op, n);

        cudaDeviceSynchronize();

        copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(array,values_out,n);


}
    template <typename Comp>
    __global__
    void fill_aux_for_isa(int* sa, int* aux, int n, Comp comp) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        if(index == 0) {
            aux[0]=0;
        }

        for (int i = index+1; i < n; i+=stride) {
            aux[i] = i * (comp(sa[i - 1], sa[i]) != 0);
        }
    }
    __global__
    void scatter_to_isa(int* isa, int* aux,int* sa, int n) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        //Maybe TODO: Avoid Bank Conflicts
        for (int i = index; i < n; i+=stride) {
            isa[sa[i]]=aux[i];
        }
    }

    template <typename Comp>
    void initialize_isa(int* isa, int* sa, int* aux, int n, Comp comp) {

        fill_aux_for_isa<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(sa,aux,n, comp);

        cudaDeviceSynchronize();

        Max_without_branching max;

        prefix_sum_cub_inclusive(aux,max, n);

        cudaDeviceSynchronize();

        scatter_to_isa<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(isa,aux,sa,n);

    }


    static void prefix_doubling_gpu(int* gpu_text, int* out_sa, int n) {
        
        int* sa;
        //Wofür??
        int* isa_container;
        int* aux_container;

        cudaMallocManaged(&sa, n*sizeof(int));
        cudaMallocManaged(&isa_container, n*sizeof(int));
        cudaMallocManaged(&aux_container, n*sizeof(int));
        cudaDeviceSynchronize();


        initialize_sa_gpu<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(n, sa);
        cudaDeviceSynchronize();

        //TODO Sort by four chars by wordpacking
        inital_sorting(gpu_text, sa, aux_container, n);
        cudaDeviceSynchronize();

        std::cout<<"SA: ";
        for(int i = 0; i<n; ++i) {
            std::cout<<sa[i]<<", ";
        }
        std::cout<<std::endl;

        Compare_first_char comp(gpu_text);
        initialize_isa(out_sa, sa, aux_container, n, comp);
        cudaDeviceSynchronize();

        std::cout<<std::endl;
        std::cout<<"ISA: ";
        for(int i = 0 ; i< n ; ++i) {
            std::cout<<out_sa[sa[i]]<<", ";
        }
        std::cout<<std::endl;

        int h = 4;

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
    int n = text_str.size()+1;
    std::cout<<"n: "<<n<<std::endl;


    int* packed_text;
    packed_text = (int *) malloc(n*sizeof(int));
    word_packing(text, packed_text, n);

    int* gpu_text;
    int* out_sa;
    cudaMallocManaged(&gpu_text, n*sizeof(int));
    //Copy text to GPU
    memset(gpu_text, 0, n*sizeof(int));
    cudaMemcpy(gpu_text, packed_text, n*sizeof(int), cudaMemcpyHostToDevice);  
    cudaMallocManaged(&out_sa, n*sizeof(int));

    cudaDeviceSynchronize();


    prefix_doubling_gpu(gpu_text, out_sa, n);
    return 0;
}
