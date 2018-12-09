#include "cub-1.8.0/cub/cub.cuh"
#include <iostream>
#include <stdlib.h>  


struct Custom_max
{
        
    template <typename T>
    __device__
    CUB_RUNTIME_FUNCTION __forceinline__ 
    T operator()(const T &a, const T &b) const {
        return (b > a) ? b : a;
    }
};
struct Custom_max_without_branching
{
    template <typename T>
    CUB_RUNTIME_FUNCTION __forceinline__ __device__
    T operator()(const T &x, const T &y) const {
        return (x ^ ((x ^ y) & -(x < y)));
    }
};

void SortWithBuffer(int N)
{

    int  *key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
    int  *key_alt_buf;     // e.g., [        ...        ]
    int  *value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
    int  *value_alt_buf;   // e.g., [        ...        ]



    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&key_buf, N*sizeof(int));
    cudaMallocManaged(&key_alt_buf, N*sizeof(int));
    cudaMallocManaged(&value_buf, N*sizeof(int));
    cudaMallocManaged(&value_alt_buf, N*sizeof(int));

    srand(time(NULL));

    for(int index = 0; index < N;++index)
    {
        key_buf[index] = rand() % 1000000;
        value_buf[index] = index;
    }

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int> d_keys(key_buf, key_alt_buf);
    cub::DoubleBuffer<int> d_values(value_buf, value_alt_buf);


    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, N);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, N);

    cudaDeviceSynchronize();

    for(int i = 0; i < N;++i) {
        std::cout<<value_buf[i]<<std::endl;
    }



    cudaFree(key_buf);
    cudaFree(key_alt_buf);

    cudaFree(value_buf);
    cudaFree(value_alt_buf);

}

void SortWithoutBuffer(int N)
{
    //Tatsächliche Werte
    int  *keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
    int  *keys_out;     // e.g., [        ...        ]
    //Indices
    int  *values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
    int  *values_out;   // e.g., [        ...        ]



    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&keys_in, N*sizeof(int));
    cudaMallocManaged(&keys_out, N*sizeof(int));
    cudaMallocManaged(&values_in, N*sizeof(int));
    cudaMallocManaged(&values_out, N*sizeof(int));

    srand(time(NULL));

    for(int index = 0; index < N;++index)
    {
        keys_in[index] = rand() % 1000000;
        values_in[index] = index;
    }

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        keys_in, keys_out, values_in, values_out, N);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
/*
    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total );
    size_t used = total - avail;
    std::cout << "Device memory used: " << used << std::endl;
*/

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        keys_in, keys_out, values_in, values_out, N);

    cudaDeviceSynchronize();
        /*
    for(int i = 0; i < N;++i) {
        std::cout<<"Value: "<<values_out[i]<<", Key:"<<keys_out[i]<<std::endl;
    }*/
    std::cout<<"Value: "<<values_out[N-2]<<", Key:"<<keys_out[N-2]<<std::endl;
    std::cout<<"Value: "<<values_out[N-1]<<", Key:"<<keys_out[N-1]<<std::endl;


    cudaFree(keys_in);
    cudaFree(keys_out);

    cudaFree(values_in);
    cudaFree(values_out);

}

void prefix_sum_cub_inclusive(int N)
{
        //Indices
        int  *values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
        int  *values_out;   // e.g., [        ...        ]
        Custom_max_without_branching max;

    
    
        // Allocate Unified Memory – accessible from CPU or GPU
        cudaMallocManaged(&values_in, N*sizeof(int));
        cudaMallocManaged(&values_out, N*sizeof(int));
    
        srand(time(NULL));
    
        for(int index = 0; index < N;++index)
        {
            values_in[index] = index;//rand() % 1000000;
        }
    
        // Determine temporary device storage requirements
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, values_in, values_out,max, N);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run exclusive prefix sum
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, values_in, values_out,max, N);

        cudaDeviceSynchronize();


        std::cout<<"Last Element: "<<values_out[N-1]<<std::endl;
    
        cudaFree(values_in);
        cudaFree(values_out);
}





int main(void)
{


    //SortWithoutBuffer(10000000);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    prefix_sum_cub_inclusive(100000000);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout<<"Elapsed Time: "<<milliseconds<<" ms"<<std::endl;

    return 0;

}
