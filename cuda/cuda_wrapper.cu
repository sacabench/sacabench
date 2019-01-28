#include<iostream>
#include<cstdint>
#include<cstddef>

#include <cuda.h>
#include "cub-1.8.0/cub/cub.cuh"
#include "cuda_util.cuh"

#include "cuda_wrapper_interface.hpp"

struct Max_without_branching
{
    template <typename T>
    CUB_RUNTIME_FUNCTION __forceinline__ __device__
    T operator()(const T &x, const T &y) const {
        return (x ^ ((x ^ y) & -(x < y)));
    }
};

/*
    DEPRECATED: Doesn't seem to allocate temporary bytes.
*/
template<typename size_type, typename function>
static void prefix_sum(size_type* d_in,
                size_t num_items,
                function Sum) {
    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    Sum(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);
    //std::cout << "tmp bytes: " << temp_storage_bytes << std::endl;
    // Allocate temporary storage
    d_temp_storage = allocate_managed_cuda_buffer(temp_storage_bytes);

    // Run prefix sum
    Sum(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);

    //cudaDeviceSynchronize();

    free_cuda_buffer(d_temp_storage);
}

template <typename size_type>
static void exclusive_sum_generic(size_type* d_in, size_t num_items) {
    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);
    //std::cout << "tmp bytes: " << temp_storage_bytes << std::endl;
    // Allocate temporary storage
    d_temp_storage = allocate_managed_cuda_buffer(temp_storage_bytes);

    // Run prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);

    //cudaDeviceSynchronize();

    free_cuda_buffer(d_temp_storage);
}

template <typename size_type>
static void inclusive_sum_generic(size_type* d_in, size_t num_items) {
    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);
    //std::cout << "tmp bytes: " << temp_storage_bytes << std::endl;
    // Allocate temporary storage
    d_temp_storage = allocate_managed_cuda_buffer(temp_storage_bytes);

    // Run prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);

    //cudaDeviceSynchronize();

    free_cuda_buffer(d_temp_storage);
}
/*
    Calculates inclusive prefix sum on GPU using the provided CUB Method
*/
template <typename OP, typename size_type>
void inclusive_scan_generic(size_type* d_in, OP op,
            size_t num_items)
{
    //TODO: submit allocated memory instead of allocating new array
    //Indices
    //sa_index  *values_out;   // e.g., [        ...        ]

    // Allocate Unified Memory – accessible from CPU or GPU
    //cudaMallocManaged(&values_out, n*sizeof(sa_index));

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in,
                d_in, op, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in,
                d_in, op, num_items);

    //cudaDeviceSynchronize();

    free_cuda_buffer(d_temp_storage);
    //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(array,values_out,n);
    /*
    cudaMemcpy(array, values_out, n*sizeof(sa_index), cudaMemcpyDeviceToDevice);

    cudaFree(values_out);*/
}




void* allocate_cuda_buffer(size_t size) {
    void* ret = nullptr;
    cuda_check(cudaMalloc(&ret, size), "cudaMalloc");
    return ret;
}

void* allocate_managed_cuda_buffer(size_t size) {
    void* ret = nullptr;
    cuda_check(cudaMallocManaged(&ret, size), "cudaMallocManaged");
    return ret;
}

void free_cuda_buffer(void* ptr) {
    cuda_check(cudaFree(ptr), "cudaFree");
}

template <typename size_type>
bool check_cuda_memory(size_t bytes_needed) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    bool sufficient_memory;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    sufficient_memory = free_bytes > bytes_needed ? true : false;
    return sufficient_memory;
}

size_t check_cuda_memory_free() {
    //size_t bytes_needed = sizeof(size_type)*num_arrays*num_items + 0.001*sizeof(size_type)*num_items*2;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

bool check_cuda_memory_32(size_t bytes_needed) {
    return check_cuda_memory<uint32_t>(bytes_needed);
}

bool check_cuda_memory_64(size_t bytes_needed) {
    return check_cuda_memory<uint64_t>(bytes_needed);
}

void exclusive_sum(uint64_t* d_in, size_t num_items) {
    exclusive_sum_generic(d_in, num_items);
}
void exclusive_sum(uint32_t* d_in, size_t num_items) {
    exclusive_sum_generic(d_in, num_items);
}

void inclusive_sum(uint64_t* d_in, size_t num_items) {
    inclusive_sum_generic(d_in, num_items);
}
void inclusive_sum(uint32_t* d_in, size_t num_items) {
    inclusive_sum_generic(d_in, num_items);
}

void inclusive_max(uint32_t* d_in, size_t size) {
    inclusive_scan_generic<Max_without_branching, uint32_t>(d_in
                    , Max_without_branching(), size);
}

void inclusive_max(uint64_t* d_in, size_t size) {
    inclusive_scan_generic<Max_without_branching, uint64_t>(d_in, Max_without_branching(), size);
}
/*
void exclusive_sum_64(uint64_t* d_in, uint64_t* d_out, size_t num_items) {
    prefix_sum(d_in, d_out, num_items, [](auto... params) {
        cuda_check(cub::DeviceScan::ExclusiveSum(params...), "ExclusiveSum");
    });
}

void inclusive_sum_64(uint64_t* d_in, uint64_t* d_out, size_t num_items) {
    prefix_sum(d_in, d_out, num_items, [](auto... params) {
        cuda_check(cub::DeviceScan::InclusiveSum(params...), "InclusiveSum");
    });
}
void exclusive_sum_32(uint32_t* d_in, uint32_t* d_out, size_t num_items) {
    prefix_sum(d_in, d_out, num_items, [](auto... params) {
        cuda_check(cub::DeviceScan::ExclusiveSum(params...), "ExclusiveSum");
    });
}

void inclusive_sum_32(uint32_t* d_in, uint32_t* d_out, size_t num_items) {
    prefix_sum(d_in, d_out, num_items, [](auto... params) {
        cuda_check(cub::DeviceScan::InclusiveSum(params...), "InclusiveSum");
    });
}
*/

void cuda_copy_device_to_device(uint32_t* d_in, uint32_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint32_t),
            cudaMemcpyDeviceToDevice));

}

void cuda_copy_device_to_device(uint64_t* d_in, uint64_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint64_t),
            cudaMemcpyDeviceToDevice));
}

void cuda_copy_host_to_device(uint32_t* d_in, uint32_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint32_t),
            cudaMemcpyHostToDevice));
}

void cuda_copy_host_to_device(uint64_t* d_in, uint64_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint64_t),
            cudaMemcpyHostToDevice));
}

void cuda_copy_device_to_host(uint32_t* d_in, uint32_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
}

void cuda_copy_device_to_host(uint64_t* d_in, uint64_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint64_t),
            cudaMemcpyDeviceToHost));
}

template <typename size_type>
void radix_sort_cub(size_type* d_in1, size_type* d_in2, size_type* aux1,
            size_type* aux2, size_t num_items) {
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cuda_check(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_in1,
                aux1, d_in2, aux2, num_items));
    d_temp_storage = allocate_cuda_buffer(temp_storage_bytes);
    cuda_check(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_in1,
                aux1, d_in2, aux2, num_items));
    //cuda_check(cudaDeviceSynchronize());
    free_cuda_buffer(d_temp_storage);
}

void radix_sort_gpu(uint32_t* d_in1, uint32_t* d_in2, uint32_t* aux1,
            uint32_t* aux2, size_t num_items) {
    radix_sort_cub(d_in1, d_in2, aux1, aux2, num_items);
    /*cuda_copy_device_to_device(d_in1, aux1, num_items);
    cuda_copy_device_to_device(d_in2, aux2, num_items);*/
}

void radix_sort_gpu(uint64_t* d_in1, uint64_t* d_in2, uint64_t* aux1,
            uint64_t* aux2, size_t num_items) {
    radix_sort_cub(d_in1, d_in2, aux1, aux2, num_items);
    /*cuda_copy_device_to_device(d_in1, aux1, num_items);
    cuda_copy_device_to_device(d_in2, aux2, num_items);*/
}
