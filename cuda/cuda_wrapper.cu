#include<iostream>
#include<cstdint>
#include<cstddef>

#include <cuda.h>
#include "cub-1.8.0/cub/cub.cuh"
#include "cuda_util.cuh"

#include "cuda_wrapper_interface.hpp"

template<typename size_type, typename function>
static void prefix_sum(size_type* d_in,
                size_type* d_out,
                size_t num_items,
                function Sum) {
    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    Sum(nullptr, temp_storage_bytes, d_in, d_out, num_items);

    // Allocate temporary storage
    void* d_temp_storage = allocate_managed_cuda_buffer(temp_storage_bytes);

    // Run prefix sum
    Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

    cudaDeviceSynchronize();

    free_cuda_buffer(d_temp_storage);
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

void exclusive_sum(uint64_t* d_in, uint64_t* d_out, size_t num_items) {
    prefix_sum(d_in, d_out, num_items, [](auto... params) {
        cuda_check(cub::DeviceScan::ExclusiveSum(params...), "ExclusiveSum");
    });
}

void inclusive_sum(uint64_t* d_in, uint64_t* d_out, size_t num_items) {
    prefix_sum(d_in, d_out, num_items, [](auto... params) {
        cuda_check(cub::DeviceScan::InclusiveSum(params...), "InclusiveSum");
    });
}

void exclusive_sum(uint32_t* d_in, uint32_t* d_out, size_t num_items) {
    prefix_sum(d_in, d_out, num_items, [](auto... params) {
        cuda_check(cub::DeviceScan::ExclusiveSum(params...), "ExclusiveSum");
    });
}

void inclusive_sum(uint32_t* d_in, uint32_t* d_out, size_t num_items) {
    prefix_sum(d_in, d_out, num_items, [](auto... params) {
        cuda_check(cub::DeviceScan::InclusiveSum(params...), "InclusiveSum");
    });
}

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
    cuda_check(cudaDeviceSynchronize());
}

void radix_sort_gpu_32(uint32_t* d_in1, uint32_t* d_in2, uint32_t* aux1,
            uint32_t* aux2, size_t num_items) {
    radix_sort_cub(d_in1, d_in2, aux1, aux2, num_items);
    /*cuda_copy_device_to_device(d_in1, aux1, num_items);
    cuda_copy_device_to_device(d_in2, aux2, num_items);*/
}

void radix_sort_gpu_64(uint64_t* d_in1, uint64_t* d_in2, uint64_t* aux1,
            uint64_t* aux2, size_t num_items) {
    radix_sort_cub(d_in1, d_in2, aux1, aux2, num_items);
    /*cuda_copy_device_to_device(d_in1, aux1, num_items);
    cuda_copy_device_to_device(d_in2, aux2, num_items);*/
}
