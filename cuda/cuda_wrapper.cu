#include<iostream>
#include<cstdint>
#include<cstddef>

#include <cuda.h>
#include "../external/cub/cub/cub.cuh"

#include "cuda_wrapper_interface.hpp"

void cuda_check(bool v, char const* reason) {
    if (!v) {
        std::cerr << "cuda error: " << reason << std::endl;
    }
    std::abort();
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

void exclusive_sum_64(uint64_t* d_in,
                      uint64_t* d_out,
                      size_t num_items)
{
    //using cub::DeviceScan::ExclusiveSum;

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cuda_check(
        cub::DeviceScan::ExclusiveSum(nullptr,
                                      temp_storage_bytes,
                                      d_in,
                                      d_out,
                                      num_items), "ExclusiveSum");

    // Allocate temporary storage
    void* d_temp_storage = allocate_managed_cuda_buffer(temp_storage_bytes);

    // Run prefix sum
    cuda_check(
        cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      d_in,
                                      d_out,
                                      num_items), "ExclusiveSum");

    cudaDeviceSynchronize();

    free_cuda_buffer(d_temp_storage);
}
