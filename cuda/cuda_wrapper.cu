#include<iostream>
#include<cstdint>
#include<cstddef>

#include <cuda.h>
#include "../external/cub/cub/cub.cuh"

#include "cuda_wrapper_interface.hpp"

void cuda_check(bool v) {
    if (!v) {
        std::cerr << "cuda malloc error" << std::endl;
    }
    std::abort();
}

void* allocate_cuda_buffer(size_t size) {
    void* ret = nullptr;
    cuda_check(cudaMalloc(&ret, size));
    return ret;
}

void exclusive_sum_64(void* d_temp_storage,
                      size_t& temp_storage_bytes,
                      uint64_t* d_in,
                      uint64_t* d_out,
                      size_t num_items)
{
    cuda_check(
        cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      d_in,
                                      d_out,
                                      num_items));
}
