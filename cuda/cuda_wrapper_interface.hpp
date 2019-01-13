#pragma once

#include <cstddef>
#include <cstdint>

// void* allocate_cuda_buffer(size_t size);
void* allocate_managed_cuda_buffer(size_t size);
template<typename T>
T* allocate_managed_cuda_buffer_of(size_t size) {
    return (T*) allocate_managed_cuda_buffer(size * sizeof(T));
}

void free_cuda_buffer(void* ptr);

void exclusive_sum_64(uint64_t* d_in, uint64_t* d_out, size_t num_items);
void exclusive_sum_32(uint32_t* d_in, uint32_t* d_out, size_t num_items);

void inclusive_sum_64(uint64_t* d_in, uint64_t* d_out, size_t num_items);
void inclusive_sum_32(uint32_t* d_in, uint32_t* d_out, size_t num_items);

//extern "C"
void cuda_copy_device_to_device(uint64_t* d_in, uint64_t* d_out,
            size_t num_items);
//extern "C"
void cuda_copy_device_to_device(uint32_t* d_in, uint32_t* d_out,
            size_t num_items);

//extern "C"
void cuda_copy_host_to_device(uint32_t* d_in, uint32_t* d_out,
            size_t num_items);
//extern "C"
void cuda_copy_host_to_device(uint64_t* d_in, uint64_t* d_out,
            size_t num_items);

//extern "C"
void cuda_copy_device_to_host(uint32_t* d_in, uint32_t* d_out,
            size_t num_items);
//extern "C"
void cuda_copy_device_to_host(uint64_t* d_in, uint64_t* d_out,
            size_t num_items);

//extern "C"
void radix_sort_gpu(uint64_t* d_in1, uint64_t* d_in2, uint64_t* aux1,
            uint64_t* aux2, size_t num_items);
//extern "C"
void radix_sort_gpu(uint32_t* d_in1, uint32_t* d_in2, uint32_t* aux1,
            uint32_t* aux2, size_t num_items);
