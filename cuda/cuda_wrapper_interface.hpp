/*******************************************************************************
 * Copyright (C) 2019 Marvin Löbel <loebel.marvin@gmail.com> 
 * Copyright (C) 2019 Oliver Magiera <oliver.magiera@tu-dortmund.de>
 * Copyright (C) 2019 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

void* allocate_cuda_buffer(size_t size);
void* allocate_managed_cuda_buffer(size_t size);
template<typename T>
T* allocate_managed_cuda_buffer_of(size_t size) {
    return (T*) allocate_managed_cuda_buffer(size * sizeof(T));
}

void free_cuda_buffer(void* ptr);

bool check_cuda_memory_32(size_t bytes_needed);
bool check_cuda_memory_64(size_t bytes_needed);

size_t check_cuda_memory_free();

void exclusive_sum(uint64_t* d_in, size_t num_items);
void exclusive_sum(uint32_t* d_in, size_t num_items);

void inclusive_sum(uint64_t* d_in, size_t num_items);
void inclusive_sum(uint32_t* d_in, size_t num_items);

void inclusive_max(uint32_t* d_in, size_t size);
void inclusive_max(uint64_t* d_in, size_t size);

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
