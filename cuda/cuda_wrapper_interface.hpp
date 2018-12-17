#pragma once

#include <cstddef>
#include <cstdint>

void* allocate_cuda_buffer(size_t size);
void* allocate_managed_cuda_buffer(size_t size);
void free_cuda_buffer(void* ptr);

void exclusive_sum(uint64_t* d_in, uint64_t* d_out, size_t num_items);
void exclusive_sum(uint32_t* d_in, uint32_t* d_out, size_t num_items);

void inclusive_sum(uint64_t* d_in, uint64_t* d_out, size_t num_items);
void inclusive_sum(uint32_t* d_in, uint32_t* d_out, size_t num_items);
