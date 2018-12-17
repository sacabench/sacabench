#pragma once

#include <cstddef>
#include <cstdint>

void* allocate_cuda_buffer(size_t size);
void free_cuda_buffer(void* ptr);

void exclusive_sum_64(void* d_temp_storage,
                      size_t& temp_storage_bytes,
                      uint64_t* d_in,
                      uint64_t* d_out,
                      size_t num_items);
