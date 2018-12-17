/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/alphabet.hpp>
#include <cuda_wrapper_interface.hpp>

TEST(Cub, prefix_sum) {
    // Declare, allocate, and initialize device-accessible pointers for input and output
    size_t   num_items = 7;
    uint64_t d_in[] = {8, 6, 7, 5, 3, 0, 9};
    uint64_t d_out[] = {0, 0, 0, 0, 0, 0, 0};

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;

    // Allocate temporary storage
    void* d_temp_storage = allocate_cuda_buffer(temp_storage_bytes);

    // Run exclusive prefix sum
    exclusive_sum_64(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

    free_cuda_buffer(d_temp_storage);
}
