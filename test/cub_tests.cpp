/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/alphabet.hpp>
#include <cuda_wrapper_interface.hpp>
#include <check_for_gpu_interface.hpp>

TEST(Cub, prefix_sum) {

    if(cuda_GPU_available()) {
        // Declare, allocate, and initialize device-accessible pointers for input and output
        size_t   num_items = 7;
        uint64_t* d_in = allocate_managed_cuda_buffer_of<uint64_t>(num_items);
        d_in[0] = 8;
        d_in[1] = 6;
        /*
        uint64_t d_in[] = {8, 6, 7, 5, 3, 0, 9};
        uint64_t d_out[] = {0, 0, 0, 0, 0, 0, 0}

        */
        // Run exclusive prefix sum
        exclusive_sum(d_in, num_items);

        free_cuda_buffer(d_in);
        }
}
