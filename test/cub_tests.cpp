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

    // Run exclusive prefix sum
    exclusive_sum_64(d_in, d_out, num_items);
}
