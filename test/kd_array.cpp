/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/kd_array.hpp>

using namespace sacabench::util;

TEST(kd_array, create) {
    { kd_array<size_t, 2> array({10, 10}); }
    { kd_array<size_t, 3> array({10, 10, 10}); }
    { kd_array<size_t, 4> array({10, 10, 10, 10}); }
    { kd_array<size_t, 5> array({10, 10, 10, 10, 10}); }
}

TEST(kd_array, index2d) {
    kd_array<size_t, 2> array({10, 10});

    size_t i = 0;

    for (size_t a = 0; a < 10; ++a) {
        for (size_t b = 0; b < 10; ++b) {
            ASSERT_EQ(i, array.index({a, b}));
            ++i;
        }
    }
}

TEST(kd_array, index3d) {
    kd_array<size_t, 3> array({10, 10, 10});

    size_t i = 0;

    for (size_t a = 0; a < 10; ++a) {
        for (size_t b = 0; b < 10; ++b) {
            for (size_t c = 0; c < 10; ++c) {
                ASSERT_EQ(i, array.index({a, b, c}));
                ++i;
            }
        }
    }
}

TEST(kd_array, index3d_different_sizes) {
    kd_array<size_t, 3> array({5, 10, 15});

    size_t i = 0;

    for (size_t a = 0; a < 5; ++a) {
        for (size_t b = 0; b < 10; ++b) {
            for (size_t c = 0; c < 15; ++c) {
                ASSERT_EQ(i, array.index({a, b, c}));
                ++i;
            }
        }
    }
}
