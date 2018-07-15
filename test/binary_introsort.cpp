/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <random>
#include <util/is_sorted.hpp>
#include <util/sort/binary_introsort.hpp>
#include <util/span.hpp>

using namespace sacabench;
using namespace sacabench::util;
using namespace sacabench::util::sort;

TEST(binary_introsort, sort) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, 1000);

    for (size_t n = 2; n < 100; ++n) {
        for (size_t r = 0; r < 100; ++r) {
            std::vector<size_t> data;

            for (;;) {
                for (size_t i = 0; i < n; ++i) {
                    data.push_back(dist(gen));
                }

                std::vector<size_t> copy = data;
                std::sort(copy.begin(), copy.end());

                bool b = false;

                for (size_t i = 1; i < copy.size(); ++i) {
                    if (copy[i - 1] == copy[i]) {
                        b = true;
                    }
                }

                if (!b) {
                    break;
                } else {
                    data.clear();
                }
            }

            auto s = util::span<size_t>(data);

            binary_introsort::sort(s);

            ASSERT_TRUE(is_sorted(s));
        }
    }
}
