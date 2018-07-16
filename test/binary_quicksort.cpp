/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/binary_quicksort.hpp>
#include <util/span.hpp>
#include <util/is_sorted.hpp>
#include <random>

using namespace sacabench;
using namespace sacabench::util;
using namespace sacabench::util::sort;

TEST(binary_quicksort, partition) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, -1ul);

    for(size_t n = 1; n < 100; ++n) {
        for(size_t r = 0; r < 100; ++r) {
            std::vector<size_t> data;

            for(size_t i = 0; i < n;++i) {
                data.push_back(dist(gen));
            }

            auto s = util::span<size_t>(data);
            const auto pivot = data[0];

            // std::cout << s << std::endl;
            const size_t part = binary_quicksort::partition(s, pivot);
            // std::cout << s.slice(0, part) << s.slice(part) << std::endl;

            for(const size_t a : s.slice(0, part)) {
                ASSERT_LT(a, pivot);
            }

            for(const size_t b : s.slice(part)) {
                ASSERT_GE(b, pivot);
            }

            // std::cout << "####################" << std::endl;
        }
    }
}
