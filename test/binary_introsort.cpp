/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <random>
#include <saca/deep_shallow/log.hpp>
#include <util/is_sorted.hpp>
#include <util/sort/binary_introsort.hpp>
#include <util/sort/insertionsort.hpp>
#include <util/span.hpp>

using sacabench::deep_shallow::duration;
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

TEST(binary_introsort, quickly) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, 1000);

    for (size_t n = 2; n < 100; ++n) {
        std::cout << "Länge " << n << ": " << std::endl;

        double d1 = 0, d2 = 0, d3 = 0;

        constexpr size_t REPS = 100;

        for (size_t r = 0; r < REPS; ++r) {
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
            std::vector<size_t> data2 = data;
            auto s2 = util::span<size_t>(data);
            std::vector<size_t> data3 = data;
            auto s3 = util::span<size_t>(data);

            d1 += duration([&]() { insertion_sort(s); });
            d2 += duration([&]() { insertion_sort2(s2); });
            d3 += duration([&]() { insertion_sort_hybrid(s3); });
        }

        std::cout << "Rosa: " << (d1 / double(REPS))
                  << " vs Meiner: " << (d2 / double(REPS))
                  << " vs Hybrid: " << (d3 / double(REPS)) << std::endl;
    }
}
