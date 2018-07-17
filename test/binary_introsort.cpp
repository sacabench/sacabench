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
#include <util/sort/ternary_quicksort.hpp>
#include <util/sort/introsort.hpp>
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
        for (size_t r = 0; r < 10; ++r) {
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

            // std::cout << s << std::endl;

            binary_introsort::sort(s);

            // std::cout << s << std::endl;

            ASSERT_TRUE(is_sorted(s));
        }
    }
}

TEST(binary_introsort, versus_ternary) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, 1000);

    for (size_t n = 2; n < 100; ++n) {
        std::cout << "Länge " << n << ": " << std::endl;

        double d0 = 0, d1 = 0, d2 = 0, d3 = 0;

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

            std::vector<size_t> data4 = data;
            auto s4 = util::span<size_t>(data);

            d0 += duration([&]() { std::sort(s4.begin(), s4.end()); });
            d1 += duration([&]() { binary_introsort::sort(s); });
            d2 += duration([&]() { ternary_quicksort::ternary_quicksort(s2, std::less<size_t>()); });
            d3 += duration([&]() { introsort(s3); });

            ASSERT_TRUE(is_sorted(s));
            ASSERT_TRUE(is_sorted(s2));
            ASSERT_TRUE(is_sorted(s3));
        }

        std::cout << "Dummy: " << (d0 / double(REPS))
                  << " vs Meiner: " << (d1 / double(REPS))
                  << " vs Hermann: " << (d2 / double(REPS))
                  << " vs Ollis: " << (d3 / double(REPS)) << std::endl;
    }
}

// TEST(binary_introsort, versus_ternary_big) {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<size_t> dist(0, 1000000);
//
//     for (size_t n = 2; n < 100000; n *= 2) {
//         std::cout << "Länge " << n << ": " << std::endl;
//
//         double d0 = 0, d1 = 0, d2 = 0, d3 = 0;
//
//         constexpr size_t REPS = 10;
//
//         for (size_t r = 0; r < REPS; ++r) {
//             std::vector<size_t> data;
//
//             for (size_t i = 0; i < n; ++i) {
//                 data.push_back(dist(gen));
//             }
//
//             {
//                 std::vector<size_t> copy = data;
//                 std::sort(copy.begin(), copy.end());
//
//                 for (size_t i = 1; i < copy.size(); ++i) {
//                     while (copy[i - 1] == copy[i]) {
//                         copy[i - 1] = dist(gen);
//                         std::cout << copy[i-1] << std::endl;
//                     }
//                     data[i-1] = copy[i-1];
//                 }
//             }
//
//             std::cout << "done generating random sequence" << std::endl;
//
//             auto s = util::span<size_t>(data);
//             std::vector<size_t> data2 = data;
//             auto s2 = util::span<size_t>(data);
//             std::vector<size_t> data3 = data;
//             auto s3 = util::span<size_t>(data);
//             std::vector<size_t> data4 = data;
//             auto s4 = util::span<size_t>(data);
//
//             d0 += duration([&]() { std::sort(s4.begin(), s4.end()); });
//             d1 += duration([&]() { introsort(s3); });
//             d2 += duration([&]() { ternary_quicksort::ternary_quicksort(s2, std::less<size_t>()); });
//             d3 += duration([&]() { binary_introsort::sort(s); });
//
//             ASSERT_TRUE(is_sorted(s));
//             ASSERT_TRUE(is_sorted(s2));
//             ASSERT_TRUE(is_sorted(s3));
//         }
//
//         std::cout << "Dummy: " << (d0 / double(REPS))
//                   << " vs Ollis: " << (d1 / double(REPS))
//                   << " vs Hermann: " << (d2 / double(REPS))
//                   << " vs Meiner: " << (d3 / double(REPS)) << std::endl;
//     }
// }
