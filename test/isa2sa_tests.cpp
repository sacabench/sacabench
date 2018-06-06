/*******************************************************************************
 * test/example_tests.cpp
 *
 * Copyright (C) 2018 PG SACABench
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "../sacabench/util/ISAtoSA.hpp"
#include "../sacabench/util/container.hpp"

using namespace sacabench::util;

TEST(isa2sa, simple_scan_test) {
    container<ssize_t> isa {13, 2, 4, 7, 0, 11, 8, 14, 1, 12, 5, 9, 3, 10, 6};
    container<ssize_t> sa { 4, 8, 1, 12, 2, 10, 14, 3, 6, 11, 13, 5, 9, 0, 7 };
    container<ssize_t> sa_to_be = make_container<ssize_t>(15);
    isa2sa_simple_scan(isa, sa_to_be);
    ASSERT_EQ(sa, sa_to_be);
}

TEST(isa2sa, inplace_stupid_test) {
    container<ssize_t> isa {-14, -3, -5, -8, -1, -12, -9, -15, -2, -13, -6, -10, -4, -11, -7};
    container<ssize_t> sa { 4, 8, 1, 12, 2, 10, 14, 3, 6, 11, 13, 5, 9, 0, 7 };
    isa2sa_inplace(isa);
    ASSERT_EQ(isa, sa);

}
TEST(isa2sa, multiscan_test) {
    container<size_t> isa {13,4,6,10,8,14,12,3,5,9,7,11,2,1,0};
    container<size_t> sa {14,13,12,7,1,8,2,10,4,9,3,11,6,0,5 };
    isa2sa_multiscan(isa);
    ASSERT_EQ(isa, sa);

}
