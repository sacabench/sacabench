/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "test/saca.hpp"
#include <gtest/gtest.h>

#include <saca/naive.hpp>

using namespace sacabench::naive;

TEST(prefix_doubling, naive) {
    test::saca_corner_cases<naive>();
}

TEST(prefix_doubling, naive_ips4o) {
    test::saca_corner_cases<naive_ips4o>();
}

TEST(prefix_doubling, naive_ips4o_parallel) {
    test::saca_corner_cases<naive_ips4o_parallel>();
}
