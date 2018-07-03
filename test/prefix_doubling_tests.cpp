/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"

#include <saca/prefix_doubling.hpp>

using namespace sacabench::prefix_doubling;

TEST(prefix_doubling, naive) {
    test::saca_corner_cases<prefix_doubling>();
}

TEST(prefix_doubling, discarding) {
    test::saca_corner_cases<prefix_doubling_discarding>();
}

TEST(prefix_doubling, quintupling) {
    test::saca_corner_cases<prefix_quintupling_discarding>();
}
