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

TEST(prefix_doubling, discarding_2) {
    test::saca_corner_cases<prefix_discarding_2>();
}

TEST(prefix_doubling, discarding_4) {
    test::saca_corner_cases<prefix_discarding_4>();
}
