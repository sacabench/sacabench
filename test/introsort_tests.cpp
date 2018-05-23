/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/introsort.hpp>
#include <util/container.hpp>

using namespace sacabench::util;

TEST(IntroSort, sort_correct) {
    container<int> test_inst = {3,1,9,6,4,2,7,8,5};
    container<int> control_inst = {1,2,3,4,5,6,7,8,9};
    introsort<int>(test_inst);
    ASSERT_EQ(test_inst, control_inst);
}
