/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/heapsort.hpp>
#include <util/container.hpp>

using namespace sacabench::util;

TEST(HeapSort, sort_correct) {
    container<int> test_inst = {3,1,9,6,4,2,7,8,5};
    container<int> control_inst = {1,2,3,4,5,6,7,8,9};
    //heapsort(span(test_inst));
    sort::heapsort<int>(test_inst);
    ASSERT_EQ(test_inst, control_inst);
}