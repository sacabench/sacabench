/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/radixsort_parallel.hpp>
#include <util/container.hpp>

using namespace sacabench::util;

TEST(radixsort_parallel_tests, sort_correct) {

    container<int> test_input = {123, 111, 912, 691};
    container<int> test_output = make_container<int>(4);

    container<int> test_control = {111, 123, 691, 912};

    sort::radixsort_parallel(test_input, test_output);

    ASSERT_EQ(test_output, test_control);
}
