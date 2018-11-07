/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *                    David Piper <david.piper@tu-dortmund.de>
 *                    Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/pss.hpp>
#include <util/container.hpp>

using namespace sacabench::util;

TEST(pss_tests, sort_correct) {
    container<int> test_inst = {3,1,9,6,4,2,7,8,5};
    container<int> control_inst = {1,2,3,4,5,6,7,8,9};
    sort::pss_sort<int>(test_inst, std::less<int>());
    ASSERT_EQ(test_inst, control_inst);
}