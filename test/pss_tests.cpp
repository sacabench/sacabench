/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *                    David Piper <david.piper@tu-dortmund.de>
 *                    Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/pss.hpp>
#include <util/sort/std_sort.hpp>
#include <util/container.hpp>
#include <parallel/algorithm>

using namespace sacabench::util;

class pss_tests : public ::testing::Test {
    protected:
        static void SetUpTestCase() {
            short_inst = {3,1,9,6,4,2,7,8,5};
            short_control_inst = container<int>(short_inst);
            sort::std_sort(short_control_inst, std::less<int>());
            long_inst = {3,1,9,6,4,2,7,8,5};
            long_control_inst = container<int>(long_inst);
            sort::std_sort(long_control_inst, std::less<int>());
        }

        void SetUp() override {
            short_test_inst = container<int>(short_inst);
            long_test_inst = container<int>(long_inst);
        }

        static container<int> short_inst;
        static container<int> short_control_inst;
        static container<int> long_inst;
        static container<int> long_control_inst;
        container<int> short_test_inst;
        container<int> long_test_inst;
};

container<int> pss_tests::short_inst;
container<int> pss_tests::long_inst;
container<int> pss_tests::short_control_inst;
container<int> pss_tests::long_control_inst;

TEST_F(pss_tests, intel_sort_correct) {
    sort::parallel_stable<int>(short_test_inst, std::less<int>());
    ASSERT_EQ(short_test_inst, short_control_inst);
}

TEST_F(pss_tests, std_sort_correct) {
    __gnu_parallel::stable_sort(std::begin(short_test_inst), std::end(short_test_inst), std::less<int>());
    ASSERT_EQ(short_test_inst, short_control_inst);
}

TEST_F(pss_tests, intel_sort_long) {
    sort::parallel_stable<int>(short_test_inst, std::less<int>());
    ASSERT_EQ(short_test_inst, short_control_inst);
}

TEST_F(pss_tests, std_sort_long) {
    __gnu_parallel::stable_sort(std::begin(long_test_inst), std::end(long_test_inst), std::less<int>());
    ASSERT_EQ(long_test_inst, long_control_inst);
}
