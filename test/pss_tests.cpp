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
            long TEST_SIZE = 100'000'000l;
            short_inst = {3,1,9,6,4,2,7,8,5};
            short_control_inst = container<int>(short_inst);
            sort::std_stable_sort(short_control_inst, std::less<int>());
            long_inst = container<int>(TEST_SIZE);
            for (long i = 0; i < TEST_SIZE; ++i) {
                long_inst[i] = rand();
            }
            long_control_inst = container<int>(long_inst);
            sort::std_stable_sort(long_control_inst, std::less<int>());
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

TEST_F(pss_tests, seq_sort_correct) {
    sort::std_stable_sort<int>(short_test_inst, std::less<int>());
    ASSERT_EQ(short_test_inst, short_control_inst);
}

TEST_F(pss_tests, intel_sort_correct) {
    sort::parallel_stable<int>(short_test_inst, std::less<int>());
    ASSERT_EQ(short_test_inst, short_control_inst);
}

TEST_F(pss_tests, std_sort_correct) {
    sort::std_par_stable_sort<int>(short_test_inst, std::less<int>());
    ASSERT_EQ(short_test_inst, short_control_inst);
}

/*
TEST_F(pss_tests, seq_sort_long) {
    sort::std_stable_sort<int>(long_test_inst, std::less<int>());
    ASSERT_EQ(long_test_inst, long_control_inst);
}

TEST_F(pss_tests, intel_sort_long) {
    sort::parallel_stable<int>(long_test_inst, std::less<int>());
    ASSERT_EQ(long_test_inst, long_control_inst);
}

TEST_F(pss_tests, std_sort_long) {
    sort::std_par_stable_sort<int>(long_test_inst, std::less<int>());
    ASSERT_EQ(long_test_inst, long_control_inst);
}
*/
