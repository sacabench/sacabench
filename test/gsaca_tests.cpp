/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"
#include <saca/gsaca/gsaca.hpp>
#include <saca/gsaca/gsaca_new.hpp>
#include <saca/gsaca/gsaca_parallel.hpp>

/*
TEST(gsaca, test_corner_cases) {
    test::saca_corner_cases<sacabench::gsaca::gsaca>();
}

TEST(gsaca_new, test_corner_cases) {
    test::saca_corner_cases<sacabench::gsaca::gsaca_new>();
}
*/

TEST(gsaca_parallel, test_corner_cases) {
    test::saca_corner_cases<sacabench::gsaca::gsaca_parallel>();
}