/*******************************************************************************
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <saca/dc3_parallel.hpp>
#include "test/saca.hpp"

using namespace sacabench::dc3_parallel;

TEST(dc3_parallel, test) {
    test::saca_corner_cases<dc3_parallel>();
}
