/*******************************************************************************
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/


#include <gtest/gtest.h>
#include <saca/dc7.hpp>
#include "test/saca.hpp"

using namespace sacabench::dc7;

TEST(dc7, test) {
    test::saca_corner_cases<dc7>();
}
