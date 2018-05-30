/*******************************************************************************
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <saca/dc3.hpp>
#include "test/saca.hpp"

using namespace sacabench::dc3;

TEST(dc3, test) {
    test::saca_corner_cases<dc3>();
}