/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "test/saca.hpp"
#include <gtest/gtest.h>
#include <random>
#include <saca/deep_shallow/blind/sort.hpp>
#include <saca/deep_shallow/blind/trie.hpp>
#include <saca/deep_shallow/saca.hpp>
#include <util/alphabet.hpp>
#include <util/container.hpp>
#include <util/sa_check.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

using namespace sacabench;
using u_char = sacabench::util::character;
using ds = sacabench::deep_shallow::serial_big_buckets;

TEST(big_buckets_deep_shallow, corner_cases) { test::saca_corner_cases<ds>(); }
