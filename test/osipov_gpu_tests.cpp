#include <gtest/gtest.h>
#include <iostream>
#include <util/alphabet.hpp>
#include <util/container.hpp>
#include <util/saca.hpp>
#include <util/sa_check.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <sstream>
#include <util/alphabet.hpp>
#include <util/bits.hpp>
#include <saca/osipov/osipov_gpu.hpp>
#include "test/saca.hpp"

using namespace sacabench::osipov;
using namespace sacabench;
using namespace sacabench::util;

TEST(OsipovGpu, CornerCases) {
    test::saca_corner_cases<sacabench::osipov::osipov_gpu>();
}
