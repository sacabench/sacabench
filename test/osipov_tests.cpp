#include <gtest/gtest.h>
#include <iostream>
#include <util/alphabet.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <saca/osipov/osipov_parallel.hpp>
#include "test/saca.hpp"

using namespace sacabench::osipov;
using namespace sacabench;

TEST(Osipov, CornerCases) {
    test::saca_corner_cases<sacabench::osipov::osipov_parallel>();
}

TEST(OsipovWp, CornerCases) {
    test::saca_corner_cases<sacabench::osipov::osipov_parallel_wp>();
}
