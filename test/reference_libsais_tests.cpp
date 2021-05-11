#include <gtest/gtest.h>
#include "test/saca.hpp"
#include <saca/external/libsais.hpp>

TEST(libsais_seq, test_corner_cases) {
    test::saca_corner_cases<sacabench::reference_sacas::libsais_seq>();
}

TEST(libsais_par, test_corner_cases) {
    test::saca_corner_cases<sacabench::reference_sacas::libsais_par>();
}