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
#include <saca/osipov/osipov_parallel.hpp>
#include "test/saca.hpp"

using namespace sacabench::osipov;
using namespace sacabench;
using namespace sacabench::util;

TEST(Osipov, CornerCases) {
    test::saca_corner_cases<sacabench::osipov::osipov_parallel>();
}

TEST(OsipovWp, CornerCases) {
    test::saca_corner_cases<sacabench::osipov::osipov_parallel_wp>();
}
/*
TEST(OsipovWp, SpecialInput) {
    string_span text = "Lorem ipsum dolor sit amet, sea ut etiam solet salut"
         "andi, sint complectitur et his, ad salutandi imperdi"
         "et gubergren per mei."_s;

         size_t slice_limit = 40;

         std::stringstream ss;

         ss << "Test SACA on ";
         if (text.size() > slice_limit) {
             size_t i = slice_limit;
             while (i < text.size() && (text[i] >> 6 == 0b10)) {
                 i++;
             }
             ss << "'" << text.slice(0, i) << "[...]'";
         } else {
             ss << "'" << text << "'";
         }
         ss << " (" << text.size() << " bytes)" << std::endl;
         std::cout << ss.str();

         auto output = prepare_and_construct_sa<sacabench::osipov::osipov_parallel_wp, size_t>(
             text_initializer_from_span(text));

         auto fast_result = sa_check(output.sa_without_sentinels(), text);
         if (fast_result != sa_check_result::ok) {
                 std::cout << ss.str();
             auto slow_result =
                 sa_check_naive(output.sa_without_sentinels(), text);
             ASSERT_EQ(bool(fast_result), bool(slow_result))
                 << "BUG IN SA CHECKER DETECTED!";
             ASSERT_EQ(fast_result, sa_check_result::ok);
         }
}*/
