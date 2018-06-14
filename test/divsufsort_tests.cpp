/*******************************************************************************
 * Copyright (C) 2018 Oliver Magiera <magiera.o@googlemail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <iostream>
#include <saca/divsufsort.hpp>
#include <util/alphabet.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

using namespace sacabench::saca::divsufsort;
using namespace sacabench;
using dss = divsufsort<std::size_t>;

TEST(DivSufSort, extractRms) {
    util::string text = "caabaccaabacaa\0"_s;
    auto output = util::make_container<std::size_t>(text.size());
    sacabench::util::apply_effective_alphabet(text.slice(0, text.size() - 1));

    auto types = util::make_container<bool>(text.size());
    ASSERT_EQ(types.size(), text.size());
    dss::get_types_tmp(text, types);

    auto count = dss::extract_rms_suffixes(text, types, output);

    auto rms_gt = util::make_container<std::size_t>(4);
    rms_gt[0] = 2;
    rms_gt[1] = 4;
    rms_gt[2] = 8;
    rms_gt[3] = 10;

    ASSERT_EQ(count, rms_gt.size());

    auto left_border = text.size() - count;

    for (std::size_t index = 0; index < count; ++index) {
        ASSERT_EQ(rms_gt[index], output[left_border + index]);
    }
}

TEST(DivSufSort, correctBucketSizes) {
    // TODO: needed for several test-cases -> extract into helper function
    util::string text = "caabaccaabacaa\0"_s;
    auto output = util::make_container<std::size_t>(text.size());
    util::alphabet alphabet =
        util::apply_effective_alphabet(text.slice(0, text.size() - 1));

    auto types = util::make_container<bool>(text.size());
    ASSERT_EQ(types.size(), text.size());
    dss::get_types_tmp(text, types);

    auto l1 =
        util::make_container<std::size_t>(alphabet.max_character_value() + 1);
    auto l2 =
        util::make_container<std::size_t>(alphabet.max_character_value() + 1);
    auto s1 = util::make_container<std::size_t>(
        pow(alphabet.max_character_value() + 1, 2));
    auto s2 = util::make_container<std::size_t>(
        pow(alphabet.max_character_value() + 1, 2));

    buckets bkts = {/*.alphabet_size=*/alphabet.max_character_value() + 1,
                    /*.l_buckets=*/l1, /*.s_buckets=*/s1};
    buckets result = {/*.alphabet_size=*/alphabet.max_character_value() + 1,
                      /*.l_buckets=*/l2, /*.s_buckets=*/s2};

    ASSERT_EQ(bkts.l_buckets.size(), std::size_t(4));

    bkts.l_buckets[0] = 0;
    bkts.l_buckets[1] = 1;
    bkts.l_buckets[2] = 9;
    bkts.l_buckets[3] = 11;

    ASSERT_EQ(bkts.s_buckets.size(), std::size_t(16));

    /*
                                // buckets for types
    bkts.s_buckets[5] = 2;      // (a,a) for s
    bkts.s_buckets[6] = 2;      // (a,b) for s
    bkts.s_buckets[7] = 2;      // (a,c) for s
    bkts.s_buckets[9] = 2;      // (a,b) for rms
    bkts.s_buckets[10] = 2;     // (b,b) for s
    bkts.s_buckets[11] = 2;     // (b,c) for s
    bkts.s_buckets[13] = 4;     // (a,c) for rms
    //bkts.s_buckets[14] = 4;     // (b,c) for rms
    //bkts.s_buckets[15] = 2;     // (c,c) for s
    */

    bkts.s_buckets[5] = 2;  // (a,a) for s
    bkts.s_buckets[9] = 2;  // (a,b) for rms
    bkts.s_buckets[13] = 4; // (a,c) for rms
    bkts.s_buckets[14] = 4; // (b,c) for rms

    dss::compute_buckets(text, alphabet, types, result);

    std::cout << "l-buckets:" << std::endl;
    // Assertions for l_buckets
    for (std::size_t index = 0; index < bkts.l_buckets.size(); ++index) {
        std::cout << "Index: " << index
                  << " # computed: " << result.l_buckets[index]
                  << " # should: " << bkts.l_buckets[index] << std::endl;
        ASSERT_EQ(bkts.l_buckets[index], result.l_buckets[index]);
    }

    std::cout << "s-buckets:" << std::endl;

    // Assertions for s_buckets (i.e. for s- and rms-buckets)
    for (std::size_t index = 0; index < bkts.s_buckets.size(); ++index) {
        std::cout << "Index: " << index
                  << " # computed: " << result.s_buckets[index]
                  << " # should: " << bkts.s_buckets[index] << std::endl;
        ASSERT_EQ(bkts.s_buckets[index], result.s_buckets[index]);
    }
}

TEST(DivSufSort, sortRmsSubstrings) {
    util::string text = "caabaccaabacaa\0"_s;
    auto output = util::make_container<std::size_t>(text.size());
    util::alphabet alphabet =
        util::apply_effective_alphabet(text.slice(0, text.size() - 1));
    auto sa_type_container = util::make_container<bool>(text.size());

    // Compute l/s types for given text; TODO: Replace with version from
    // 'extract_types.hpp' after RTL-Insertion was merged.
    dss::get_types_tmp(text, sa_type_container);
    size_t rms_count =
        dss::extract_rms_suffixes(text, sa_type_container, output);
    // Initialize struct rms_suffixes with text, relative positions
    // (first rms_count positions in output) and absolute positions
    // (last rms_count positions in output) for rms-suffixes
    rms_suffixes<size_t> rms_suf = {
        /*.text=*/text,
        /*.relative_indices=*/output.slice(0, rms_count),
        /*.absolute_indices=*/
        output.slice(output.size() - rms_count, output.size())};
    auto s_bkt = util::make_container<std::size_t>(
        pow(alphabet.max_character_value() + 1, 2));
    auto l_bkt =
        util::make_container<std::size_t>(alphabet.max_character_value() + 1);
    // Initialize buckets: alphabet_size slots for l-buckets,
    // alphabet_size² for s-buckets
    buckets bkts = {/*.alphabet_size=*/alphabet.max_character_value() + 1,
                    /*.l_buckets=*/l_bkt, /*.s_buckets=*/s_bkt};

    std::cout << "Computing bucket sizes." << std::endl;
    dss::compute_buckets(text, alphabet, sa_type_container, bkts);

    std::cout << "Inserting rms-suffixes into buckets" << std::endl;
    dss::insert_into_buckets(rms_suf, bkts);

    std::cout << "Sorting RMS-Substrings." << std::endl;
    dss::sort_rms_substrings(rms_suf, alphabet, bkts);

    auto rel_ind = util::container<size_t>({2, 0, 3, 1});
    for (size_t pos = 0; pos < rel_ind.size(); ++pos) {
        std::cout << "Index: " << rms_suf.relative_indices[pos]
                  << " , should: " << rel_ind[pos] << std::endl;
        ASSERT_EQ(rms_suf.relative_indices[pos], rel_ind[pos]);
    }
}
