/*******************************************************************************
 * Copyright (C) 2018 Oliver Magiera <magiera.o@googlemail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <sacabench/saca/divsufsort.hpp>
#include <util/alphabet>
#include <util/span>
#include <util/string>

using namespace sacabench::saca::divsufsort {

    TEST(DivSufSort, extract_rms) {
        util::string text = "caabaccaabacaa\0"_s;
        auto output = make_container<sa_index_t>(text.size());
        util::alphabet alphabet = util::apply_effective_alphabet(text);

        auto types = make_container<bool>(text.size());
        ASSERT_EQ(types.size(), text.size());
        get_types_tmp(text, types);

        auto left_border = extract_rms_suffixes(text, types, output);

        auto rms_gt = make_container<size_t>(4);
        rms_gt[0] = 2;
        rms_gt[1] = 4;
        rms_gt[2] = 8;
        rms_gt[3] = 10;

        ASSERT_EQ(output.size() - left_border, rms_gt.size());

        for (size_t index = 0; index < rms_gt.size(); ++index) {
            ASSERT_EQ(rms_gt[index], output[left_border + index]);
        }
    }

    TEST(DivSufSort, correct_bucket_sizes) {
        // TODO: needed for several test-cases -> extract into helper function
        util::string text = "caabaccaabacaa\0"_s;
        auto output = make_container<sa_index_t>(text.size());
        util::alphabet alphabet = util::apply_effective_alphabet(text);

        auto types = make_container<bool>(text.size());
        ASSERT_EQ(types.size(), text.size());
        get_types_tmp(text, types);

        buckets bkts = {/*.alphabet_size=*/alphabet.max_character_value(),
                        /*.l_buckets=*/
                        util::make_container<sa_index>(
                            alphabet.max_character_value() + 1), /*.s_buckets=*/
                        util::make_container<sa_index>(
                            pow(alphabet.max_character_value() + 1, 2))};
        buckets result = {/*.alphabet_size=*/alphabet.max_character_value(),
                        /*.l_buckets=*/
                        util::make_container<sa_index>(
                            alphabet.max_character_value() + 1), /*.s_buckets=*/
                        util::make_container<sa_index>(
                            pow(alphabet.max_character_value() + 1, 2))};

        ASSERT_EQ(bkts.l_buckets.size(), 4);

        bkts.l_buckets[0] = 0;
        bkts.l_buckets[1] = 1;
        bkts.l_buckets[2] = 9;
        bkts.l_buckets[3] = 11;

        ASSERT_EQ(bkts.s_buckets.size(), 16);
        
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
        
        
        bkts.s_buckets[5] = 2;      // (a,a) for s
        bkts.s_buckets[9] = 2;      // (a,b) for rms
        bkts.s_buckets[13] = 4;     // (a,c) for rms
        bkts.s_buckets[14] = 4;     // (b,c) for rms
        
        compute_buckets(text, alphabet, types, result);
        
        // Assertions for l_buckets
        for(size_t index = 0; index < bkts.l_buckets.size(); ++index) {
            ASSERT_EQ(bkts.l_buckets[index], result.l_buckets[index]);
        }
        
        // Assertions for s_buckets (i.e. for s- and rms-buckets)
        for(size_t index = 0; index < bkts.s_buckets.size(); ++index) {
            ASSERT_EQ(bkts.s_buckets[index], result.s_buckets[index]);
        }
    }
}
