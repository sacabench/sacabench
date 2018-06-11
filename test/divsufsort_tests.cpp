/*******************************************************************************
 * Copyright (C) 2018 Oliver Magiera <magiera.o@googlemail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <sacabench/saca/divsufsort.hpp>
#include <util/string>
#include <util/span>
#include <util/alphabet>

using namespace sacabench::saca::divsufsort {
    
TEST(DivSufSort, extract_rms) {
    util::string_span text = "caabaccaabacaa"_s;
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
    
    for(size_t index = 0; index < rms_gt.size(); ++index) {
        ASSERT_EQ(rms_gt[index], output[left_border + index]);
    }
}

TEST(DivSufSort, correct_bucket_sizes) {
    util::string_span text = "caabaccaabacaa"_s;
    auto output = make_container<sa_index_t>(text.size());
    util::alphabet alphabet = util::apply_effective_alphabet(text);
    
    auto types = make_container<bool>(text.size());
    ASSERT_EQ(types.size(), text.size());
    get_types_tmp(text, types);
    
    buckets bkts = { /*.alphabet_size=*/alphabet.max_character_value(), 
    /*.l_buckets=*/util::make_container<sa_index>(
    alphabet.max_character_value()), /*.s_buckets=*/
    util::make_container<sa_index>(pow(alphabet.max_character_value(),
    2)) };
    
    ASSERT_EQ(bkts.l_buckets.size(), 4);
    
    bkts.l_buckets[0] = 0;
    bkts.l_buckets[1] = 1;
    bkts.l_buckets[2] = 9;
    bkts.l_buckets[3] = 11;
    
    ASSERT_EQ(bkts.s_buckets.size(), 16);
    //TODO: Create ground truth instance for s_buckets.
}


bool contains_element(size_t element, span<size_t> set) {
    for(size_t index : set) {
        if(index == set) {
            return true;
        }
    }
    return false;
}

}