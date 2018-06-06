/*******************************************************************************
 * Copyright (C) 2018 Oliver Magiera <magiera.o@googlemail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cmath>
#include <tuple>
#include <util/string.hpp>
#include <util/span.hpp>
#include <util/alphabet.hpp>
#include <util/container.hpp>
#include <util/sort/bucketsort.hpp>
#include <util/assertions.hpp>

//TODO: Move helper functions to different files/classes/structs
namespace sacabench::saca::divsufsort {
    template <typename sa_index>
    struct sa_types {
        enum class s_type { l, s, rms };

        inline static bool is_l_type(sa_index suffix, util::span<bool> suffix_types) {
            DCHECK_LT(suffix, suffix_types.size());
            return suffix_types[suffix] == 1;
        }

        inline static bool is_s_type(sa_index suffix, util::span<bool> suffix_types) {
            DCHECK_LT(suffix, suffix_types.size());
            return suffix_types[suffix] == 0;
        }

        inline static bool is_rms_type(sa_index suffix, util::span<bool> suffix_types) {
            DCHECK_LT(suffix, suffix_types.size());
            // Check wether suffix is last index.
            if(suffix + 1 >= suffix_types.size()) {
                return 0;
            }
            //Checks wether suffix at position suffix is s type and suffix at
            //pos suffix + 1 is l type (i.e. rms)
            return suffix_types[suffix] == 0 && suffix_types[suffix + 1] == 1;
        }

    };

    template <typename sa_index>
    struct rms_suffixes {
        const util::string_span text;
        util::span<sa_index> relative_indices;
        util::span<sa_index> absolute_indices;
    }

    struct buckets {
        size_t alphabet_size;

        //l_buckets containing buckets for l-suffixes of size of alphabet
        util::span<std::size_t> l_buckets;
        //s_buckets containing buckets for s- and rms-suffixes of size
        //of alphabet squared
        util::span<std::size_t> s_buckets;

        inline static size_t get_alphabet_size() {
            return alphabet_size;
        }

        template <typename sa_index>
        inline static size_t get_s_bucket_index(character first_letter,
            character second_letter) {
            return first_letter * alphabet_size + second_letter;
        }

        template <typename sa_index>
        inline static size_t get_rms_bucket_index(character first_letter,
            character second_letter) {
                return second_letter * alphabet_size + first_letter;
            }
    };

    class divsufsort {
    public:
        //TODO
        template<typename sa_index>
        static void construct_sa(util::string_span text,
                                 util::alphabet const& alphabet,
                                 util::span<sa_index> out_sa) {
            //Compute effective alphabet

            //Create container for l/s types
            util::container<bool> sa_types = util::make_container(text.size());
            util::span<bool> span_sa_types = util::span(sa_types);

            //Compute l/s types for given text

        };

        //TODO
        template<typename sa_index>
        inline static void insert_into_buckets(rms_suffixes rms_suf,
            buckets bkts) {
            sa_index current_index, relative_index;
            character first_letter, second_letter;
            for(size_t pos = 0; pos < rms_suf.absolute_indices.size(); ++pos) {
                // Retrieve index and first two characters for current rms-
                // suffix
                current_index = rms_suf.absolute_indices[pos];
                first_letter = rms_suf.text[current_index];
                second_letter = rms_suf.text[current_index+1];
                // Retrieve index for current bucket containing the bucket's
                // border.
                //TODO: Check wether new bucket borders are correct.
                bucket_border = bkts.get_rms_bucket_index(first_letter,
                    second_letter);
                relative_index = bkts[bucket_border]--;
                // Set current suffix into correct "bucket" at beginning of sa
                // (i.e. into relative_indices)
                rms_suf.relative_indices[relative_index] = current_index;
            }
        }

        //TODO
        template<typename sa_index>
        inline static void sort_rms_substrings(rms_suffixes rms_suf) {
            //Compute RMS-Substrings (tupel with start-/end-position)

            // Create tupel for last rms-substring: from index of last
            // rms-suffix to index of sentinel
            std::tuple<sa_index, sa_index> substring =
            std::make_tuple(
                rms_suf.absolute_indices[rms_suf.absolute_indices.size()-1],
                text.size()-1);
            util::container<tuple<sa_index, sa_index>> substrings =
            make_container(rms_suf.absolute_indices.size());
            util::span<tuple<sa_index, sa_index>> substrings_span =
            util::span(substrings);
            substrings_span[subtrings_span.end()] = substring;
            for(std::size_t current_index = 0; current_index <
            rms_suf.absolute_indices.size() - 1; ++current_index) {
                // Create RMS-Substring for rms-suffix from suffix-index of rms
                // and starting index of following rms-suffix + 1
                substring = std::make_tuple(
                    rms_suf.absolute_indices[current_index],
                    rms_suf.absolute_indices[current_index+1] +1);
                substrings_span[current_index] = substring;
            }

            //Sort rms-substrings inside of buckets
        }


        // Temporary function for suffix-types, until RTL-Extraction merged.
        static void get_types_tmp(util::string_span text, util::span<bool> types) {
            // Check wether given span has same size as text.
            DCHECK_EQ(text.size(), types.size());
            // Last index always l-type suffix
            types[text.size()-1] = 0;

            for(std::size_t prev_pos = text.size() - 1; prev_pos > 0;
            --prev_pos) {
                if(text[prev_pos - 1] == text[prev_pos]) {
                    types[prev_pos - 1] = types[prev_pos];
                } else {
                    // S == 0, L == 1
                    types[prev_pos - 1] = (text[prev_pos - 1] < text[prev_pos])
                    ? 0 : 1;
                }
            }
        }


        //TODO: Maybe split into count_for_s_type, count_for_rms_type for easier use
        inline static std::size_t count_for_type_in_bucket(
                util::sort::bucket bucket_to_search,
                        util::container<bool> suffix_types,
                sa_types::s_type type_to_check) {
            std::size_t counted = 0;
            std::size_t next_bucket = bucket_to_search.position +
                    bucket_to_search.count;
            for(std::size_t pos = bucket_to_search.position;
                    pos < next_bucket; ++pos) {
                // TODO: Check for l-types probably not needed
                if(type_to_check == sa_types::s_type::l) {
                    if(sa_types::is_l_type(pos, suffix_types)) {
                        counted++;
                    }
                } else if(type_to_check == sa_types::s_type::s) {
                    if(sa_types::is_s_type(pos, suffix_types)) {
                        counted++;
                    }
                } else {
                    if(sa_types::is_rms_type(pos, suffix_types)) {
                        counted++;
                    }
                }
            }
            return counted;
        }

        /**
         * Computes the bucket sizes for l-, s- and rms-suffixes.
         * @param input The input text.
         * @param alphabet The given alphabet of the text.
         * @param suffix_types The suffix types of the corresponding suffixes.
         * Needed to determine which bucket to increase.
         * @param l_buckets The buckets for the l-suffixes. Contains a bucket
         * for each symbol (sentinel included), i.e. l_buckets.size() =
         * |alphabet|+1
         * @param s_buckets The buckets for the s- and rms-suffixes. Contains a
         * bucket for each two symbols (sentinel included for completeness),
         * i.e. s_buckets.size() = (|alphabet| + 1)²
         */
         //TODO: Testing
        inline static void compute_buckets
                (util::string_span input, util::alphabet alphabet,
                 util::span<bool> suffix_types, buckets sa_buckets) {
            const std::size_t bucket_depth = 2;
            // Use Methods from bucketsort.hpp to compute bucket sizes.
            auto buckets = util::sort::get_buckets(input,
                    alphabet.size_without_sentinel(), bucket_depth);
            // Indices indicating the buckets. current_leftmost for first letter only,
            // current_rightmost for first + second letter.
            for(std::size_t first_letter = 0, current_leftmost = 0,
                    current_rightmost = 0; first_letter <
                            alphabet.max_character_value() + 1; ++first_letter) {
                // Need to adjust index for current l-bucket: only one character
                // considered, but buckets contain two (for s/rms buckets)
                current_leftmost = current_rightmost;
                sa_buckets.l_buckets[first_letter] =
                        buckets[current_leftmost].position;

                // Compute relative starting positions for rms-buckets.
                for(std::size_t second_letter = 0; second_letter <
                        alphabet.max_character_value() + 1; ++second_letter,
                        ++current_rightmost) {
                    std::size_t counted_s =
                            count_for_type_in_bucket(buckets[current_rightmost],
                                                     suffix_types,
                                                     sa_types::s_type::s);
                    std::size_t counted_rms =
                            count_for_type_in_bucket(buckets[current_rightmost],
                                                     suffix_types,
                                                     sa_types::s_type::rms);
                    std::size_t index_s = first_letter *
                            (alphabet.max_character_value()+1) + second_letter;
                    std::size_t index_rms = second_letter *
                            (alphabet.max_character_value()+1) + first_letter;
                    sa_buckets.s_buckets[index_s] = counted_s;
                    sa_buckets.s_buckets[index_rms] = counted_rms;
                }
                // Contains value for current_leftmost for next iteration
                current_rightmost++;
            }
        }
    };
}
