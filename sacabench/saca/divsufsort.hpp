/*******************************************************************************
 * Copyright (C) 2018 Oliver Magiera <magiera.o@googlemail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cmath>
#include <iostream>
//#include <tuple>
#include <util/alphabet.hpp>
#include <util/assertions.hpp>
#include <util/container.hpp>
#include <util/sort/bucketsort.hpp>
#include <util/sort/introsort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <utility>

// TODO: Move helper functions to different files/classes/structs
namespace sacabench::saca::divsufsort {
template <typename sa_index>
struct sa_types {
    enum class s_type { l, s, rms };

    inline static bool is_l_type(sa_index suffix,
                                 util::span<bool> suffix_types) {
        DCHECK_LT(suffix, suffix_types.size());
        return suffix_types[suffix] == 1;
    }

    inline static bool is_s_type(sa_index suffix,
                                 util::span<bool> suffix_types) {
        // Last suffix must be l-type
        DCHECK_LT(suffix, suffix_types.size() - 1);
        return suffix_types[suffix] == 0 && suffix_types[suffix + 1] == 0;
    }

    inline static bool is_rms_type(sa_index suffix,
                                   util::span<bool> suffix_types) {
        // Last suffix must be l-type
        DCHECK_LT(suffix, suffix_types.size() - 1);
        // Checks wether suffix at position suffix is s type and suffix at
        // pos suffix + 1 is l type (i.e. rms)
        return suffix_types[suffix] == 0 && suffix_types[suffix + 1] == 1;
    }
};

template <typename sa_index>
struct rms_suffixes {
    const util::string_span text;
    util::span<sa_index> relative_indices;
    util::span<sa_index> absolute_indices;
    // util::span<sa_index> partial_isa;
};

struct buckets {
    size_t alphabet_size;

    // l_buckets containing buckets for l-suffixes of size of alphabet
    util::container<std::size_t>& l_buckets;
    // s_buckets containing buckets for s- and rms-suffixes of size
    // of alphabet squared
    util::container<std::size_t>& s_buckets;

    // TODO: Check wether size_t for first_letter/second_letter of better use
    inline size_t get_s_bucket_index(util::character first_letter,
                                     util::character second_letter) {
        return first_letter * alphabet_size + second_letter;
    }

    inline size_t get_rms_bucket_index(util::character first_letter,
                                       util::character second_letter) {
        return second_letter * alphabet_size + first_letter;
    }
};

template <typename sa_index>
struct compare_rms_substrings {
public:
    inline compare_rms_substrings(
        const util::string_span text,
        util::container<std::pair<sa_index, sa_index>>& substrings)
        : input(text), substrings(substrings) {
        std::cout << "Initializing rms-substrings compare fct." << std::endl;
    }

    inline bool operator()(const sa_index& elem,
                           const sa_index& compare_to) const {
        // DCHECK_NE(elem, compare_to);
        if (elem == compare_to) {
            return false;
        }
        const bool elem_too_large = (elem >= substrings.size());
        const bool compare_to_too_large = (compare_to >= substrings.size());

        if (elem_too_large) {
            if (compare_to_too_large) {
                // Check how to handle this case (possibly the shorter one is
                // smaller)
                return elem < compare_to;
            }
            return true;
        }
        if (compare_to_too_large) {
            DCHECK_EQ(elem_too_large, false);
            return false;
        }
        sa_index elem_size =
            std::get<1>(substrings[elem]) - std::get<0>(substrings[elem]) + 1;
        sa_index compare_to_size = std::get<1>(substrings[compare_to]) -
                                   std::get<0>(substrings[compare_to]) + 1;
        sa_index max_pos = std::min(elem_size, compare_to_size);
        sa_index elem_begin = std::get<0>(substrings[elem]);
        sa_index compare_to_begin = std::get<0>(substrings[compare_to]);
        sa_index elem_index = elem_begin + 2,
                 compare_to_index = compare_to_begin + 2;

        for (sa_index pos = 2; pos < max_pos; ++pos) {
            std::cout << "Current index :" << pos << std::endl;
            std::cout << "Comparing " << (size_t)input[elem_index] << " to "
                      << (size_t)input[compare_to_index] << std::endl;
            if (input[elem_index] == input[compare_to_index]) {
                ++elem_index;
                ++compare_to_index;
            } else {
                std::cout << "Symbol " << (size_t)input[elem_index]
                          << " differs from " << (size_t)input[compare_to_index]
                          << std::endl;
                return input[elem_index] < input[compare_to_index];
            }
        }
        std::cout << "Substrings have been the same until now." << std::endl;
        // If one substring is shorter than the other and they are the same
        // until now:
        elem_size =
            std::get<1>(substrings[elem]) - std::get<0>(substrings[elem]);
        compare_to_size = std::get<1>(substrings[compare_to]) -
                          std::get<0>(substrings[compare_to]);
        // Either they differ in length (shorter string is smaller) or they have
        // the same length (i.e. return false)
        return (elem_size == compare_to_size) ? false
                                              : elem_size < compare_to_size;
    }

private:
    const util::string_span input;
    util::container<std::pair<sa_index, sa_index>>& substrings;
};

template <typename sa_index>
class divsufsort {
public:
    static const std::size_t EXTRA_SENTINELS = 1;
    static constexpr sa_index NEGATIVE_MASK = size_t(1)
                                              << (sizeof(sa_index) * 8 - 1);

    // TODO
    static void construct_sa(util::string_span text,
                             util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {
        DCHECK_EQ(text.size(), out_sa.size());
        // Create container for l/s types
        auto sa_type_container = util::make_container<bool>(text.size());

        // Compute l/s types for given text; TODO: Replace with version from
        // 'extract_types.hpp' after RTL-Insertion was merged.
        get_types_tmp(text, sa_type_container);
        sa_index rms_count =
            extract_rms_suffixes(text, sa_type_container, out_sa);
        // Initialize struct rms_suffixes with text, relative positions
        // (first rms_count positions in out_sa) and absolute positions
        // (last rms_count positions in out_sa) for rms-suffixes
        rms_suffixes rms_suf = {
            /*.text=*/text, /*.relative_indices=*/
            out_sa.slice(0, rms_count),
            /*.absolute_indices=*/
            out_sa.slice(out_sa.size() - rms_count, out_sa.size())};

        // Initialize buckets: alphabet_size slots for l-buckets,
        // alphabet_size² for s-buckets
        buckets bkts = {/*.alphabet_size=*/alphabet.max_character_value() + 1,
                        /*.l_buckets=*/
                        util::make_container<sa_index>(
                            alphabet.max_character_value() + 1), /*.s_buckets=*/
                        util::make_container<sa_index>(
                            pow(alphabet.max_character_value() + 1, 2))};

        compute_buckets(text, alphabet, sa_type_container, bkts);

        insert_into_buckets(rms_suf, bkts);

        sort_rms_substrings(rms_suf, alphabet, bkts);
    }

    // TODO: Change in optimization-phase (while computing l/s-types,
    // counting bucket sizes)
    inline static sa_index
    extract_rms_suffixes(util::string_span text,
                         util::container<bool>& sa_types_container,
                         util::span<sa_index> out_sa) {
        DCHECK_EQ(text.size(), sa_types_container.size());
        // First (right) index from interval of already found rms-suffixes
        // [rms_begin, rms_end)
        sa_index right_border = out_sa.size();
        // Insert rms-suffixes from right to left
        for (sa_index current = text.size() - 1; current > 0; --current) {
            if (sa_types<sa_index>::is_rms_type(current - 1,
                                                sa_types_container)) {
                // Adjust border to new entry (rms-suffix)
                out_sa[--right_border] = current - 1;
            }
        }
        // Count of rms-suffixes
        return text.size() - right_border;
    }

    inline static void insert_into_buckets(rms_suffixes<sa_index>& rms_suf,
                                           buckets& bkts) {
        sa_index current_index, relative_index;
        util::character first_letter, second_letter;
        size_t bucket_index;
        sa_index rms_count = rms_suf.absolute_indices.size();
        // Skip last rms-suffix in this loop
        for (sa_index pos = rms_count - 1; 0 < pos; --pos) {
            // Retrieve index and first two characters for current rms-
            // suffix
            current_index = rms_suf.absolute_indices[pos - 1];
            first_letter = rms_suf.text[current_index];
            second_letter = rms_suf.text[current_index + 1];
            // Retrieve index for current bucket containing the bucket's
            // border.
            // TODO: Check wether new bucket borders are correct.
            bucket_index =
                bkts.get_rms_bucket_index(first_letter, second_letter);
            relative_index = --bkts.s_buckets[bucket_index];
            // Set current suffix into correct "bucket" at beginning of sa
            // (i.e. into relative_indices)
            std::cout << "Inserting " << pos - 1 << " into " << relative_index
                      << std::endl;
            rms_suf.relative_indices[relative_index] = pos - 1;
        }
        current_index = rms_suf.absolute_indices[rms_count - 1];
        first_letter = rms_suf.text[current_index];
        second_letter = rms_suf.text[current_index + 1];
        // Retrieve index for current bucket containing the bucket's
        // border.
        // TODO: Check wether new bucket borders are correct.
        bucket_index = bkts.get_rms_bucket_index(first_letter, second_letter);
        relative_index = --bkts.s_buckets[bucket_index];
        // Sort last rms-suffix into correct bucket:
        std::cout << "Inserting " << rms_count - 1 << " into " << relative_index
                  << std::endl;
        rms_suf.relative_indices[relative_index] = rms_count - 1;
    }

    inline static util::container<std::pair<sa_index, sa_index>>
    extract_rms_suffixes(rms_suffixes<sa_index>& rms_suf) {
        sa_index rms_count = rms_suf.absolute_indices.size();
        // Create tupel for last rms-substring: from index of last
        // rms-suffix to index of sentinel
        std::pair<sa_index, sa_index> substring = std::make_pair(
            rms_suf.absolute_indices[rms_suf.absolute_indices.size() - 1],
            rms_suf.text.size() - 1);
        auto substrings_container =
            util::make_container<std::pair<sa_index, sa_index>>(rms_count);
        std::cout << substrings_container.end() << std::endl;
        substrings_container[substrings_container.size() - 1] = substring;

        sa_index substr_start, substr_end;
        for (sa_index current_index = 0; current_index < rms_count - 1;
             ++current_index) {

            substr_start = rms_suf.absolute_indices[current_index];
            substr_end = rms_suf.absolute_indices[current_index + 1] + 1;
            std::cout << "Creating substring <" << substr_start << ","
                      << substr_end << ">" << std::endl;
            // Create RMS-Substring for rms-suffix from suffix-index of rms
            // and starting index of following rms-suffix + 1
            substring = std::make_pair(substr_start, substr_end);
            substrings_container[current_index] = substring;
        }
        // Create substring for last rms-suffix from rms-suffix index to
        // sentinel
        substr_start = rms_suf.absolute_indices[rms_count - 1];
        substr_end = rms_suf.text.size() - 1;
        std::cout << "Creating substring <" << substr_start << "," << substr_end
                  << ">" << std::endl;
        // Create RMS-Substring for rms-suffix from suffix-index of rms
        // and starting index of following rms-suffix + 1
        substring = std::make_pair(substr_start, substr_end);
        substrings_container[rms_count - 1] = substring;
        return substrings_container;
    }

    inline static void set_unsorted_rms_substring_intervals(
        rms_suffixes<sa_index>& rms_suf, compare_rms_substrings<sa_index> cmp,
        sa_index interval_start, sa_index interval_end) {
        sa_index elem, compare_to;
        // Last element to be compared to its predecessor: interval_start + 1
        // interval_end must not contain last element of interval
        bool less, greater;
        for (sa_index pos = interval_end - 1; interval_start < pos; --pos) {
            elem = rms_suf.relative_indices[pos - 1];
            compare_to = rms_suf.relative_indices[pos];
            // None of the substrings is smaller than the other (i.e. the same)
            less = cmp(elem, compare_to);
            greater = cmp(compare_to, elem);
            std::cout << "Index " << elem << " is smaller than " << compare_to
                      << ":" << less << std::endl;
            std::cout << "Index " << elem << " is greater than " << compare_to
                      << ":" << greater << std::endl;

            if (!(cmp(elem, compare_to) || cmp(compare_to, elem))) {
                rms_suf.relative_indices[pos] |= NEGATIVE_MASK;
                std::cout
                    << "Negated index " << compare_to
                    << " because it's the same substring as its predecessor."
                    << std::endl;
            }
        }
    }

    inline static void sort_rms_substrings(rms_suffixes<sa_index>& rms_suf,
                                           util::alphabet& alph,
                                           buckets sa_buckets) {
        // Compute RMS-Substrings (tupel with start-/end-position)
        auto substrings_container = extract_rms_suffixes(rms_suf);
        compare_rms_substrings<sa_index> cmp(rms_suf.text,
                                             substrings_container);

        // Sort rms-substrings (suffix-indices) inside of buckets
        size_t bucket_index;
        sa_index interval_begin, interval_end = rms_suf.relative_indices.size();
        util::span<sa_index> current_interval;
        // Sort every rms-bucket, starting at last bucket
        for (util::character first_letter = alph.max_character_value() - 1;
             0 < first_letter; --first_letter) {
            for (util::character second_letter = alph.max_character_value();
                 first_letter < second_letter; --second_letter) {
                std::cout << "Currently sorting (" << (size_t)first_letter
                          << "," << (size_t)second_letter << ")-bucket."
                          << std::endl;
                bucket_index = sa_buckets.get_rms_bucket_index(first_letter,
                                                               second_letter);
                interval_begin = sa_buckets.s_buckets[bucket_index];
                // Interval of indices to sort
                current_interval = rms_suf.relative_indices.slice(
                    interval_begin, interval_end);
                // sort current interval/bucket
                util::sort::introsort<sa_index,
                                      compare_rms_substrings<sa_index>>(
                    current_interval, cmp);

                // Modify elements in interval if they are the same (MSB set)
                std::cout
                    << "Checking bucket for unsorted (i.e. same) substrings."
                    << std::endl;
                set_unsorted_rms_substring_intervals(
                    rms_suf, cmp, interval_begin, interval_end);
                // Refresh end_index
                interval_end = interval_begin;
            }
        }
    }

    inline static void compute_partial_isa(rms_suffixes<sa_index>& rms_suf) {
        sa_index sorted_count = 0;
        sa_index unsorted_count = 0;
        util::span<sa_index> rel_ind = rms_suf.relative_indices;
        sa_index rank = rel_ind.size() - 1;
        for (sa_index pos = rel_ind.size(); 0 < pos; --pos) {
            // Has been negated
            if (rel_ind[pos - 1] & NEGATIVE_MASK == 0) {
                if (unsorted_count > 0) {
                    std::cout << "Set rank for index " << rel_ind[pos] << " to "
                              << rank << std::endl;
                    // Set rank for "final" index in unsorted interval (i.e.
                    // interval with same substrings).
                    rms_suf.partial_isa[rel_ind[pos]] = rank;
                    // Reduce rank by unsorted_count (number of indices with
                    // same rank)
                    rank -= unsorted_count;
                    std::cout << "Reduced rank to " << rank << std::endl;
                    // Reset unsorted count;
                    unsorted_count = 0;
                }
                std::cout << "Set rank for index " << rel_ind[pos - 1] << " to "
                          << rank << std::endl;
                rms_suf.partial_isa[rel_ind[pos - 1]] = rank--;
                ++sorted_count;
            } else {
                // Can only occur if there have been non-negated indices
                if (sorted_count > 0) {
                    std::cout << "Sorted interval of length " << sorted_count
                              << "ending at " << pos << std::endl;
                    // Could cause issues with inducing (normally inverted)
                    rel_ind[pos] = sorted_count | NEGATIVE_MASK;
                    // Reset sorted count
                    sorted_count = 0;
                }
                std::cout << "Set rank for index " << rel_ind[pos] << " to "
                          << rank << std::endl;
                rms_suf.partial_isa[rel_ind[pos - 1]] = rank;
                ++unsorted_count;
            }
        }
    }

    // Temporary function for suffix-types, until RTL-Extraction merged.
    inline static void get_types_tmp(util::string_span text,
                                     util::container<bool>& types) {
        // Check wether given span has same size as text.
        DCHECK_EQ(text.size(), types.size());
        // Last index always l-type suffix
        types[text.size() - 1] = 1;
        for (std::size_t prev_pos = text.size() - 1; prev_pos > 0; --prev_pos) {

            if (text[prev_pos - 1] == text[prev_pos]) {
                types[prev_pos - 1] = types[prev_pos];
            } else {
                // S == 0, L == 1
                types[prev_pos - 1] =
                    (text[prev_pos - 1] < text[prev_pos]) ? 0 : 1;
            }
        }
    }

    /** \brief Counts the amount of rms-suffixes in a given (two-character)
        bucket. Needed to compute the initial rms-buckets

    */
    inline static sa_index count_for_rms_type_in_bucket(
        util::sort::bucket bucket_to_search,
        util::container<bool>& suffix_types_container) {
        sa_index counted = 0;
        // std::cout << "Counting for rms-types:" << std::endl;
        sa_index next_bucket =
            bucket_to_search.position + bucket_to_search.count;
        // std::cout << "Start position of current bucket: " <<
        // bucket_to_search.position << std::endl; std::cout << "Start position
        // of next bucket: " << next_bucket << std::endl;
        for (sa_index pos = bucket_to_search.position; pos < next_bucket;
             ++pos) {
            bool is_rms =
                sa_types<sa_index>::is_rms_type(pos, suffix_types_container);

            // std::cout << "Pos " << pos << " is rms-type: " << is_rms <<
            // std::endl;
            if (is_rms) {
                ++counted;
            }
        }
        return counted;
    }

    /** \brief Counts the amount of s-suffixes (not rms) in a given
        (two-character) bucket. Needed to compute the initial s-buckets

    */
    inline static sa_index
    count_for_s_type_in_bucket(util::sort::bucket bucket_to_search,
                               util::container<bool>& suffix_types_container) {
        sa_index counted = 0;
        sa_index next_bucket =
            bucket_to_search.position + bucket_to_search.count;
        std::cout << "Searching for s-buckets from "
                  << bucket_to_search.position << " to " << next_bucket - 1
                  << std::endl;
        for (sa_index pos = bucket_to_search.position; pos < next_bucket;
             ++pos) {
            bool is_s =
                sa_types<sa_index>::is_s_type(pos, suffix_types_container);
            std::cout << "Pos " << pos << " is s-type: " << is_s << std::endl;
            if (is_s) {
                ++counted;
            }
        }
        // std::cout << counted << " s-types have been counted." << std::endl;
        return counted;
    }

    /**
     * Computes the bucket sizes for l-, s- and rms-suffixes.
     * @param input The input text.
     * @param alphabet The given alphabet of the text.
     * @param suffix_types The suffix types of the corresponding
     * suffixes. Needed to determine which bucket to increase.
     * @param l_buckets The buckets for the l-suffixes. Contains a
     * bucket for each symbol (sentinel included), i.e. l_buckets.size()
     * = |alphabet|+1
     * @param s_buckets The buckets for the s- and rms-suffixes.
     * Contains a bucket for each two symbols (sentinel included for
     * completeness), i.e. s_buckets.size() = (|alphabet| + 1)²
     */
    // TODO: Testing
    inline static void compute_buckets_old(util::string_span input,
                                           util::alphabet alphabet,
                                           util::container<bool>& suffix_types,
                                           buckets sa_buckets) {
        const std::size_t bucket_depth = 2;
        // Use Methods from bucketsort.hpp to compute bucket sizes.
        auto bkts = util::sort::get_buckets(
            input, alphabet.size_without_sentinel(), bucket_depth);
        std::cout << "Total of " << bkts.size()
                  << " buckets precomputed (bucketsort)." << std::endl;
        // Variables used in (inner) loop.
        sa_index counted_s, counted_rms = 0;
        std::size_t index_s, index_rms, current_leftmost, current_rightmost = 0;
        // Indices indicating the buckets. current_leftmost for first
        // letter only, current_rightmost for first + second letter.
        for (util::character first_letter = 0;
             first_letter < alphabet.max_character_value() + 1;
             ++first_letter) {
            // Need to adjust index for current l-bucket: only one
            // character considered, but buckets contain two (for s/rms
            // buckets)
            current_leftmost = current_rightmost;
            std::cout << "Getting first bucket position (l-bucket) for letter "
                      << (size_t)first_letter << ":"
                      << bkts[current_leftmost].position << std::endl;
            sa_buckets.l_buckets[first_letter] =
                bkts[current_leftmost].position + 1;
            // Compute relative starting positions for rms-buckets.
            for (util::character second_letter = 0;
                 second_letter < alphabet.max_character_value() + 1;
                 ++second_letter, ++current_rightmost) {
                std::cout << "Current bucket: " << current_rightmost
                          << " | Starting pos: "
                          << bkts[current_rightmost].position
                          << " | Counted in bucket: "
                          << bkts[current_rightmost].count << std::endl;
                // This order in the text can't be s- or rms-type:
                if (second_letter < first_letter) {
                    continue;
                }
                // Only currently counted for s-bucket
                std::cout << "Counting for s-types in bucket ("
                          << (size_t)first_letter << ","
                          << (size_t)second_letter << ")" << std::endl;
                counted_s = count_for_s_type_in_bucket(bkts[current_rightmost],
                                                       suffix_types);
                std::cout << counted_s << " s-types have been counted."
                          << std::endl;
                index_s =
                    sa_buckets.get_s_bucket_index(first_letter, second_letter);
                std::cout << "Bucket index for s-type bucket ("
                          << (size_t)first_letter << ","
                          << (size_t)second_letter << "):" << index_s
                          << std::endl;
                sa_buckets.s_buckets[index_s] = counted_s;
                // Relative left border of rms-bucket (i.e. sum over all
                // preceding rms-buckets
                std::cout << "Counting for rms-types in bucket ("
                          << (size_t)first_letter << ","
                          << (size_t)second_letter << ")" << std::endl;
                if (first_letter != second_letter) {
                    counted_rms += count_for_rms_type_in_bucket(
                        bkts[current_rightmost], suffix_types);
                    std::cout << counted_rms << " rms-types have been counted."
                              << std::endl;
                    index_rms = sa_buckets.get_rms_bucket_index(first_letter,
                                                                second_letter);
                    std::cout << "Bucket index for rms-type bucket ("
                              << (size_t)first_letter << ","
                              << (size_t)second_letter << "):" << index_rms
                              << std::endl;
                    sa_buckets.s_buckets[index_rms] = counted_rms;
                }
            }
        }
    }

    inline static void compute_buckets(util::string_span input,
                                       util::alphabet alphabet,
                                       util::container<bool>& suffix_types,
                                       buckets sa_buckets) {
        count_buckets(input, suffix_types, sa_buckets);
        prefix_sum(alphabet, sa_buckets);
    }

    inline static void count_buckets(util::string_span input,
                                     util::container<bool>& suffix_types,
                                     buckets sa_buckets) {
        util::character first_letter, second_letter;
        // Used for accessing buckets in sa_buckets.s_buckets
        std::size_t bucket_index;
        for (sa_index current; current < input.size(); ++current) {
            first_letter = input[current];
            if (suffix_types[current] == 1) {
                ++sa_buckets.l_buckets[first_letter];
            } else {
                // Indexing safe because last two indices are always l-type.
                DCHECK_LT(current, input.size() - 1);
                second_letter = input[current + 1];
                // Compute bucket_index regarding current suffix being either
                // s- or rms-type
                // std::cout << "(" << (size_t)first_letter << "," <<
                // (size_t)second_letter << ")-bucket" << std::endl;
                bucket_index = (sa_types<sa_index>::is_rms_type)
                                   ? sa_buckets.get_rms_bucket_index(
                                         first_letter, second_letter)
                                   : sa_buckets.get_s_bucket_index(
                                         first_letter, second_letter);

                // Increase count for bucket at bucket_index by one.
                ++sa_buckets.s_buckets[bucket_index];
                // std::cout << "s-bucket index: " << bucket_index << ", new
                // count: " << sa_buckets.s_buckets[bucket_index] << std::endl;
            }
        }
    }

    inline static void prefix_sum(util::alphabet& alph, buckets sa_buckets) {
        // l_count starts at one because of sentinel (skipped in loop)
        sa_index l_count = 1, rms_relative_count = 0, l_border = 0;
        size_t s_bucket_index;
        // Adjust left border for first l-bucket (sentinel)
        sa_buckets.l_buckets[0] = 0;

        for (util::character first_letter = 1;
             first_letter < alph.max_character_value() + 1; ++first_letter) {
            // New left border completely computed (see inner loop)
            l_border += l_count;
            // Save count for current l-bucket for l_border of next l-bucket
            l_count = sa_buckets.l_buckets[first_letter];
            // Set left border of current bucket
            sa_buckets.l_buckets[first_letter] = l_border;
            // std::cout << "New left border for l-bucket " << (size_t)
            // first_letter << ":" << l_border << std::endl;
            for (util::character second_letter = first_letter;
                 second_letter < alph.max_character_value() + 1;
                 ++second_letter) {
                // Compute index for current s-bucket in s_buckets
                s_bucket_index =
                    sa_buckets.get_s_bucket_index(first_letter, second_letter);
                // Add count for s-bucket to left-border of following l-bucket
                l_border += sa_buckets.s_buckets[s_bucket_index];
                // (c0,c0) buckets can be skipped for rms-buckets (because they
                // don't exist)
                if (first_letter == second_letter) {
                    continue;
                }
                // Compute index for current rms-bucket in s_buckets
                s_bucket_index = sa_buckets.get_rms_bucket_index(first_letter,
                                                                 second_letter);
                // Add current count for rms-bucket to l-border of next bucket
                l_border += sa_buckets.s_buckets[s_bucket_index];
                // Compute new relative right border for rms-bucket
                rms_relative_count += sa_buckets.s_buckets[s_bucket_index];
                // Set new relative right border for rms-bucket
                sa_buckets.s_buckets[s_bucket_index] = rms_relative_count;
            }
        }
    }

    inline static void incude_S_suffixes(util::string_span input,
                                         util::span<bool> suffix_types,
                                         buckets buckets,
                                         util::span<sa_index> sa,
                                         util::character max_character) {
        // bit mask: 1000...000
        constexpr sa_index NEGATIVE_MASK = size_t(1)
                                           << (sizeof(sa_index) * 8 - 1);

        for (util::character c0 = max_character; c0 > '\1'; --c0) {
            // c1 = c0 - 1
            // start at rightmost position of L-bucket of c1
            size_t interval_start = buckets.l_buckets[c0] - 1;
            // end at RMS-bucket[c1, c1 + 1]
            size_t interval_end =
                buckets.s_buckets[buckets.get_rms_bucket_index(c0 - 1, c0)];

            // induce positions for each suffix in range
            for (size_t i = interval_start; i >= interval_end; --i) {
                if ((sa[i] & NEGATIVE_MASK) == 0) {
                    // entry is not negative -> induce predecessor

                    // insert suffix i-1 at rightmost free index of
                    // associated S-bucket
                    size_t destination_bucket = buckets.get_s_bucket_index(
                        input[sa[i] - 1], input[sa[i]]);
                    sa[buckets.s_buckets[destination_bucket]--] = i - 1;
                }

                // toggle flag
                sa[i] ^= NEGATIVE_MASK;
            }
        }

        // "$" is the first index
        sa[0] = input.size() - 1;

        // if predecessor is S-suffix
        if (input[input.size() - 2] < input[input.size() - 1]) {
            sa[0] |= NEGATIVE_MASK;
        }
    }

    inline static void induce_L_suffixes(util::string_span input,
                                         util::span<bool> suffix_types,
                                         buckets buckets,
                                         util::span<sa_index> sa,
                                         util::character max_character) {
        // bit mask: 1000...000
        constexpr sa_index NEGATIVE_MASK = size_t(1)
                                           << (sizeof(sa_index) * 8 - 1);

        for (size_t i = 0; i < sa.size(); ++i) {
            if (sa[i] & NEGATIVE_MASK > 0) {
                // entry is negative: sa[i-1] already induced -> remove flag
                sa[i] ^= NEGATIVE_MASK;
            } else {
                // predecessor has yet to be induced
                size_t insert_position = buckets.l_buckets[input[sa[i] - 1]]++;
                sa[insert_position] = sa[i] - 1;
                if (input[sa[i] - 2] < input[sa[i] - 1]) {
                    // predecessor of induced index i S-suffix
                    sa[insert_position] |= NEGATIVE_MASK;
                }
            }
        }
    }
};
} // namespace sacabench::saca::divsufsort
