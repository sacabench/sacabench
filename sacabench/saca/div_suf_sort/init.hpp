#pragma once

#include <tuple>
#include <util/span.hpp>
#include <util/type_extraction.hpp>

namespace sacabench::div_suf_sort {

template <typename sa_index>
inline static void insert_into_buckets(rms_suffixes<sa_index>& rms_suf,
                                       buckets<sa_index>& bkts) {
    size_t current_index, relative_index, bucket_index;
    size_t first_letter, second_letter;
    size_t rms_count = rms_suf.absolute_indices.size();
    // Skip last rms-suffix in this loop
    for (size_t pos = rms_count - 1; 0 < pos; --pos) {
        // Retrieve index and first two characters for current rms-
        // suffix
        current_index = rms_suf.absolute_indices[pos - 1];
        first_letter = rms_suf.text[current_index];
        second_letter = rms_suf.text[current_index + 1];
        // Retrieve index for current bucket containing the bucket's
        // border.
        // TODO: Check wether new bucket borders are correct.
        bucket_index = bkts.get_rms_bucket_index(first_letter, second_letter);
        // May be that prefix operator not supported here
        relative_index = --bkts.s_buckets[bucket_index];
        // Set current suffix into correct "bucket" at beginning of sa
        // (i.e. into relative_indices)
        rms_suf.relative_indices[relative_index] = pos - 1;
    }
    current_index = rms_suf.absolute_indices[rms_count - 1];
    first_letter = rms_suf.text[current_index];
    second_letter = rms_suf.text[current_index + 1];
    // Retrieve index for current bucket containing the bucket's
    // border.
    // TODO: Check wether new bucket borders are correct.
    bucket_index = bkts.get_rms_bucket_index(first_letter, second_letter);

    // May be that prefix operator is not supported here
    relative_index = --bkts.s_buckets[bucket_index];
    // Sort last rms-suffix into correct bucket:
    rms_suf.relative_indices[relative_index] = rms_count - 1;
}

template <typename sa_index>
inline static util::container<std::pair<sa_index, sa_index>>
extract_rms_substrings(rms_suffixes<sa_index>& rms_suf) {
    size_t rms_count = rms_suf.absolute_indices.size();
    // Create tupel for last rms-substring: from index of last
    // rms-suffix to index of sentinel
    std::pair<sa_index, sa_index> substring = std::make_pair(
        rms_suf.absolute_indices[rms_suf.absolute_indices.size() - 1],
        rms_suf.text.size() - 1);
    auto substrings_container =
        util::make_container<std::pair<sa_index, sa_index>>(rms_count);
    // Insert last rms-substring at last position in container
    substrings_container[substrings_container.size() - 1] = substring;

    size_t substr_start, substr_end;
    for (size_t current_index = 0; current_index < rms_count - 1;
         ++current_index) {

        substr_start = rms_suf.absolute_indices[current_index];
        substr_end = rms_suf.absolute_indices[current_index + 1] + sa_index(1);
        // Create RMS-Substring for rms-suffix from suffix-index of rms
        // and starting index of following rms-suffix + 1
        substring = std::make_pair(substr_start, substr_end);
        substrings_container[current_index] = substring;
    }
    // Create substring for last rms-suffix from rms-suffix index to
    // sentinel
    substr_start = rms_suf.absolute_indices[rms_count - 1];
    substr_end = rms_suf.text.size() - 1;
    // Create RMS-Substring for rms-suffix from suffix-index of rms
    // and starting index of following rms-suffix + 1
    substring = std::make_pair(substr_start, substr_end);
    substrings_container[rms_count - 1] = substring;
    return substrings_container;
}

// TODO: Change in optimization-phase (while computing l/s-types,
// counting bucket sizes)
template <typename sa_index>
inline static size_t
extract_rms_suffixes_and_count_buckets(util::string_span text,
                     util::span<sa_index> out_sa, buckets<sa_index>& sa_buckets) {
    // First (right) index from interval of already found rms-suffixes
    // [rms_begin, rms_end)
    size_t right_border = out_sa.size(), bucket_index, first_letter,
        second_letter;
    // Prev initialized for sentinel
    bool prev = true, current;
    // Insert rms-suffixes from right to left
    for (size_t pos = text.size()-1; pos > 0; --pos) {
        current = util::get_type_rtl_dynamic(text, pos-1, prev);
        first_letter = text[pos-1];
        second_letter = text[pos];
        if (sa_types::is_rms_type(current, prev)) {
            // Adjust border to new entry (rms-suffix)
            DCHECK_NE(first_letter, second_letter);
            out_sa[--right_border] = pos-1;
            bucket_index = sa_buckets.get_rms_bucket_index(first_letter,
                second_letter);
            ++sa_buckets.s_buckets[bucket_index];
        } else if(current) {
            ++sa_buckets.l_buckets[first_letter];
        } else {
            bucket_index = sa_buckets.get_s_bucket_index(first_letter,
            second_letter);
            ++sa_buckets.s_buckets[bucket_index];
        }
        prev = current;
    }
    // Count of rms-suffixes
    return text.size() - right_border;
}


// TODO: Change in optimization-phase (while computing l/s-types,
// counting bucket sizes)
template <typename sa_index>
inline static size_t
extract_rms_suffixes(util::string_span text,
                     util::container<bool>& sa_types_container,
                     util::span<sa_index> out_sa) {
    DCHECK_EQ(text.size(), sa_types_container.size());
    // First (right) index from interval of already found rms-suffixes
    // [rms_begin, rms_end)
    size_t right_border = out_sa.size();
    // Insert rms-suffixes from right to left
    for (size_t current = text.size() - 1; current > 0; --current) {
        if (sa_types::is_rms_type(current - 1, sa_types_container)) {
            // Adjust border to new entry (rms-suffix)
            out_sa[--right_border] = current - 1;
        }
    }
    // Count of rms-suffixes
    return text.size() - right_border;
}

// Temporary function for suffix-types, until RTL-Extraction merged.

template <typename sa_index>
inline static void get_types_tmp(util::string_span text,
                                 util::container<bool>& types) {
    // Check wether given span has same size as text.
    DCHECK_EQ(text.size(), types.size());
    // Last index always l-type suffix
    types[text.size() - 1] = 1;
    for (size_t prev_pos = text.size() - 1; prev_pos > 0; --prev_pos) {

        if (text[prev_pos - 1] == text[prev_pos]) {
            types[prev_pos - 1] = types[prev_pos];
        } else {
            // S == 0, L == 1
            types[prev_pos - 1] = (text[prev_pos - 1] < text[prev_pos]) ? 0 : 1;
        }
    }
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
 *
 * OBSOLETE
 */
template <typename sa_index>
inline static void compute_buckets(util::string_span input,
                                   const size_t max_character_value,
                                   util::container<bool>& suffix_types,
                                   buckets<sa_index>& sa_buckets) {
    count_buckets(input, suffix_types, sa_buckets);
    prefix_sum(max_character_value, sa_buckets);
    std::cout << "Finished" << std::endl;
}

template <typename sa_index>
inline static void count_buckets(util::string_span input,
                                 util::container<bool>& suffix_types,
                                 buckets<sa_index>& sa_buckets) {
    size_t first_letter, second_letter;
    // Used for accessing buckets in sa_buckets.s_buckets
    size_t bucket_index;
    for (size_t current = 0; current < input.size(); ++current) {
        first_letter = input[current];
        if (suffix_types[current] == 1) {
            ++sa_buckets.l_buckets[first_letter];
        } else {
            // Indexing safe because last two indices are always l-type.
            DCHECK_LT(current, input.size() - 1);
            second_letter = input[current + 1];
            // Compute bucket_index regarding current suffix being either
            // s- or rms-type
            bucket_index = (sa_types::is_rms_type(current, suffix_types))
                               ? sa_buckets.get_rms_bucket_index(first_letter,
                                                                 second_letter)
                               : sa_buckets.get_s_bucket_index(first_letter,
                                                               second_letter);

            // Increase count for bucket at bucket_index by one.
            ++sa_buckets.s_buckets[bucket_index];
        }
    }
}

template <typename sa_index>
inline static void count_buckets(util::string_span input,
                                 buckets<sa_index>& sa_buckets) {
    size_t first_letter = input[0], second_letter;
    // Used for accessing buckets in sa_buckets.s_buckets
    size_t bucket_index;
    std::tuple<bool, int> current_tuple = util::get_type_ltr_dynamic(input, 0);
    size_t same_char_amount = std::get<1>(current_tuple);

    // implicitly set to 1 in prefix_sum
    // sa_buckets.l_buckets[0] = 1;

    // Types for current (pos-1) and next (pos) element
    bool current = std::get<0>(current_tuple), next = current, refresh_type = false;

    for(size_t pos=1; pos < input.size(); ++pos) {
        if(same_char_amount == 0 || (pos == 1 && same_char_amount == 1)) {
            current_tuple = util::get_type_ltr_dynamic(input, pos);
            next = std::get<0>(current_tuple);
            same_char_amount = std::get<1>(current_tuple);
            refresh_type = true;
        }
        // Decreasing here helps with readability of conditions with this variable
        --same_char_amount;
        if(current) {
            ++sa_buckets.l_buckets[first_letter];
        } else {
            // Doesn't hold if type has been refreshed in this iteration
            if(same_char_amount > 0 && !refresh_type) {
                // next character is the same -> never rms-type
                bucket_index = sa_buckets.get_s_bucket_index(first_letter, first_letter);
            } else {
                second_letter = input[pos];
                    std::cout << "s-type: " << first_letter << "," << second_letter << ","
                    << size_t(input[pos+1]) << ", " << next << std::endl;
                bucket_index = next ? sa_buckets.get_rms_bucket_index(first_letter, second_letter) :
                    sa_buckets.get_s_bucket_index(first_letter, second_letter);
                ++sa_buckets.s_buckets[bucket_index];
            }
        }

        // Correct spot to refresh. Earlier -> buckets increased incorrectly
        if(refresh_type) {
            refresh_type = false;
            current = next;
            first_letter = input[pos];
        }
    }

    /*
    for(size_t pos = 0; pos < input.size()-1; ++pos) {
        if(same_char_amount == 0) {
            std::cout << "retrieving next element" << std::endl;
            current_tuple = util::get_type_ltr_dynamic(input, pos);
            current = std::get<0>(current_tuple);
            same_char_amount = std::get<1>(current_tuple);
            first_letter = input[pos];
            std::cout << "index " << pos << ", char " << first_letter
            << ", l_type " << current << ", same_chars " << same_char_amount <<
            std::endl;
        } else {
            // Types stay the same (prev refreshed in last iteration, where
            // same_char_amount was set)
            DCHECK_EQ(first_letter, input[pos]);
        }
        if(current) {++sa_buckets.l_buckets[first_letter];}
        else {
            second_letter = input[pos+1];
                std::cout << "s-type: " << first_letter << "," << second_letter << ","
                << size_t(input[pos+2]) << ", " << (input[pos+2] < second_letter) << std::endl;
            bucket_index = (input[pos+2] < second_letter) ?
                sa_buckets.get_rms_bucket_index(first_letter, second_letter) :
                sa_buckets.get_s_bucket_index(first_letter, second_letter);
            ++sa_buckets.s_buckets[bucket_index];
        }
        --same_char_amount;
    } */
    std::cout << "Finished" << std::endl;
}


template <typename sa_index>
inline static void prefix_sum(const size_t max_character_value,
                              buckets<sa_index>& sa_buckets) {
    // l_count starts at one because of sentinel (skipped in loop)
    size_t l_count = 1, rms_relative_count = 0, l_border = 0, s_bucket_index;
    // Adjust left border for first l-bucket (sentinel)
    sa_buckets.l_buckets[0] = 0;

    for (size_t first_letter = 1; first_letter < max_character_value + 1;
         ++first_letter) {
        // New left border completely computed (see inner loop)
        l_border += l_count;
        // Save count for current l-bucket for l_border of next l-bucket
        l_count = sa_buckets.l_buckets[first_letter];
        // DCHECK_LE(l_border+l_count, input.size());
        // Set left border of current bucket
        sa_buckets.l_buckets[first_letter] = l_border;
        for (size_t second_letter = first_letter;
             second_letter < max_character_value + 1; ++second_letter) {
            // Compute index for current s-bucket in s_buckets
            s_bucket_index =
                sa_buckets.get_s_bucket_index(first_letter, second_letter);
            // Add count for s-bucket to left-border of following l-bucket
            l_border += sa_buckets.s_buckets[s_bucket_index];
            // (c0,c0) buckets can be skipped for rms-buckets (because they
            // don't exist)
            if (first_letter != second_letter) {

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
}

} // namespace sacabench::div_suf_sort
